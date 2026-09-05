# Arbor Low-Bit Training Kernel 実装方針・設計

## 1. 目的

Arbor / BitLinear の学習速度を上げるため、BitNet b1.58 系の ternary weight 特性を **forward だけでなく backward にも利用する**。

対象は主に BitLinear の以下3計算。

\[
Y = X W_q^T
\]

\[
dX = dY W_q
\]

\[
dW = dY^T X
\]

ここで、

- \(X\): activation
- \(W\): BF16 shadow weight
- \(W_q\): ternary quantized weight, \(W_q \in \{-1,0,+1\}\)
- \(Y\): BitLinear output
- \(dY\): 後段から来る gradient
- \(dX\): 前段へ返す gradient
- \(dW\): shadow weight 更新用 gradient

とする。

本設計の基本方針は以下。

1. **forward** は packed ternary weight 専用 GEMM
2. **backward dX** も同じ packed ternary weight 専用 GEMM
3. **backward dW** は ternary を利用できないため、activation / gradient を low-bit 化した dense GEMM
4. Shadow Weight は引き続き BF16 で保持
5. Python は orchestration に残し、主要演算のみ Triton / CUDA / CUTLASS に移す

---

## 2. 現在の前提

### Shadow Weight

学習対象の元の重み。

```text
W_shadow : BF16
```

optimizer が更新するのはこれ。

forward / dX 用には、毎回または cache 更新時に ternary 化する。

\[
W_{shadow} \rightarrow Q(W) \rightarrow W_q \in \{-1,0,+1\}
\]

---

### Ternary Weight

値として必要なのは3状態だけ。

```text
-1
 0
+1
```

したがって情報量は理論上、

\[
\log_2 3 \approx 1.585\text{ bit}
\]

実装上は 2bit / weight で十分。

例:

```text
00 =  0
01 = +1
10 = -1
11 = reserved
```

1 byte に4 weightを格納できる。

```text
uint8 packed = [w0][w1][w2][w3]
                 2b  2b  2b  2b
```

---

## 3. なぜ通常の INT8 GEMM だけでは不十分か

現在の一般的な INT8 GEMM は、

\[
INT8 \times INT8 \rightarrow INT32\ accumulator
\]

を前提にしている。

しかし Arbor の weight は、

\[
w_i \in \{-1,0,+1\}
\]

なので、実際の積は

\[
x_i w_i =
\begin{cases}
+x_i & w_i=+1 \\
0 & w_i=0 \\
-x_i & w_i=-1
\end{cases}
\]

である。

つまり **乗算そのものは不要**。

必要なのは、

```text
+X
0
-X
```

の選択と加算のみ。

したがって、通常の INT8×INT8 Tensor Core GEMM は計算資源を過剰に使っている可能性がある。

本設計では、weight を INT8 に展開せず、**2bit packed のまま読み、ADD / SUB / SKIP として処理する専用 kernel** を目標とする。

---

# 4. Forward 設計

## 4.1 数式

\[
Y = XW_q^T
\]

\(W_q\) は ternary。

各 output element は、

\[
y_{m,n} = \sum_{k=1}^{K} x_{m,k} w_{n,k}
\]

だが、ternary なので、

\[
y_{m,n}
= \sum_{k:w_{n,k}=+1}x_{m,k}
- \sum_{k:w_{n,k}=-1}x_{m,k}
\]

となる。

---

## 4.2 入力

```text
X          : A8 / INT8 または low-bit activation
W_packed   : 2bit ternary packed
scale_x    : per-token activation scale
scale_w    : ternary weight scale
```

---

## 4.3 Kernel の基本動作

```text
load activation tile
load packed 2bit weight tile
    ↓
weight code decode
    ↓
00 → skip
01 → +activation
10 → -activation
    ↓
partial accumulate
    ↓
scale_x * scale_w
    ↓
BF16 output
```

重要なのは、**W を INT8 に unpack してから GEMM しないこと**。

packed 状態のまま tile 内で decode し、register / shared memory 上で ADD/SUB に変換する。

---

## 4.4 Accumulator

activation を \([-127,127]\) とすると、1積の絶対値は最大127。

最悪値は、

\[
|y| \le 127K
\]

例えば、

```text
K=4096 → 520,192
```

であり、数学上は約20bit程度で足りる。

ただし RTX 4090 上に 20bit accumulator 専用演算器はないため、実装では原則 INT32 または FP32 accumulator を使用する。

狙うべき最適化は accumulator bit 幅の削減ではなく、

- weight bandwidth 削減
- unpack 回避
- multiply 回避
- kernel launch 削減
- tile reuse

である。

---

# 5. Backward dX 設計

## 5.1 数式

\[
dX = dY W_q
\]

ここでも \(W_q\) は同じ ternary weight。

したがって、forward と本質的に同じ構造。

```text
forward : X  × Wq
backward: dY × Wq
```

このため、forward 用 ternary GEMM kernel を **transpose / layout 対応だけ変えて再利用可能**。

---

## 5.2 設計

```text
dY
 │
 │ low-bit quantize
 ▼
dY_q
 │
 │ × packed ternary W
 ▼
custom ternary GEMM
 │
 ▼
dX
```

### 推奨 dtype

初期実装:

```text
dY_q      : INT8
W_packed  : 2bit ternary
acc       : INT32 / FP32
output dX : BF16
```

将来的には dY を INT4 / FP4 相当に落とす余地もあるが、まず INT8 で精度を確認する。

---

## 5.3 重要度

現在の学習ログでは backward が step time の主成分。

forward だけを高速化しても、全体速度向上は限定される。

したがって、

```text
forward ternary kernel
      +
dX ternary kernel
```

をセットで実装する。

---

# 6. Backward dW 設計

## 6.1 数式

\[
dW = dY^T X
\]

ここには ternary weight が存在しない。

入力は、

```text
dY : gradient
X  : activation
```

なので、forward / dX の ternary kernel は利用できない。

---

## 6.2 基本方針

activation と gradient を low-bit 化して、

\[
dW \approx Q(dY)^T Q(X)
\]

として計算する。

初期候補:

```text
X_q   : INT8
 dY_q : INT8
      ↓
INT8 GEMM
      ↓
INT32 accumulate
      ↓
scale restore
      ↓
BF16 dW
```

---

## 6.3 将来候補

RTX 4090 で性能が出るなら、

```text
dY : INT4 / packed 4bit
X  : INT8 or INT4
```

も検討する。

ただし CUDA hardware の実際の命令 throughput に合わせる。

単に bit 幅を小さくしても、unpack / conversion overhead が大きければ遅くなるため、必ず end-to-end benchmark で判断する。

---

# 7. 学習全体のデータフロー

```text
                  ┌────────────────────┐
                  │ Shadow Weight BF16 │
                  └─────────┬──────────┘
                            │
                    ternary quantize
                            │
                            ▼
                  ┌────────────────────┐
                  │ packed ternary W   │
                  │ 2 bit / weight     │
                  └──────┬───────┬─────┘
                         │       │
            FORWARD      │       │      BACKWARD dX
                         │       │
X BF16                   │       │                 dY BF16
 │                       │       │                    │
A8 quantize              │       │              low-bit quantize
 │                       │       │                    │
 ▼                       │       │                    ▼
X_q ────────────────×────┘       └────×──────────── dY_q
 │                              
 ▼                                       ▼
Y BF16                                 dX BF16
 │                                       │
 ▼                                       ▼
next layer                           previous layer


              BACKWARD dW

X_q / X BF16                 dY_q / dY BF16
       │                             │
       └──────────×──────────────────┘
                  │
                  ▼
                dW
                  │
                  ▼
             Optimizer
                  │
                  ▼
         Shadow Weight BF16
```

---

# 8. Kernel API 設計案

Python 側からは `torch.autograd.Function` または custom op として呼ぶ。

```python
class TernaryBitLinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shadow_weight, ...):
        ...

    @staticmethod
    def backward(ctx, grad_out):
        ...
```

内部 kernel は以下を想定。

```text
ternary_forward(
    x_q,
    packed_w,
    x_scale,
    w_scale
) -> y

ternary_dgrad(
    dy_q,
    packed_w,
    dy_scale,
    w_scale
) -> dx

lowbit_wgrad(
    dy_q,
    x_q,
    dy_scale,
    x_scale
) -> dw
```

---

# 9. Weight cache

Shadow Weight は optimizer step 後にのみ変わる。

そのため grad accumulation 中に毎回 ternary pack する必要はない。

```text
optimizer.step()
     │
     ▼
Shadow Weight 更新
     │
     ▼
ternary quantize
     │
     ▼
2bit pack
     │
     ▼
cache
```

その後の forward / dX は cache を使う。

```text
cached packed W
   ├─ forward
   └─ dX
```

これにより、

- quantization
- transpose
- packing

を micro-step ごとに繰り返さない。

---

# 10. 実装言語の優先順位

## Phase 1: Triton prototype

目的:

- algorithm の成立確認
- tile shape 探索
- packed ternary decode overhead 測定
- forward / dX の throughput 比較

まず Triton で実装する。

理由:

- 実装速度が速い
- tile / num_warps / num_stages の探索が容易
- PyTorch integration が容易

---

## Phase 2: CUDA / CUTLASS

Triton が cuBLAS INT8 / BF16 より優位になった場合のみ進む。

CUDA 側では、

- vectorized packed load
- warp-level decode
- shared memory reuse
- cp.async 相当の pipeline
- warp specialization
- layout tuning

を行う。

必要なら CUTLASS の GEMM pipeline をベースに、ternary weight operand の fetch / transform 部だけ custom 化する。

---

# 11. RTX 4090 向けの設計重点

RTX 4090 では「2bit ternary 専用 Tensor Core」は存在しない。

したがって、目標は新しい演算器を作ることではなく、既存 hardware 上で、

```text
memory traffic
unpack overhead
multiply overhead
kernel launch overhead
```

を削ること。

特に比較対象は以下。

```text
BF16 GEMM
INT8 GEMM
FP8 emulation / existing path
packed ternary custom kernel
```

custom kernel が少なくとも INT8 GEMM を超えなければ採用しない。

---

# 12. 精度方針

## Shadow Weight

```text
BF16 維持
```

理由:

- 学習対象の連続値
- ternary threshold を跨ぐ小さい更新を蓄積する必要がある
- FP32 master 追加は VRAM コストが大きい

---

## Forward Activation

初期:

```text
A8 / INT8
```

既存 Arbor の設計を維持。

---

## dY

初期:

```text
INT8 quantization
```

精度が維持できることを確認してから INT4 / FP4 相当を検討。

---

## dW

結果は Shadow Weight gradient として利用するため、

```text
accumulate: INT32 / FP32
output: BF16
```

を初期実装とする。

---

# 13. STE

ternary quantization は非微分なので STE を使用する。

概念的には、

```python
w_q = w + (quantize(w) - w).detach()
```

forward では ternary weight を使用するが、backward では Shadow Weight に gradient を返す。

custom autograd 実装でも、この意味を維持する。

---

# 14. 実装ステップ

## Step 1

既存 BitLinear の benchmark を固定する。

取得:

```text
forward ms
dX ms
dW ms
quantization ms
pack ms
optimizer ms
```

Nsight Systems / torch.profiler で kernel breakdown も取る。

---

## Step 2

2bit ternary pack format を固定する。

```text
4 weights / uint8
```

既存 `pack_ternary_weight` が利用できるなら形式を合わせる。

---

## Step 3

Triton `ternary_forward` を実装。

比較:

```text
existing BF16 fake-quant
existing INT8 path
existing FP8 path
new packed ternary path
```

---

## Step 4

同じ kernel core を使って `ternary_dgrad` を実装。

forward より dX の方を優先して最適化する。

理由:

現在 backward が training step の主要コストだから。

---

## Step 5

`lowbit_wgrad` を実装。

まずは、

```text
INT8 dY × INT8 X → INT32
```

を基準にする。

既存 cuBLAS / PyTorch INT8 GEMM が十分速いなら custom GEMM を書かない。

custom 化する場合も quantization + transpose + GEMM の融合を優先する。

---

## Step 6

3 kernel を custom autograd に統合。

```text
forward
  └─ ternary_forward

backward
  ├─ ternary_dgrad
  └─ lowbit_wgrad
```

---

## Step 7

full training benchmark。

最低比較項目:

```text
step_ms
fwd_ms
bwd_ms
bytes/s
GPU util
Tensor Core util
memory bandwidth
kernel launch count
VRAM usage
```

---

# 15. 採用基準

microbenchmark が速いだけでは採用しない。

最終判断は end-to-end training throughput。

最低条件:

```text
1. validation loss が既存経路と同等
2. NaN / divergence なし
3. full step が有意に高速
4. VRAM が悪化しない、または速度向上に見合う
```

目安として、

```text
< 5%  : 原則不採用
5-10% : 実装複雑性次第
10%+  : 採用候補
20%+  : 強く採用
```

---

# 16. 実装上の優先順位

```text
1. backward dX ternary kernel
2. forward ternary kernel
3. dW low-bit GEMM
4. quantization fusion
5. transpose/layout fusion
6. norm/residual/activation fusion
7. optimizer
8. CPU packing / DataLoader
```

ただし forward と dX は同一コアを共有する前提で同時開発する。

---

# 17. 最終目標

現状:

```text
Shadow BF16
    ↓
ternary fake quant
    ↓
汎用 BF16 / FP8 / INT8 GEMM
```

から、最終的に、

```text
                    Shadow Weight BF16
                           │
                           ▼
                    2bit packed ternary
                      ┌────┴────┐
                      │         │
                  forward      dX
                      │         │
                      ▼         ▼
             ternary custom GEMM

X / dY
  │
  ▼
low-bit quantization
  │
  ▼
dW low-bit dense GEMM
  │
  ▼
BF16 gradient
  │
  ▼
optimizer
  │
  ▼
Shadow Weight BF16
```

とする。

狙いは、単に dtype を小さくすることではない。

**Arbor の「weight が {-1,0,+1} しか取らない」という数学的構造を forward と backward dX の両方で直接利用し、通常の dense multiply を ADD / SUB / SKIP に置き換えること**が本質である。

