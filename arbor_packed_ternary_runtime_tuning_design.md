# Arbor Packed Ternary Runtime Tuning Design

対象ブランチ: `feat/packed-ternary-kmajor-layout`
基準コミット: `1140a479efe3e341e213d77147d8bbd1f0d75a27`
目的: packed ternary BitLinear の tile/backend 選択を、RTX 4090 や特定 shape に依存した手書き `if` から切り離し、GPU 非依存・再現可能・拡張可能な runtime autotune 設計へ移行する。

> **2026-09-11 status:** tuning data model、候補生成、fingerprint付き永続cache、設定、benchmark統合、legacy rollback、およびcache済みplanをraw Tritonで実行する`raw_plan`は実装済みである。CPU test、CUDAの4実行経路、autotune→cache→`raw_plan`の`torch.compile(fullgraph=True)` lifecycleも通過した。ただし性能受け入れは未完了である。旧commitのA比でB3は`bytes/s -7.14%`、同一worktreeのA2比でB1は`+2.17%`、B2は`+1.57%`だが、旧A/A2間に`-5.82%`のrun間差がある。また実学習B1はraw A2より速く、`custom_op`単独主因仮説は支持されない。機能完成と性能採用を分け、A-B-B-A交互順の再測とsteady-state timelineで非劣性が確認できるまで、既定は`dot_current + legacy_raw + tuning=off`とする。詳細は [arbor_packed_ternary_runtime_tuning_findings.md](arbor_packed_ternary_runtime_tuning_findings.md) を参照すること。

---

## 1. 背景

現在の packed ternary kernel は性能自体は十分高い。

特に `kmajor_single_dot` は以下の流れを実装済み。

```text
[K/4, N] packed ternary
  -> packed byte を coalesced load
  -> register 上で w0/w1/w2/w3 へ decode
  -> tl.join / tl.permute / tl.reshape
  -> [BLOCK_K, BLOCK_N] INT8 fragment
  -> tl.dot x1
```

RTX 4090 / `M=1024 K=2048 N=11264` の実測では、kernel と tile の選択だけで大きく性能が変わる。

```text
default policy
dot_current              0.895 ms
kmajor_current           0.513 ms
kmajor_single_dot        0.371 ms

forced tile = 128x64x64, warps=4
dot_current              0.456 ms
kmajor_current           0.223 ms
kmajor_single_dot        0.183 ms
```

dX でも同様。

```text
default
dot_current              0.990 ms
kmajor_single_dot        0.357 ms

forced 128x64x64
dot_current              0.513 ms
kmajor_single_dot        0.188 ms
```

つまり問題は「packed ternary が遅い」ことではない。

問題は、**良い kernel があっても production path が適切な launch config を選べないこと**にある。

現行 `_packed_linear_tile()` は特定 shape に対する実測値を `if` で持ち、未知 shape を generic `32x64x32` に落としている。この方式では、patch size、micro batch、GPU 世代、Triton codegen が変わるたびに性能 cliff が再発する。

この設計を廃止する。

---

# 2. 設計目標

実装後は次を満たすこと。

1. RTX 4090 / RTX 5090 / RTX PRO / 将来 GPU の SKU 名を kernel policy に直接書かない。
2. `(M, K, N)` ごとの最適 tile を手書き `if` で列挙しない。
3. kernel の数値意味と、hardware tuning policy を分離する。
4. 実行 GPU 上で候補を実測し、最良 config を選択する。
5. autotune 結果を cache し、学習中に毎 step 再探索しない。
6. forward / dX を特別扱いせず、実際の `(M,K,N)` を key として別 shape として扱う。
7. backend ごとに autotune できる。
8. GPU / Torch / Triton / kernel 実装が変わった場合、古い cache を誤用しない。
9. autotune の結果・使用 backend・tile を起動ログで確認できる。
10. 数値結果を一切変えない。
11. autotune 失敗時に暗黙で低速 backend に落ちない。
12. benchmark 用と training 用で同じ tuning policy を使用する。

---

# 3. 非目標

今回の変更では以下は行わない。

- Arbor の local/global モデル構造変更
- patching algorithm の変更
- BitNet の量子化仕様変更
- weight encoding の変更
- `kmajor_single_dot` decode algorithm の変更
- CUDA/CuTe/CUTLASS への全面移植
- FP8 dW policy の大規模再設計
- GPU SKU ごとの静的 tuning table 作成
- `if RTX4090 ...`, `if sm89 ... return tile` のような固定選択

architecture 情報は利用してよいが、用途は以下に限定する。

```text
1. illegal / unsupported config の除外
2. autotune cache namespace の分離
3. capability-dependent kernel availability 判定
```

architecture から最適 tile を直接決めてはいけない。

---

# 4. 現在の問題点

現在の `_packed_linear_tile()` は概ね次の責務を同時に持っている。

```text
shape classification
hardware tuning
backend tuning
fallback selection
historical benchmark knowledge
```

これは責務過多。

特に、

```python
if m == 2048:
    if k == 2048 and n == 11264:
        return ...
    ...
if m >= 32 and n >= 64:
    return 32, 64, 32, 4
```

のような設計では、`M=1024` が generic path に落ちる。

実際、`M=1024 K=2048 N=11264` では generic tile が大幅に遅いことが確認済み。

今後 `patch_size=8/16/32`、micro batch、sequence packing、別モデル幅などを試すたびに新しい M/K/N が発生するため、shape 列挙方式は維持不能。

---

# 5. 新しい責務分離

packed ternary 実行系を以下の層へ分ける。

```text
Packed Kernel
    |
    v
Kernel Backend Definition
    |
    v
Candidate Generator
    |
    v
Legality / Resource Filter
    |
    v
Autotuner
    |
    v
Persistent + Process Cache
    |
    v
Selected Launch Config
```

各層の責務を混ぜない。

---

# 6. Kernel Backend Definition

backend は「計算方法」だけを表す。

既存の以下は維持する。

```text
dot_current
kmajor_current
kmajor_single_dot
dot
```

将来的には enum/dataclass 化してよい。

例:

```python
@dataclass(frozen=True)
class PackedBackendSpec:
    name: str
    kmajor_layout: bool
    grouped_decode: bool
    decode_v2: bool
```

例:

```python
PACKED_BACKENDS = {
    "dot_current": PackedBackendSpec(
        name="dot_current",
        kmajor_layout=False,
        grouped_decode=False,
        decode_v2=False,
    ),
    "kmajor_current": PackedBackendSpec(
        name="kmajor_current",
        kmajor_layout=True,
        grouped_decode=False,
        decode_v2=False,
    ),
    "kmajor_single_dot": PackedBackendSpec(
        name="kmajor_single_dot",
        kmajor_layout=True,
        grouped_decode=False,
        decode_v2=True,
    ),
}
```

backend definition に tile を持たせない。

---

# 7. Launch Config

tile を明示的な value object にする。

```python
@dataclass(frozen=True)
class PackedLaunchConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 2
```

tuple の位置依存を減らす。

`num_stages` も Phase 2 の tuning 対象に含める。

---

# 8. Candidate Generator

最適値を返してはいけない。

**探索候補を返すだけ**にする。

例:

```python
def packed_launch_candidates(
    *,
    m: int,
    n: int,
    k: int,
    backend: str,
    device: DeviceInfo,
) -> tuple[PackedLaunchConfig, ...]:
    ...
```

候補は原則 GPU SKU 非依存。

初期候補は小さく始める。

```text
BM = 64, 128
BN = 64, 128
BK = 32, 64
warps = 4, 8
stages = 2, 3
```

ただし直積をそのまま全探索すると、shape ごとの初回 JIT compile 時間が
production の起動コストとして大きすぎる。初期実装では、M/N/K の大小と
warp 数を比較できる GPU SKU 非依存の structural template を 20 個程度に限定し、
そこへ conservative default を加える。

template は性能 winner の予想ではなく探索予算である。追加・削除は複数 GPU と
代表 shape の sweep に基づいて行い、特定 shape の答えを埋め込んではならない。
特に同じ最大 output tile では W4/W8 の両方を残し、warp 数だけの比較を可能にする。

---

# 9. Candidate Pruning

生成した structural template から「最適そうだから」という理由で config を
shape/GPU ごとに捨てない。

**template 生成後の pruning では、明確に不合理・非合法なものだけ除外**する。

例:

```text
decode_v2:
    BLOCK_K % 4 == 0
    BLOCK_K >= 16

tl.dot INT8:
    Tensor Core に適した K alignment を要求

BLOCK_M > M かつ無駄が極端:
    必要に応じて除外

shared memory limit 超過:
    除外

明らかな compile-time resource limit:
    除外
```

shape に応じた軽い pruning は許可する。

例えば `M < 64` なら `BM=128` を候補から外す、など。

ただし、

```python
if m == 1024 and k == 2048 and n == 11264:
    return [(128,64,64,4)]
```

のような答えの埋め込みは禁止。

---

# 10. GPU 情報の扱い

GPU 情報は `DeviceInfo` に集約する。

例:

```python
@dataclass(frozen=True)
class DeviceInfo:
    device_index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory: int
    multi_processor_count: int
```

必要なら以下も追加する。

```text
max_threads_per_multi_processor
shared_memory_per_block
warp_size
```

重要:

```text
GPU name / compute capability
        ↓
tile を直接決定
```

してはいけない。

正しい用途:

```text
GPU info
   |- unsupported candidate を除外
   |- autotune cache key を分離
   `- capability-dependent backend availability を判定
```

---

# 11. Autotune Key

autotune の実質 key は最低限、

```text
backend
M
K
N
output dtype
scale mode
```

とする。

forward / dX という文字列を key に含める必要はない。

forward:

```text
M=1024 K=2048 N=11264
```

dX:

```text
M=1024 K=11264 N=2048
```

で自然に別 key になるため。

必要なら kernel compile semantics に影響する flag を追加する。

```text
GROUPED_DECODE
K_MAJOR_LAYOUT
DECODE_V2
SCALE_PER_OUTPUT
```

---

# 12. Cache Key

persistent cache は、単純な `(M,K,N)` だけでは危険。

最低限以下を fingerprint に含める。

```text
kernel schema/version
backend
M
K
N
dtype
scale mode

GPU compute capability
GPU model/name or stable device identifier

torch version
CUDA runtime/build version
Triton version
```

例:

```json
{
  "schema": 1,
  "kernel_version": "packed_ternary_v2",
  "backend": "kmajor_single_dot",
  "shape": [1024, 2048, 11264],
  "dtype": "bfloat16",
  "scale_per_output": true,
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "cc": [8, 9]
  },
  "software": {
    "torch": "2.14.0+cu130",
    "triton": "...",
    "cuda": "13.0"
  }
}
```

kernel 実装が変わったら `kernel_version` を上げる。

古い cache を無理に migrate しない。

---

# 13. Cache Layer

二段階 cache にする。

```text
Process memory cache
        |
        v
Persistent disk cache
```

process cache は dict でよい。

persistent cache の配置例:

```text
~/.cache/arbor/packed_ternary_autotune.json
```

または XDG を尊重する。

```text
$XDG_CACHE_HOME/arbor/...
```

training checkpoint directory には置かない。

理由:

- model checkpoint と hardware tuning は別責務
- 同じ checkpoint を別 GPU で使う
- cache 削除が model state に影響してはいけない

---

# 14. Autotune 実行タイミング

autotune は hot training loop の途中で何度も走らせない。

基本:

```text
first occurrence of tuning key
        |
        +-- process cache hit -> use
        |
        +-- persistent hit -> validate/use
        |
        `-- miss -> benchmark candidates once
```

Arbor は shape がほぼ固定なので、通常は起動直後に少数 shape を tune すれば終わる。

可能なら training 開始前に model/config から代表 shape を列挙し、pre-tune するモードも用意する。

例:

```text
--bitlinear-autotune warmup
```

ただし最初の実装では lazy tuning でよい。

---

# 15. Autotune Benchmark Method

候補測定は CUDA Event を使う。

既存 benchmark と同等の測定方法を共通化する。

最低限:

```text
warmup: 10-30
measure: 30-100
metric: median
```

開発 benchmark では 500 iteration を使ってよいが、training startup autotune では重すぎる。

候補間の差が大きいので、runtime autotune は少数 iteration で十分。

推奨:

```text
warmup 10
measure 30
```

必要なら上位2候補だけ追加測定する two-stage tune を実装する。

例:

```text
全候補 10回
 -> 上位2候補
 -> 各50回
 -> final winner
```

これは startup time を抑えつつ noise を減らせる。

---

# 16. Synchronization

候補測定時だけ明示 synchronization を許可する。

通常 training path では既存どおり unnecessary sync を入れない。

autotune function の外へ synchronization を漏らさない。

---

# 17. torch.compile との境界

重要。

runtime autotune を `torch.compile` graph 内で動かさない。

推奨構造:

```text
Python dispatch
    |
    +-- resolve launch config
    |
    `-- Triton kernel launch
```

config が決まった後に compile される形を優先する。

もし custom autograd / Dynamo tracing と衝突する場合は、tuning resolver を `torch._dynamo.disable` 相当の境界へ隔離することを検討する。

ただし無条件に graph break を増やしてはいけない。

現行 compile 性能を必ず A/B する。

---

# 18. Triton @autotune を直接使うか

Triton `@autotune` を使ってもよいが、以下を満たせるか確認する。

```text
persistent cache の制御
kernel versioning
GPU/software fingerprint
benchmark logging
torch.compile compatibility
candidate pruning
failure reporting
```

これらが扱いづらければ Arbor 側に薄い autotune layer を持つ。

目的は「Triton autotuneを使うこと」ではない。

目的は、

```text
hardcoded shape -> tile mapping を消すこと
```

である。

---

# 19. Backend Selection と Tile Selection を分離

重要。

Phase 2 ではまず、

```text
backend は config で固定
tile は autotune
```

とする。

例:

```yaml
bitlinear_ternary_backend: kmajor_single_dot
bitlinear_ternary_tuning: auto
```

いきなり backend まで runtime benchmark で自動選択すると、原因切り分けが難しくなる。

まず backend 内 tile autotune を完成させる。

その後 optional Phase 3 として、

```text
dot_current
kmajor_current
kmajor_single_dot
```

の backend 自体を benchmark して選ぶ `backend=auto` を検討してよい。

---

# 20. 設定案

例:

```yaml
speed:
  bitlinear_ternary_backend: kmajor_single_dot

  bitlinear_ternary_tuning: auto
  bitlinear_ternary_tuning_cache: true
  bitlinear_ternary_tuning_cache_path: auto

  # debug / reproducibility only
  bitlinear_ternary_fixed_tile: null
```

choices:

```text
bitlinear_ternary_tuning:
  auto
  fixed
  off
```

意味:

```text
auto:
    cache -> autotune -> cache

fixed:
    fixed_tile を必須とする

off:
    conservative default config を使う
```

`off` を「4090で以前速かったtile」にしてはいけない。

`fixed` は benchmark / regression test 用。

---

# 21. Conservative Default

autotune 不使用時の default は、最速を狙わず安全性を優先する。

ただし現行 `32x64x32` が極端に遅いことがあるため、default の選定は別 benchmark で再検討する。

ここも GPU SKU 別 `if` にしない。

default はあくまで、

```text
compile可能
resource usage が過大でない
広い shape で破綻しない
```

ことを優先する。

---

# 22. Autotune Failure

autotune 中に候補 kernel が compile/runtime failure した場合:

```text
その candidate だけ失格
```

でよい。

全候補が失敗した場合:

```text
RuntimeError
```

とする。

暗黙で BF16 や別 backend へ fallback しない。

エラーには以下を出す。

```text
backend
shape
dtype
GPU
試した candidate
各 candidate の failure reason
```

---

# 23. Logging

training startup で必ず選択結果を見えるようにする。

cache hit:

```text
[bitlinear-tune] backend=kmajor_single_dot
  shape=1024x2048x11264
  dtype=bfloat16 scale=per_output
  tile=128x64x64 warps=4 stages=2
  source=cache
  gpu="NVIDIA GeForce RTX 4090" cc=8.9
```

新規 autotune:

```text
[bitlinear-tune] backend=kmajor_single_dot
  shape=1024x2048x11264
  dtype=bfloat16 scale=per_output
  candidates=21
  best=128x64x64 warps=4 stages=2
  median=0.183ms
  source=measured
```

verbose/debug mode では全候補を表示可能にする。

通常ログでは winner だけでよい。

---

# 24. Benchmark CLI

既存 `scripts.bench_bitlinear_kernels` を tuning system と共通化する。

追加候補:

```text
--backend kmajor_single_dot
--autotune
--show-candidates
--clear-tune-cache
--fixed-tile BM,BN,BK,WARPS,STAGES
```

benchmark script 内に独自 tile policy を複製しない。

production resolver と同じ candidate generator を import して使う。

---

# 25. 既存 `_packed_linear_tile()` の扱い

最終的に shape-specific `if` を削除する。

移行中は compatibility wrapper として残してよい。

最終形のイメージ:

```python
def _resolve_packed_launch_config(
    *,
    m: int,
    n: int,
    k: int,
    backend: str,
    dtype: torch.dtype,
    scale_per_output: bool,
) -> PackedLaunchConfig:
    key = make_tune_key(...)
    return tuner.resolve(key)
```

`_packed_linear()` は、

```python
config = _resolve_packed_launch_config(...)
```

だけを呼ぶ。

`_packed_linear()` 自身に shape tuning logic を入れない。

---

# 26. 推奨ファイル分割

`src/model/bitlinear.py` が肥大化しているため、tuning logic は別ファイルへ出すことを推奨する。

例:

```text
src/model/bitlinear.py
src/model/bitlinear_tuning.py
```

`bitlinear_tuning.py`:

```text
PackedLaunchConfig
DeviceInfo
TuneKey
candidate generation
candidate filtering
benchmark runner
memory cache
persistent cache
logging
```

kernel 本体は当面 `bitlinear.py` に残してよい。

循環 import に注意する。

---

# 27. 数値 correctness

autotune は性能だけを選ぶ機構なので、数値結果は tile に依存せず一致しなければならない。

テストでは少なくとも以下を比較する。

```text
int8_int_mm reference
dot_current
kmajor_current
kmajor_single_dot
```

代表 shape:

```text
unaligned small shape
M=1024 K=2048 N=11264
M=1024 K=11264 N=2048
M=2048 K=2048 N=11264
M=4096 K=2048 N=11264
M=8192 K=2048 N=11264
```

既存 tolerance を維持する。

---

# 28. Autotune Unit Tests

CPU で可能な部分は GPU なしでテストする。

必要:

```text
TuneKey equality/hash
cache serialization
kernel version mismatch invalidation
Torch version mismatch invalidation
GPU fingerprint mismatch invalidation
candidate pruning
fixed mode
cache disabled mode
malformed cache recovery
```

CUDA test:

```text
auto tune returns valid config
second call hits memory cache
persistent cache can be reloaded
selected config produces correct output
torch.compile forward/backward succeeds
```

---

# 29. Performance Regression Tests

CI の通常GPUに厳密ms thresholdを入れない。

GPU timing は環境差が大きい。

代わりに benchmark command を用意し、手動/専用GPU CIで以下を見る。

RTX 4090で既知の参考値:

```text
M=1024 K=2048 N=11264
kmajor_single_dot
128x64x64 W4

forward ~0.18 ms
dX      ~0.19 ms
```

これは correctness CI の hard threshold にはしない。

ただし autotuner が明らかな slow config を選んでいないか確認するため、

```text
selected <= 1.20 * measured_best
```

のような同一run内相対評価は専用benchmark testで可能。

---

# 30. 実学習受け入れ条件

最終判断は microbenchmark だけで行わない。

同一 machine / software / config で A/B する。

最低比較:

```text
A:
bitlinear_ternary_backend=dot_current
old fixed heuristic

B:
bitlinear_ternary_backend=kmajor_single_dot
new autotune
```

同じ:

```text
micro_batch
grad_accum
patch_size
patch_pooling
Torch
CUDA
driver
dataset
```

を維持する。

比較項目:

```text
bytes/s
step_ms
fwd_ms
bwd_ms
opt_ms
VRAM peak
loss/EMA
```

最低でも compile/warmup 後の steady 100 step 程度で比較する。

---

# 31. M=1024 は必須代表 shape

現在の `patch_size=16`, `micro_batch=2` では global token rows は、

```text
512 patches/seq * 2 = M=1024
```

となる。

したがって Phase 2 の tuning coverage に M=1024 を必ず入れる。

最低限:

```text
gate/up
1024,2048,11264

down
1024,5632,2048

QKV
1024,2048,3072

output
1024,2048,2048
```

さらに p8 / MB2 等で M=2048、MB4 等で M=4096、MB8 等で M=8192 が発生するため、それらも benchmark coverage に残す。

ただしこれらを `if` として実装してはいけない。

**benchmark coverage と production dispatch policy を混同しないこと。**

---

# 32. Forward / dX

forward と dX 用に別の if table を作ってはいけない。

以下は単に別 shape。

```text
forward:
[M, K] @ [K, N]

dX:
[M, N] @ [N, K]
```

autotuner の key が `(M,K,N)` なら自然に別 config が選択される。

将来 kernel implementation が forward/dX で分かれた場合のみ operation kind を key に追加する。

現状は不要。

---

# 33. Backend Auto は後段

将来的には、

```yaml
bitlinear_ternary_backend: auto
```

を実装してよい。

その場合は、

```text
candidate backend
    x
candidate launch config
```

を探索する。

ただし初回実装ではやらない。

まず `kmajor_single_dot` を固定 backend にして tile autotune を完成させる。

理由:

- 問題切り分けしやすい
- compile 数を抑えられる
- cache 設計を単純化できる
- backend algorithm と launch tuning の責務を分離できる

---

# 34. num_stages

現行 tile tuple に `num_stages` がないなら追加する。

`kmajor_single_dot` は、

```text
packed load
decode
fragment construction
MMA
```

を繰り返すため software pipeline の効果を受ける可能性がある。

候補はまず、

```text
2
3
```

程度でよい。

4以上は register/shared-resource pressure を確認してから追加する。

---

# 35. Nsight は autotune の代替ではない

Nsight Compute は、

```text
register spill
occupancy
Tensor Core utilization
memory throughput
```

の診断に使う。

tile selection を Nsight 指標から一意に決めようとしない。

最終性能は elapsed time で決める。

Nsight は、

```text
なぜその config が勝つ/負けるか
```

を理解するために使う。

WSL 等で NCU が使いづらくても autotune 設計は成立しなければならない。

---

# 36. 実装順序

## Phase A: tuning data model

追加:

```text
PackedLaunchConfig
TuneKey
DeviceInfo
cache schema
```

既存 kernel behavior は変えない。

---

## Phase B: candidate generator

shape-specific winner table を使わない candidate generation を追加。

既存 `_packed_linear_tile()` は一時的に残してよい。

---

## Phase C: runtime tuner

CUDA Event を使って candidate を測る。

memory cache を実装。

最初は persistent cache なしでもよい。

---

## Phase D: persistent cache

GPU/software/kernel fingerprint 付きで保存。

cache corruption は警告して再測定。

---

## Phase E: `_packed_linear()` 統合

`_packed_linear_tile()` 直接選択から resolver に切り替える。

fixed mode を残し benchmark/debug に使えるようにする。

---

## Phase F: benchmark 統合

benchmark script の tile candidate source を production tuner と共通化する。

重複した candidate/table を削除。

---

## Phase G: hardcoded table 削除

既存の以下を削除。

```text
M=2048 special cases
local shape special cases
generic shape winner assumptions
RTX4090 benchmark由来コメント
```

必要なら conservative fallback だけ残す。

---

## Phase H: kmajor_single_dot 実学習 A/B

autotune完成後にbackendを`kmajor_single_dot`にしてsteady throughputを比較する。1回目のA2/B1/B2/B3は完了したが、run間変動を超える優位性を示さなかった。A-B-B-Aの交互順で最低2 run/caseを取るまで性能合格としない。

---

# 37. 実装時の禁止事項

以下は禁止。

```python
if gpu_name == "RTX 4090":
    return ...

if compute_capability == (8, 9):
    return 128, 64, 64, 4

if m == 1024 and k == 2048 and n == 11264:
    return 128, 64, 64, 4
```

また benchmark script と production code に別々の tile table を持たせない。

autotune miss のたびに全候補を毎 step 測定しない。

cache key に GPU/software fingerprint を入れず再利用しない。

autotune failure を無言で BF16/fallback backend に逃がさない。

---

# 38. 推奨ログ例

起動時:

```text
[bitlinear] packed backend=kmajor_single_dot tuning=auto
[bitlinear-tune] gpu="NVIDIA GeForce RTX 4090" cc=8.9
[bitlinear-tune] cache=/home/.../.cache/arbor/packed_ternary_autotune.json
```

first tune:

```text
[bitlinear-tune] shape=1024x2048x11264 backend=kmajor_single_dot
[bitlinear-tune] candidates=8 source=measured
[bitlinear-tune] selected BM=128 BN=64 BK=64 warps=4 stages=2 median=0.183ms
```

cache hit:

```text
[bitlinear-tune] shape=1024x2048x11264 backend=kmajor_single_dot
[bitlinear-tune] selected BM=128 BN=64 BK=64 warps=4 stages=2 source=cache
```

---

# 39. 完了条件

機能完成と性能採用を分ける。上段の主要機能条件は満たしているが、下段の性能条件は未達である。

## 39.1 機能完成条件

- `_packed_linear_tile()` の shape-specific performance table が消えている。
- RTX 4090 / 5090 等の SKU 固有 tile mapping が存在しない。
- `kmajor_single_dot` が runtime autotune を使用できる。
- M=1024/2048/4096/8192 をコード変更なしで tune できる。
- forward/dX が shape key により独立して tune される。
- autotune は初回のみ実行され、その後 cache hit する。
- GPU/software/kernel version 違いで cache が分離される。
- benchmark と production が同じ candidate generator/resolver を使用する。
- numerical correctness test が通る。
- torch.compile forward/backward が通る。
- repoに導入済みのRuff、pytest、`git diff --check`が通る。型チェッカを導入する場合は別changeでCIと同時に追加する。
- autotune disabled/fixed mode のテストがある。
- 実学習で old `dot_current` baseline と A/B する。
- startup logからbackend、execution path、cache path、plan entry数が分かる。verbose時はshapeごとのtileとcache sourceが分かる。

## 39.2 性能採用条件

- `legacy_raw`、`legacy_custom_op`、`raw_plan`、`custom_op`をone-factor-at-a-timeで比較できる。
- 同一machine/configでA-B-B-A交互順の最低2 run/caseを取る。
- 各runの開始温度、P-state、SM/memory clockとsteady区間の分散を保存する。
- `bytes/s`の中央値がlegacyに非劣で、loss/EMAが固定seedで一致する。
- compileを含まないsteady-state timelineで、改善またはregressionの支配要因を説明できる。
- 上記合格後にのみproduction defaultを新経路へ変更する。

---

# 40. 最終的に目指す構造

```text
Arbor / BitLinear
      |
      v
packed_linear(M,K,N,backend)
      |
      v
TuneKey
      |
      +------> memory cache
      |             |
      |             v
      |         cache hit
      |
      +------> persistent cache
      |             |
      |             v
      |         cache hit
      |
      `------> candidate generator
                    |
                    v
              legality filter
                    |
                    v
              runtime benchmark
                    |
                    v
                winner
                    |
            +-------+-------+
            |               |
            v               v
       memory cache    persistent cache
            |
            v
      Triton kernel launch
```

kernel は「どう計算するか」だけを持つ。

tuner は「この環境でどう起動すると速いか」だけを持つ。

model architecture はそのどちらも知らない。

---

# Codex-Fugu への実装指示

この文書を仕様として扱い、`feat/packed-ternary-kmajor-layout` ブランチ上で実装すること。

最初に現在の `src/model/bitlinear.py`、`scripts/bench_bitlinear_kernels.py`、training config/test を読み、既存 `kmajor_single_dot` と training cache / torch.compile path を壊さないこと。

実装中に RTX 4090 の既知best tileを production codeへ直接埋め込んではいけない。既知値は benchmark/regression の参考値としてのみ使用する。

最初の目標は `kmajor_single_dot` の launch-config autotune であり、backend 自動選択までは広げない。

作業後は、変更点だけでなく以下を報告すること。

```text
1. 削除した hardcoded tuning policy
2. 新しい責務分割
3. autotune candidate 数と pruning 条件
4. cache key / invalidation 条件
5. M=1024/2048/4096/8192 の選択結果
6. correctness test 結果
7. torch.compile test 結果
8. Ruff / Pyright / diff-check
9. RTX 4090での microbenchmark A/B
10. 実学習 steady-state A/B
```

実測結果が期待より悪い場合は、速く見せるために固定 tile を追加して回避せず、その shape で autotuner が何を選んだか、候補ごとの timing を報告して原因を切り分けること。
