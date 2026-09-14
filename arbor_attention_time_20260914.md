# 学習時 attention 16% の内訳と local attention の SDPA backend 修正 (2026-09-14)

branch: `feat/local-attn-efficient-sdpa`。前提は `arbor_gpu_idle_root_cause_20260912.md`
の「attention 382 ms/update (16%)」。FLOPs がほぼゼロの区間がなぜ 16% を占めるのかを
nsys で分解し、local 側の原因を修正した。

## 計測

- Nsight Systems 2025.3 (`scripts/profile_training_nsys.sh`、wait 25 / active 2)、
  `configs/arbor.yaml` (926.8M / 8k / micro2、batch warmup 中なので accum 8)、
  入力 `data/random_bytes.bin`。カーネル名でカテゴリ集計 (2 update 平均)。
- 単体 microbench は同形状の SDPA を CUDA event で計測 (fwd+bwd、warmup 5 / 50 回)。

## 事実確認: attention は 92.2 ms/update (15.9%) = accum 32 換算 369 ms

| 区分 | ms/update (accum 8) | 比率 |
|---|---:|---:|
| GPU busy 全体 | 578.7 | 100% |
| local attention (flash、3 層) | 50.0 | 8.6% |
| global attention (flex、20 層) | 42.2 | 7.3% |

## local: flash の 128 行 tile を 16 行にしか使えていない

local attention は `(B·K=1024, 12 head, 16 token, 64)` の causal SDPA。SDPA の backend
選択は固定優先度 `[FLASH, EFFICIENT, MATH, CUDNN]` (`torch._C._get_sdp_priority_order()`)
で制約を満たす最初のものを取り、形状は見ない。flash の tile は fwd 128 行 / bwd 128 列
固定なので、系列長 16 では各 CTA (grid (1,1024,12) = 12,288 個) が tile の 1/8 しか使わない。

1 層 fwd+bwd のカーネル:

| kernel | µs |
|---|---:|
| `flash_fwd_kernel<64,128,128>` | 397 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel` | 822 |
| `flash_bwd_convert_dq` (fp32→bf16 変換のみ) | 398 |
| `flash_bwd_dot_do_o` (rowsum(dO∘O) のみ) | 380 |

補助カーネルまで 400 µs (帯域の 1/10) なのが tile 空振りの証拠。

単体 microbench (同形状、fwd+bwd、ms/層):

| 実装 | ms | vs flash |
|---|---:|---:|
| sdpa flash (旧既定) | 2.28 | 1.0x |
| sdpa cudnn | 1.76 | 1.3x |
| compile した bmm+softmax | 0.84 | 2.7x |
| **sdpa efficient** (`fmha_cutlassF/B_bf16_aligned_64x64`, fwd/bwd 各 1 kernel) | **0.81** | **2.8x** |
| メモリ下限 (専用 kernel の理論値) | ~0.3 | ~7x |

flash / efficient の交点 (head_dim 64、同 token 数): T=16 2.69x、32 1.89x、64 1.56x、
128 1.44x、256 1.20x、512 0.99x。

### 修正

`src/model/arbor.py` `Attention.forward` の素の causal 経路で、CUDA かつ T ≤ 256
(`_SHORT_SEQ_EFFICIENT_SDPA_MAX`) かつ native GQA でないとき
`sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` を明示する。efficient は `enable_gqa`
非対応 (kernel 無しエラー) なので GQA は既定に任せる。global (flex) / 動的 patching /
kv_cache 経路は変更なし。テスト `test_short_seq_attention_uses_efficient_sdpa`
(eager / compile、kernel 名と flash との値一致)。

学習 benchmark (bench config = arbor.yaml + random_bytes、micro2 / accum 8、step 60/80):

| | bytes/s | step ms | fwd ms | bwd ms | peak reserved |
|---|---:|---:|---:|---:|---:|
| flash (旧) | 220,365 / 219,884 | 588.4 / 589.3 | 131.1 | 375.0 | 14.25 GiB |
| efficient (新) | 233,133 / 233,617 | 552.5 / 554.6 | 125.3 | 346.0 | 14.07 GiB |

**+6.1%** (−35 ms/update、予測 −32 ms)。loss / ema は step 80 まで 1e-4 桁で一致
(厳密 attention 同士、bf16 加算順の差のみ)。

## global: latency bound だが micro_batch 増では回収できない

flex は fwd 41 µs + bwd 217 µs = 0.26 ms/層、bwd grid (40,2,4)=320 CTA で latency
bound (bwd/fwd 5.3 倍)。単体では B=2→8 で 2 seq あたり 0.98→0.16 ms (6x) だが、これは
CUDA Graph 外の CPU launch 分の償却が大半で、in-model では出ない。

同 bytes/update で micro 2/4/8 を実測:

| | micro2 (accum 8) | micro4 (accum 4) | micro8 (accum 2) |
|---|---:|---:|---|
| bytes/s | 220.4k | 219.6k | OOM (optimizer state 確保で WSL "device not ready") |
| peak reserved | 14.25 GiB | 17.67 GiB | — |

nsys 内訳 (ms/update): flex 42.2→30.9 (償却 27%)、grad 累積 59.4→30.3 は期待通りだが、
**packed GEMM 160.8→189.9** (M が倍になり autotune の winner が探索空間の境界に張り付く
警告) と inductor/quant +15 で相殺され、合計 578.7→580.3 で差なし。micro_batch 増は
packed GEMM の大 M tuning を直さない限り効かない。

## 残り

- local 専用 kernel (16×16 をレジスタで完結) なら efficient からさらに ~2x (−15 ms/update)。
- global は kernel 側の余地が小さい (0.26 ms/層)。micro 増と組み合わせるなら packed GEMM
  の M=2048 候補 tile を先に。
