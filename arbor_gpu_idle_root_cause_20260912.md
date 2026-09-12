# 学習時GPU idleの根本原因調査と修正 (2026-09-12)

branch: `fix/gpu-idle-root-cause`。micro-batchやgrad accumulationは変更せず、
GPUが待たされる原因そのものを特定して除去した。RTX 4090固有のtile調整は
行わず、CUDA Graph / kernel launchの一般的な性質に基づく修正だけを入れた。

## 計測方法

- Nsight Systems 2025.3.2 (`scripts/profile_training_nsys.sh`、CUDA/NVTX trace、
  graph node trace)。25 update待って26〜27 update目をcapture。
- 集計はsqlite exportから kernel/memcpy/memset のunionを「GPU busy」、
  それ以外を「idle」とし、NVTX区間 (`arbor.forward` / `arbor.backward` /
  `arbor.optimizer`) で分解した。1 update = micro 32回 + optimizer。
- 学習速度は `configs/arbor.yaml` と同じ926.8M / 8k / micro2 / accum32、
  入力を `data/random_bytes.bin` に固定した `--benchmark-steps 60` で、
  step 50/60 (phase=steady) を比較した。前回値は
  `logs/benchmark_arbor-adamw-triton_20260912-172654_81057.jsonl`。
- CUDA Graph再captureの理由は `TORCH_LOGS=+cudagraphs` で確認した。

## 修正前のidle内訳 (1 updateあたり、trace込み)

| 区間 | wall | GPU busy | idle |
|---|---:|---:|---:|
| micro 0 (optimizer直後) | 510.8 ms | 71.0 ms | **439.9 ms (86%)** |
| micro 1〜31 | 2146.3 ms | 2102.1 ms | 44.1 ms (2.1%) |
| optimizer区間 | 292.2 ms | 222.4 ms | **69.7 ms (24%)** |
| 合計 | 2950.4 ms | 2395.6 ms | 554.8 ms (18.8%) |

CUDA API traceには2 updateで `cudaStreamBeginCapture` / `EndCapture` /
`cudaGraphInstantiateWithFlags` / `cudaGraphDestroy` が各4回あり、update
ごとにforwardとbackwardのgraphが再captureされていた。

## 原因1: low-bit cache bufferの差し替えによるCUDA Graph再capture

`BitLinear.refresh_training_weight_cache` と `BitLinearGroup` 版は、optimizer
stepごとに `_train_w_packed` / `_train_w_packed_t` / `_train_w_scale` を
**新しいtensorで置き換えて**いた。これらはmodule bufferとしてcompiled
forward/backwardのstatic inputになり、CUDA Graphはcapture時のdevice
addressをgraphに焼き込む。addressが変わるとCUDAGraph Treesは
`static input data pointer changed` を理由にforward/backwardを再captureし、
その間 (capture + instantiate + `cudaDeviceSynchronize`) GPUはほぼ何も
実行しない。`TORCH_LOGS=+cudagraphs` の出力で `primals_5` 等の再capture理由が
`TernaryBitLinearSTE.apply` に渡るcache bufferであることを確認した。

さらにPyTorchは想定外の再captureを関数ごとに
`cudagraph_unexpected_rerecord_limit=128` 回まで許容し、超えるとその関数の
CUDA Graphを永続的に無効化してeager実行に落とす。長時間学習ではこれも
起こり得る挙動だった。

修正: `_refresh_cache_tensor` を導入し、既存bufferとshape/dtype/deviceが
一致する限り `copy_` でin-place更新する (初回とshape変更時だけ新規確保)。
これでaddressが安定し、再captureは起きない。

## 原因2: cache再生成のlaunch律速

再生成は1層あたり absmean → 除算 → round → clamp → int8化 → zero pad → +1 →
uint8化 → shift/or → transpose copy を2 layout分、個別kernelとして発行して
いた (optimizer区間のkernel 3,878個/updateのうち約3,600個)。各kernelは数µsで
終わるため、GPUはCPUのlaunchを待つ状態になり、optimizer区間のidleの主因
だった。これはGPU機種に依らないlaunch overheadの問題である。

修正: `src/model/ternary_pack.py` に、shadow weightのtileを1回loadして
forward用 (k方向4値/byte) とdX用 (n方向4値/byte) の両layoutとrow scaleを
同時に書くTriton kernelを追加した。1層あたり absmean (PyTorch、3 launch) +
1 kernelになる。数値は純PyTorch経路とbit一致させる:

- absmean scaleは従来どおりPyTorchでweight dtypeのまま計算 (reduction順序差
  を持ち込まない)。
- 除算は `libdevice.div_rn` (IEEE RN) で行い、BF16/FP16 weightではPyTorchの
  二項演算と同じく結果をweight dtypeへ丸めてから `rint`。
- pad位置は従来と同じくzero weight (code 1)。

`tests/test_ternary_pack.py` で6 shape × 3 dtype × 2 backend、量子化境界
(±0.5·scale, ±1.5·scale) を多く含む入力で `torch.equal` を確認した。
`BitLinearGroup` はmember行数が4の倍数のときだけmember単位でslice書き込み
し、それ以外は従来の連結→pack経路 (in-place copy) を使う。CPUやTriton無し
環境も従来経路のまま。

## 修正後

| 区間 | wall | GPU busy | idle |
|---|---:|---:|---:|
| micro 0 | 20.9 ms | 18.5 ms | 2.5 ms |
| micro 1〜31 | 2182.3 ms | 2140.3 ms | 42.0 ms (1.9%) |
| optimizer区間 | 184.6 ms | 179.3 ms | 5.4 ms (2.9%) |
| 合計 | 2388.8 ms | 2338.1 ms | **50.8 ms (2.1%)** |

CUDA API traceのgraph関連は `cudaGraphLaunch` 128回のみ (re-capture 0)。

学習ベンチマーク (step 50/60, steady):

| | step ms | opt_ms | bytes/s |
|---|---:|---:|---:|
| 修正前 (B2) | 2893 / 2920 | 113 | 181,193 / 179,520 |
| 原因1のみ修正 | 2439 / 2430 | 107 | 214,948 / 213,857 |
| 原因1+2修正 | 2370 / 2376 | 41 | 221,162 / 218,509 |

step time **-18.4%**、throughput **+22%**。loss / EMAは全log点
(step 10〜60) で修正前と完全一致しており、数値は変わっていない。
peak memory (`benchmark_cuda_memory`) は修正前B run 14.14 / 14.48 GiB
(allocated / reserved) → 原因1のみ修正 10.92 / 14.35 GiB → 原因1+2修正
10.83 / 14.25 GiB。再capture時に旧graphのprivate poolと新graphが同時に
生きていた分のallocated peakが無くなった。

## 残るidle (対象外)

micro-stepごとに約1.2 ms (1.9%) の空白があり、内訳はbackward graph replay後に
graph外で走る parameter grad累積 (260 tensor分のeager add、各数µs) と
loss計算のlaunch律速である。1 updateで約40 msと小さいため今回は手を付けない。
無くすには勾配累積そのものをcompiled backwardに含める構造変更が要る。

## 補足

- 調査中に `checkpoints/arbor2_1b_8k_lowbit/metrics.jsonl` のstep 340〜372で
  bytes/sが185k→33kに落ちているのを見つけたが、時刻は本調査のpytest実行
  (同じGPU) と重なっており、学習側の問題ではない。
- `arbor_training_fusion_20260912.md` 末尾の「Graph再記録とcache寿命」の
  仮説は本調査で因果まで確認・修正した。
- 再現: `scripts/profile_training_nsys.sh --config CONFIG --wait 25 --active 2
  --output OUT -- --benchmark-steps 27` の後、sqlite exportをNVTX区間で
  分解する (本文の表はこの方法)。

## 修正後のGPU時間内訳 (どこが律速か)

修正後trace (update 1、wall 2410 ms、kernel合計 2360 ms) をkernel名で集計:

| 種別 | GPU ms/update | 比率 |
|---|---:|---:|
| packed ternary GEMM (`_packed_bitlinear_kernel`, forward 311 + dX 355) | 666 | 28% |
| attention (flash fwd/bwd 42+166、flex fwd/bwd 34+140) | 382 | 16% |
| dW FP8 GEMM (`sm89_xmma_gemm_e4m3…`) | 277 | 12% |
| parameter grad累積 (`CUDAFunctor_add<BFloat16>` 260 tensor × 32 micro) | 247 | 10% |
| A8量子化 / FP8 cast・transpose (`_a8_quantize_*`, `_fp8_cast_transpose`, `_a8_dequant_fp8_cast_transpose`) | 217 | 9% |
| RMSNorm / gated FFN / その他 Inductor fused kernel | ~540 | 23% |
| AdamW (`_adamw_kernel`) | 29 | 1% |
| cache再生成 (`_ternary_pack_dual_kernel` + absmean) | 2 | <0.1% |

idleは2.1%なので、残りはGPU計算そのものの時間になった。特筆点は
grad累積の247 ms: BF16 `.grad += dW` は1 micro-stepごとに1.73 GiB×3の
メモリ往復 (約7.7 ms) で、micro 32回分が積み上がる。これはCUDA Graph外の
eager addで、無くすにはdW GEMMのepilogueで累積 (beta=1) するか、累積を
compiled backwardに含める構造変更が要る。

## VRAM内訳と「2bit表現」の実態

`configs/arbor.yaml` の構成で実測 (allocated):

| 項目 | GiB | B/param |
|---|---:|---:|
| shadow weight BF16 (926.8M param、うちBitLinear 923.0M) | 1.73 | 2 |
| parameter grad BF16 (CUDA Graph用固定buffer) | 1.73 | 2 |
| AdamW moment FP32 ×2 (`optim.state_precision: fp32`) | 6.91 | 8 |
| low-bit cache (packed 2bit × 2 layout + row scale) | 0.43 | 0.5 |
| 合計 (静的) | 10.79 | 12.5 |

benchmarkの `peak_allocated` 10.83 GiBとほぼ一致し、micro2でのactivationは
ごく小さい。`peak_reserved` 14.25 GiBとの差はCUDA Graph private poolと
allocatorの余裕分。

BitLinearの2bit表現は実装どおりである: `_train_w_packed` [K/4, N] と
`_train_w_packed_t` [N/4, K] はuint8で4 weight/byte (例: 768×768×3のQKV group
→ packed (192, 2304)、packed_t (576, 768))、cache合計は0.43 GiB = 2 layout ×
2 bit/weight。forward/dX GEMMはこのpacked bufferをkernel内でdecodeして使う。

ただし学習中のVRAMの96%は2bit表現ではなくSTE学習に必要な高精度状態
(BF16 latent weight + BF16 grad + FP32 moment = 12 B/param) で占められて
いる。BitNetの1.58bit重みはlatent weightの量子化viewであり、latent weight・
勾配・optimizer stateまで2bit化する仕組みではない。VRAMを下げる余地は
主に `optim.state_precision: int8` (moment 8→2 B/param、約5.2 GiB削減。
READMEにあるとおり既定fp32は安定性・速度優先の選択) にあり、2bit cacheや
packed kernel側にはない。
