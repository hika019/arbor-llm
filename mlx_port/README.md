# Arbor MLX port (Apple Silicon)

`src/model/arbor.py` (PyTorch/MPS) の static patching + BitNet b1.58 部分を
Apple Silicon の MLX (Metal GPU) へ移植したもの。**Mac 上での学習スループット
比較が目的**で、学習パイプライン全体 (HF streaming / checkpoint / 動的 patching /
entropy 等) は移植していない。

## 含むもの
- `arbor_mlx.py`: ArborMLX (static patching)。BitLinear b1.58 (W1.58 absmean
  ternary / A8 int8 fake-quant / detach-STE / SubLN / ReLU^2 gated FFN)、
  GQA + RoPE (global/local 別 theta)、文書境界 block-diagonal マスク (#2) を含む。
- `bench_mlx.py`: 合成データで fwd+bwd+optimizer の壁時計を実測 (MLX は遅延評価
  なので各 iter で mx.eval して測る)。

## 使い方
```bash
pip install mlx
python -m mlx_port.bench_mlx --seq 2048 --micro-batch 4 --grad-accum 16 --dtype bfloat16
```

## 実測 (959.1M params, ctx=2048, bf16, activations int8 fake-quant)
合成データ。同一 Mac で PyTorch/MPS 本走 (ctx2048 / gradient checkpointing /
mb1) が定常 ~1050 bytes/s だったのと比較する。

| 構成 | per opt-step | bytes/s | 1000 step |
|---|---|---|---|
| MLX mb1  ga1  | ~1.99 s | ~1032 | 0.55 h (実効batch1) |
| MLX mb2  ga1  | ~3.20 s | ~1280 | 0.89 h (実効batch2) |
| MLX mb4  ga1  | ~5.52 s | ~1484 | 1.53 h (実効batch4) |
| MLX mb8  ga1  | ~23.0 s | ~713  | メモリ逼迫で低下 |
| MLX mb4  ga16 (実効batch64) | ~116 s | ~1132 | ~32 h |
| PyTorch/MPS mb1(ckpt) 実効batch64 | ~125 s | ~1050 | ~35 h |

## 結論
- MLX は同モデルで MPS と同等〜微速 (実効batch64 で ~10%、小batch で最大 ~1.4x)。
  **劇的な高速化ではない**。ボトルネックは 959M BitNet の GPU 計算と unified memory。
- gradient accumulation は accumulator を毎 micro `mx.eval` しないと遅延グラフが
  溜まりメモリ逼迫で激減する (68h -> 32h に改善した)。
- 200k step 本走は Mac では依然非現実的 (~267 日)。本学習は CUDA 前提。
- Neural Engine (ANE) は PyTorch/MLX の学習からは使えない (両者とも Metal GPU)。
