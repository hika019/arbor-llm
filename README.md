# arbor-llm

BLT (Byte Latent Transformer) を fork し、Global Latent Transformer に BitNet b1.58
(W1.58A8 ternary) を統合した、トークナイザ不要・1B 規模の LLM を RTX 4090 一枚で
ゼロ学習するプロジェクト。

詳細は [`bitnet_blt_project_spec.md`](./bitnet_blt_project_spec.md) を参照。

## セットアップ

```bash
# 1. venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools

# 2. PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 依存
pip install -r requirements.txt

# 4. flash-attn (toolchain 確認後)
pip install flash-attn --no-build-isolation

# 5. BLT は third_party/blt に clone 済み (upstream=facebookresearch/blt)
#    自分の fork を origin として設定:
#      ./scripts/setup_blt_fork.sh git@github.com:<you>/blt.git
```

## 学習

```bash
# 新規学習
python -m src.train.train --config configs/arbor_1b.yaml

# 最新 checkpoint から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume latest

# best loss から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume best
```

学習中 `Ctrl+C` (SIGINT) または `kill -TERM <pid>` で次 step 境界にて
安全保存して終了する。二度押しで強制終了。

## ディレクトリ

```
src/
  model/   BitLinear, Global Latent Transformer, Local Enc/Dec, 全体組立
  data/    バイト直 dataset, packing
  train/   train.py, checkpoint, signals, optim
  eval/    perplexity 他
configs/   arbor_1b.yaml 等
third_party/blt/  facebookresearch/blt の fork (submodule)
checkpoints/      既定の外部保存先 (.gitignore)
```
