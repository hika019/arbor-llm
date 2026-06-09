# arbor-llm

BLT (Byte Latent Transformer) を fork し、Global Latent Transformer に BitNet b1.58
(W1.58A8 ternary) 風の BitLinear を統合する実験プロジェクト。

現状の BitLinear は CUDA forward に Triton の packed ternary/int8 kernel を持つ。
CPU または Triton 不可の環境では PyTorch 参照実装へ fallback する。backward はまだ
STE の参照実装で、optimizer state や勾配まで含めた完全な fused BitNet training stack
ではない。1B 設定は目標設定であり、現在の smoke 検証は小モデル・短ステップで
学習ループと checkpoint を確認する段階です。

現時点ではコードを正とする。`bitnet_blt_project_spec.md` には目標・未実装項目も含まれるため、
実装済みの挙動は README と `src/` / `configs/` を優先して確認する。

`model.backend: blt` が既定。BLT 本体の import / 構築に失敗した場合、既定ではエラーにする。
stub は `model.backend: stub` を明示した smoke 用、または `model.allow_stub_fallback: true` を
明示した場合だけ使う。

詳細は [`bitnet_blt_project_spec.md`](./bitnet_blt_project_spec.md) を参照。

## セットアップ

```bash
# 0. OS パッケージ (Ubuntu / WSL)
sudo apt update
sudo apt install -y git python3 python3-venv python3-dev python3.12-dev build-essential ninja-build ripgrep

# CUDA forward / flash-attn を使う場合は nvcc も必要。
# nvidia-smi と nvcc --version で CUDA の major version が合うことを確認する。

# 1. venv (Python 3.10-3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools

# 2. PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 依存
pip install -r requirements.txt

# 4. flash-attn (任意; toolchain 確認後)
pip install flash-attn --no-build-isolation

# 5. BLT 本体
git clone https://github.com/facebookresearch/blt third_party/blt
git -C third_party/blt remote rename origin upstream

# 6. 自分の BLT fork を origin として設定する場合
./scripts/setup_blt_fork.sh git@github.com:<you>/blt.git
```

`third_party/blt` は親リポジトリでは submodule 化しておらず、`.gitignore` で無視している。
clone 済みかは `test -d third_party/blt/.git` で確認できる。既に clone 済みの場合は手順 5 を飛ばす。

`scripts/env.sh` を source すると `.venv` を有効化し、`PYTHONPATH` に親プロジェクトと
`third_party/blt` を追加する。

## 学習

```bash
source scripts/env.sh

# smoke 確認
python -m src.train.train --config configs/smoke.yaml --dry-run

# 新規学習
python -m src.train.train --config configs/arbor_1b.yaml

# 最新 checkpoint から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume latest

# best loss から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume best
```

### 1B の速度について

README の 1B コマンドは「起動方法」であり、短時間で 1B を十分に学習できる保証ではない。
現状の BitLinear は backward / optimizer まで fused した BitNet training stack ではないため、
1B 設定の速度は通常の大規模 BF16 学習に近い。

RTX 4090 の実測メモでは、固定形状 bench は `compile=default bs=8` で約 59.6k tok/s。
ただし現行の 1B 本走は BLT の動的 patching で系列形状が揺れるため、この値は出ない。
2026-06-08 の再測定では、`torch_compile: true` / `compile_dynamic: true` は追加 compile が
頻発し、速い step だけでも 14-16k tok/s、compile 待ち込み平均は大きく崩れた。
`torch_compile: false` / `micro_batch_size: 2` / `grad_accum_steps: 32` が現時点の安全な実測最速で、
ローカルデータ上の定常は約 13k tok/s。

1B 本走では `source scripts/env.sh` により以下を既定で入れる。

```bash
TORCHINDUCTOR_COMPILE_THREADS=12
TORCHINDUCTOR_FX_GRAPH_CACHE=1
```

速度設定は `configs/arbor_1b.yaml` の `speed` が正で、現在は
`torch_compile: false` / `micro_batch_size: 2` / `grad_accum_steps: 32`。

2026-06-09 の synthetic 1B bench (RTX 4090 / WSL, data I/O 除外, `batch=4`
`seq=2048` `grad_accum=4`) では以下。

```text
base arbor_1b          23.5k tok/s  13.5GiB
BitLinear weight cache 24.7k tok/s  14.4GiB
local_hidden_size=1024 29.5k tok/s  12.0GiB
1.00B fast config      27.7k tok/s  13.0GiB
```

`configs/arbor_1b_fast.yaml` は `hidden_size=2048` / `num_hidden_layers=22` を維持し、
FP の Local Encoder/Decoder を `1024` 幅に縮小、Global FFN を `6528` に広げて約
1.00B params に合わせた速度実験用 config。`speed.bitlinear_weight_cache: true` で
grad accumulation 中の packed ternary weight を再利用する。

`f16` や `f4` だけに寄せても、現状は backward / optimizer が BF16/8bit Adam 経路に残るため
学習速度は単純には伸びない。今効いているのは「forward packed weight の再利用」と
「FP local 部の縮小」。

学習中 `Ctrl+C` (SIGINT) または `kill -TERM <pid>` で次 step 境界にて
安全保存して終了する。二度押しで強制終了。

Checkpoint は `<step>.tmp/` に同期保存してから step ディレクトリへ publish する。
同じ step の上書きは拒否し、`latest` / `best` / `final` symlink の参照先は prune から保護する。

## テスト

```bash
python -m pytest
```

親プロジェクトの pytest は `tests/` のみを対象にする。`third_party/blt` の upstream テストは
追加依存を要求するため、通常の収集対象から外している。

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
