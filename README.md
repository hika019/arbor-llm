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
# smoke 確認
python -m src.train.train --config configs/smoke.yaml --dry-run

# 新規学習
python -m src.train.train --config configs/arbor_1b.yaml

# 最新 checkpoint から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume latest

# best loss から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume best
```

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
