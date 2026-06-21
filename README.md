# arbor-llm

バイトレベル階層 Transformer × **BitNet b1.58** の LLM (約 0.95B params) を
RTX 4090 単機で学習するプロジェクト。自己完結実装 (モデルの依存は torch のみ)。

## アーキテクチャ (Arbor v2)

```
bytes (T=2048)                          token = byte + 4, vocab 260, tokenizer 不要
  └ byte embedding (FP)
  └ Local Encoder ×2      … patch 内 attention (BitLinear)
  └ 静的 patching          … 4 bytes/patch → 512 patches
  └ Global Transformer ×20 … d=2048, GQA, causal (BitLinear)   ← パラメータの 95%
  └ Local Decoder ×4      … patch 内 causal (BitLinear)
  └ byte logits (FP head)
```

- **BitNet b1.58 公式レシピ準拠** (Microsoft "The Era of 1-bit LLMs" / 2B4T):
  - 重み: per-tensor absmean で ternary {-1,0,+1} (W1.58)
  - 活性: per-token absmax で int8 (A8)
  - STE は detach トリック (勾配は量子化後の値で計算)
  - SubLN: 全 BitLinear の入力は直前に RMSNorm を通る
    (q/k/v ← input_norm, o ← attn_sub_norm, gate/up ← ffn_norm, down ← ffn_sub_norm)
  - FFN は ReLU² gated、Linear は全て bias 無し
  - Embedding / patch 射影 / 出力 head / RMSNorm は FP (これも仕様どおり)
- **patching は 3 モード** (`model.patching_mode`):
  - `static` (既定・本走用): 固定 4 bytes/patch (MegaByte 方式)。形状固定で
    torch.compile が常時効き最速。
  - `space`: 空白・改行の直後で区切る (BLT の space patching)。
  - `entropy`: 小型バイト LM の次バイト予測エントロピーが閾値を超えた所で区切る
    (BLT 本命方式)。区切り用 LM は `configs/entropy_lm.yaml` (`arch: byte_lm`) で
    先に学習し、`model.entropy_model_ckpt` で渡す。凍結サブモジュールとして
    本体 checkpoint / HF エクスポートに同梱される。
  - 動的 2 モードも patch 数を固定長 pad + block 対角マスクで処理するため
    tensor 形状は固定。境界候補から patch start への変換は CUDA extension で
    GPU 上に閉じる (CPU はテスト用 torch 実装)。動作確認用の小規模設定が
    `configs/trial_space.yaml` / `configs/trial_entropy.yaml`。
  - 因果性 (未来バイト→過去 logits の漏れ無し) は 3 モードともテストで検証済み。
- 学習は BF16 シャドウ重みの QAT。BitNet の推論側の利点 (packed ternary kernel に
  よる省メモリ・高速化) は未実装で、現状の推論は bf16 で on-the-fly 量子化する。

実測 (RTX 4090 / WSL2, synthetic, `micro_batch=8` `T=2048` compile 込み):
**51.2k bytes/s, VRAM 16.1 GiB** (旧 BLT 版の本走実測 ~13k bytes/s から大幅改善)。

データは日本語 60% (fineweb-2 ja) + 英語 news 15% (cc_news) + 英語 edu 25%
(fineweb-edu) の streaming 行レベル混合。

## セットアップ

```bash
sudo apt install -y git python3 python3-venv python3-dev build-essential ninja-build
# entropy/space 動的 patching を CUDA で使う場合は nvcc + CUDA headers も必要
# (例: nvidia-cuda-toolkit または CUDA toolkit)。nvcc は gcc-12 系が無難。

git clone <this-repo-url> arbor-llm
cd arbor-llm

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install -r requirements.txt
```

検証済み環境: Python 3.12 / torch 2.5.1+cu121 / transformers 4.57+ / datasets 4.8 /
bitsandbytes 0.49 (RTX 4090, WSL2)。`source scripts/env.sh` で venv +
CUDA アロケータ設定 (expandable_segments) + inductor 設定が入る。
BLT repo の clone / submodule / third_party 配置は不要 (旧 BLT fork 依存は廃止)。
`third_party/blt` がローカルに存在する環境もあるが、これは Git 管理外の旧作業ツリーを
無視するために `.gitignore` へ残しているだけで、現行コードからは参照しない。

動的 patching の CUDA extension は初回実行時に `.torch_extensions/` へ JIT build
される。`scripts/env.sh` が新しすぎる gcc を PATH に入れていても、extension 側は
見つかれば `gcc-12` / `g++-12` を優先して nvcc に渡す。別 compiler を使う場合は
`ARBOR_EXT_CC` / `ARBOR_EXT_CXX` を指定する。
`TORCHINDUCTOR_COMPILE_THREADS` は既定では指定しない。PyTorch が CPU 数から
compile worker 数を決めるので、この環境では 20 thread になる。メモリ不足で
compile が落ちる場合だけ、`TORCHINDUCTOR_COMPILE_THREADS=4 source scripts/env.sh`
のように明示的に下げる。

## 学習

```bash
source scripts/env.sh

# smoke 確認 (小モデル・ローカルデータ・50 step, CPU でも可)
python -m src.train.train --config configs/smoke.yaml

# 1B 本走 (初回は torch.compile に ~2 分)
python -m src.train.train --config configs/arbor_1b.yaml

# 最新 checkpoint から再開
python -m src.train.train --config configs/arbor_1b.yaml --resume latest
```

- `Ctrl+C` (SIGINT) / `kill -TERM` で次 step 境界に安全保存して終了。二度押しで強制終了。
- checkpoint は 1000 step ごとに `./checkpoints/step_XXXXXXXXXX/` へアトミック保存
  (weights safetensors + optimizer + scheduler + RNG + dataloader 位置 + 実効 config)。
  `latest` / `best` / `final` symlink は prune から保護される。
- **resume の正確性**: HF streaming の位置は `datasets` の state_dict API で復元する
  (最初から流し直して skip しない)。RNG・dataloader 位置も復元。
- **resume 時の config 不一致はエラー** (checkpoint 内 config.yaml と model 節を照合)。
  意図的に変える場合のみ `--allow-config-mismatch`。
- checkpoint のロードは strict (部分ロードを黙って通さない)。保存は compile 前の
  モデルで行うので `_orig_mod.` prefix 問題も起きない。
- `best` は **train loss の EMA** が最良だった checkpoint (validation best ではない)。
- `speed.cuda_prefetch: true` で次 batch を別 CUDA stream で GPU へ先行転送する。
  prefetched batch は checkpoint state に同梱されるため、resume で 1 batch 欠落しない。
- `speed.sync_each_step: false` が既定。毎 step の `torch.cuda.synchronize()` は行わず、
  ログ/保存など scalar 化が必要な箇所でのみ同期する。
- ログの throughput は `bytes/s`。entropy/space patching では `patches/s`,
  `bytes/patch`, `patches/seq`, `max_patch/seq`, `patch_headroom` も出す。
  `ByteLM_ms` / `patching_ms` / `Arbor_ms` は `profile_sections_every_steps` 間隔で
  `ByteLM_ms` / `patching_ms` を no-grad probe で同期計測し、`Arbor_ms` は
  compiled forward 時間からの概算として出す。

### entropy patching を使う手順 (区切り用 LM の学習)

entropy モードは「次バイトの予測しにくさ」を測る小型バイト LM (ByteLM) を
**事前に別途学習**して凍結利用する (本体と同時には学習しない。境界判定は
離散なので勾配が流れず、判定基準が動くと本体の学習も不安定になるため)。

```bash
# 1. 区切り用 ByteLM を学習 (データ混合は本走と同じにすること)
python -m src.train.train --config configs/entropy_lm.yaml

# 2. 本体 config で entropy モードを指定して学習
#    model.patching_mode: entropy
#    model.entropy_model: (ByteLM の構成: entropy_lm.yaml の model 節と一致させる)
#    model.entropy_model_ckpt: ./checkpoints/entropy_lm/latest
python -m src.train.train --config configs/trial_entropy.yaml   # 小規模な実例

# 1B / 8k 本走用
python -m src.train.train --config configs/arbor_1b_8k_entropy.yaml
```

軽量な区切り用LMを使う場合は `configs/entropy_lm_compact.yaml` を学習し、
本体は `configs/arbor_1b_8k_entropy_compact.yaml` を使う。この構成は
ByteLM を 384 hidden / 4 layers / attention window 512 に落とすため速いが、
scorer が変わるので本体学習前に threshold を校正する。

```bash
python -m src.train.train --config configs/entropy_lm_compact.yaml
python scripts/calibrate_entropy_threshold.py \
  --config configs/arbor_1b_8k_entropy_compact.yaml \
  --checkpoint checkpoints/entropy_lm_compact/final \
  --target-bytes-per-patch 5.0 --write
python -m src.train.train --config configs/arbor_1b_8k_entropy_compact.yaml
```

学習後の ByteLM は本体の checkpoint / HF エクスポートに同梱されるので、
推論側で別途用意する必要は無い。`model.entropy_threshold` (nats) で
区切りの細かさを調整する (小さいほど細かく切れる)。凍結 ByteLM の
entropy score 計算は `torch.no_grad()` で実行され、本体側の autograd graph には
入らない。

### checkpoint 時の自動サンプル生成

`sampling.enabled: true` で、checkpoint 保存のたびに固定プロンプト・固定 seed で
短文を生成し、ログ + checkpoint dir の `samples.txt` に保存する。step 間で
出力品質の変化を同条件比較できる。

## 推論 (checkpoint を試す)

```bash
python -m src.infer.generate --ckpt latest --ckpt-dir checkpoints/arbor2_1b_8k_entropy \
    --prompt "日本の四季は" --max-new-bytes 200
python -m src.infer.generate --ckpt best --ckpt-dir checkpoints/arbor2_1b_8k_entropy \
    --interactive    # 対話モード
python -m src.infer.generate --ckpt 5000 --ckpt-dir checkpoints/arbor2_1b_8k_entropy \
    --prompt "日本の四季は"                         # 特定 step
```

モデル構成は checkpoint 内の `config.yaml` から自動復元される。
`--ckpt latest` / `best` / step 数は `--ckpt-dir` で指定した run ディレクトリ内で解決される。
生成は 2 階層 KV cache (global は patch 確定ごとに追記、local は patch 内のみ
再計算) + BitLinear 推論凍結 (packed ternary / dequant キャッシュ) を使う。
1B 実測 27 B/s (4090/WSL2。フルフォワード方式 `--no-cache` 比 ~4 倍)。
それ以上はカーネル起動レイテンシ律速 (issues.md 参照)。

## HuggingFace 形式エクスポート

```bash
python scripts/export_hf.py --ckpt latest --verify
# -> export/<run_name>-step<N>/。--verify は学習スタックとのロジットビット一致を確認
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "export/arbor2_1b-step10000"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype="auto").cuda()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids = tok("日本の四季は", return_tensors="pt").input_ids.cuda()
print(tok.decode(model.generate(ids, max_new_tokens=100)[0]))
```

モデル定義 (`arbor_model/`, torch のみ依存) とバイト tokenizer を同梱した
`trust_remote_code` 形式。3 つの patching モードすべてエクスポート可能で、
entropy モードでは凍結 ByteLM も safetensors に同梱される。

HF Hub への公開も可能:

```bash
huggingface-cli upload <user>/<repo> export/arbor2_1b-step10000 .
# 利用側: AutoModelForCausalLM.from_pretrained("<user>/<repo>", trust_remote_code=True)
```

### ollama / LM Studio について

**非対応。** これらは llama.cpp (GGUF) の既知アーキテクチャ専用で、バイトレベル
階層構造 + BitLinear の変換器は存在しない。transformers (Python) から利用すること。

## テスト

```bash
python -m pytest
```

因果性テスト (未来バイトの変更が過去の logits に漏れないこと)、BitLinear の
量子化/STE 勾配の正しさ、checkpoint の保存/再開、HF tokenizer 往復などを含む。

## ディレクトリ

```
src/
  model/   bitlinear.py (BitNet b1.58), arbor.py (モデル本体, 自己完結)
  data/    バイト直 streaming dataset (HF interleave / local mmap), 正確 resume
  train/   train.py, checkpoint, signals, optim
  infer/   generate.py (checkpoint からの生成 CLI / 学習中サンプル生成)
  hf/      HF エクスポートに同梱する modeling / tokenizer テンプレート
  eval/    perplexity 他
scripts/   export_hf.py, env.sh
configs/   arbor_1b.yaml (本走), smoke.yaml
checkpoints/   学習 checkpoint (.gitignore)
export/        HF 形式エクスポート先 (.gitignore)
```
