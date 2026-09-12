# arbor-llm

バイトレベル階層 Transformer × **BitNet b1.58** の LLM (約 0.95B params) を
CUDA GPU / Apple Silicon MPS で学習するプロジェクト。自己完結実装
(モデルの依存は torch のみ)。

## アーキテクチャ (Arbor v2)

```
bytes (T=8192)                          token = byte + 4, vocab 260, tokenizer 不要
  └ byte embedding (FP)
  └ Local Encoder ×2      … patch 内 attention (BitLinear)
  └ 静的 patching          … 8 bytes/patch → 1024 patches
  └ Global Transformer ×20 … d=2048, GQA, causal (BitLinear)   ← パラメータの 95%
  └ Local Decoder ×4      … patch 内 causal (BitLinear)
  └ byte logits (FP head)
```

- **BitNet b1.58 公式レシピ準拠** (Microsoft "The Era of 1-bit LLMs" / 2B4T):
  - 重み: per-tensor absmean で ternary {-1,0,+1} (W1.58)
  - 活性: `model.activation_precision` で選択 (int8=公式A8 | bf8=float8_e5m2 | bf16=非量子化)。
    重みは常に W1.58 ternary。
  - STE は detach トリック (勾配は量子化後の値で計算)
  - SubLN: 全 BitLinear の入力は直前に RMSNorm を通る
    (q/k/v ← input_norm, o ← attn_sub_norm, gate/up ← ffn_norm, down ← ffn_sub_norm)
  - FFN は ReLU² gated、Linear は全て bias 無し
  - Embedding / patch 射影 / 出力 head / RMSNorm は FP (これも仕様どおり)
- **patching は 4 モード** (`model.patching_mode`):
  - `static` (既定・本走用): 固定 8 bytes/patch (MegaByte 方式)。形状固定で
    torch.compile が常時効き最速。
  - `utf8`: UTF-8文字の先頭byteを境界候補にする。
  - `space`: 空白・改行の直後で区切る (BLT の space patching)。
  - `entropy`: 小型バイト LM の次バイト予測エントロピーが閾値を超えた所で区切る
    (BLT 本命方式)。区切り用 LM は `configs/entropy_lm.yaml` (`arch: byte_lm`) で
    先に学習し、`model.entropy_model_ckpt` で渡す。凍結サブモジュールとして
    本体 checkpoint / HF エクスポートに同梱される。
  - 動的モードも patch 数を固定長 pad + block 対角マスクで処理するため
    tensor 形状は固定。境界候補から patch start への変換は CUDA extension で
    GPU 上に閉じる (CPU/MPS は torch 実装)。
  - 因果性 (未来バイト→過去 logits の漏れ無し) は全モードでテスト済み。
- 学習は BF16 シャドウ重みの QAT。推論は BitLinear の dequant キャッシュで
  毎回の重み再量子化を省く。packed ternary Triton 経路は速度診断用の明示 opt-in。

実データ1000-step A/B (RTX 4090 / WSL2, torch 2.11+cu128, `T=8192`,
`micro_batch=2`, `grad_accum=32`, peak LR 2e-4):

- `state_precision=fp32` (既定): loss **5.7 → 1.89**、EMA 1.96
- 改良dynamic `state_precision=int8`: loss **5.7 → 1.90**、EMA 2.00
- 旧linear int8 / 無スケールbf8: step 300–500で発散

改良int8は二次モーメントを対数間隔のdynamic符号帳で保持し、外れ値と同一blockに
ある小さな値が0へunderflowする問題を解消した。学習曲線はfp32とほぼ一致する。
ただし現状は符号帳検索のoptimizer処理が重いため、既定は高速かつ安定なfp32。
VRAM制約がある場合のみ改良int8を明示選択する。

`speed.bitlinear_compute_mode=bwd`はforwardを従来BF16のまま維持し、backward GEMMだけをFP8化する。
`speed.bitlinear_compute_mode=int8`はnativeな
`A8 INT8 × ternary INT8 → INT32 accumulation` forwardを使う。既定の
`speed.bitlinear_compute_mode=ternary`は2bit packed weightをkernel内でdecodeし、backwardは
optimizer step単位でcacheしたFP8 weightのN×K/K×N両layoutを使う。sm89+ (RTX
4090/5090) で動く。既定 `configs/arbor.yaml` はこのpacked経路 + patch_size=16 +
固定dim mean pooling + local encoder/decoder=1/2層で構成している。
INT8 GEMMは`speed.bitlinear_int8_backend: auto|int_mm|triton`でA/Bできる。
互換のため旧表記`speed.bitlinear_fp8`も引き続き受理する。

`data.packing=document`かつstatic patchingでは、dataloaderが新しいdocumentを
`model.patch_size`境界へPAD alignする。これによりlocal encoder/decoderでも
document境界を跨ぐpatchを作らない。また新document先頭ではglobal residual入力も
BOSへresetし、attention mask外のresidual経路から前文書が漏れるのを防ぐ。

`grad_accum_steps>1`でCUDA Graphsを使う`compile_mode=reduce-overhead`または
`max-autotune`を選ぶ場合は、parameter gradをgraph外の固定bufferとして事前確保し、
各micro-step前にCUDAGraph Treesのstep境界を明示する。これにより次のforward replayが
累積途中のgrad storageを上書きすることを防ぐ。固定buffer分のVRAMは起動時から確保される。
`default`と`max-autotune-no-cudagraphs`は従来どおりCUDA Graphsを使わない。
既定の1B/8k構成では `custom_op + auto + reduce-overhead` を使う。RTX 4090で
A-B-B-A各120 step（step 21--120集計）した結果、旧
`dot_current + legacy_raw + default`比で `bytes/s +19.0%`、step time `-16.1%`。
peak allocatedは同等（10.92 vs 10.94 GiB）だが、Graph用poolによりpeak reservedは
14.45 GiB（旧11.14 GiB）へ増える。

checkpoint stepでは、まずmodel/optimizer/scheduler/dataloaderを含むrecovery
checkpointを完全にpublishし、その後にvalidationを実行する。validation成功後は
modelを再保存せず`meta.json`と`best` symlinkだけをatomic更新する。validationが
CUDA errorで終了しても、そのstepから`--resume latest`できる。

validationはtraining用`torch.compile` wrapperを流用しない。既定は共有
`base_model`のeager evalで、`validation.torch_compile: true`を指定した場合だけ
`validation.compile_mode`による独立wrapperを作る。

データは日本語 (fineweb-2 ja / wikipedia ja / 青空文庫 / 法令) + 英語
(fineweb-edu / fineweb) + 数学 (finemath) の streaming 行レベル混合。既定 config
では正規化後で日本語 ~75% / 英語 ~19% / 数学 ~7%。コード系データセット
(OpenCoder-LLM/opc-fineweb-code-corpus) は提供元が非公開化したため除外済み。

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
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install -r requirements.txt
```

この作業環境にはPython 3.13.15をproject-localにも導入している。既存のCUDA依存を
持つ`.venv` (Python 3.12)は壊さず、3.13を選ぶ場合だけ次を使う:

```bash
source scripts/env313.sh
python3.13 --version
# 初回のみ:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Apple SiliconではPyTorchの通常wheelを使用する:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch
pip install -r requirements.txt
```

別環境で `.venv` を作り直した場合は、学習前に最低限これを確認する:

```bash
python - <<'PY'
import torch, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "mps", torch.backends.mps.is_available())
print("datasets", datasets.__version__)
PY
```

- `RuntimeError: \`datasets\` が必要 (pip install datasets)` /
  `ModuleNotFoundError: No module named 'datasets'`:
  `pip install -r requirements.txt` が入っていない。HF streaming データセットを読む
  本走 config (`configs/arbor.yaml`) では `datasets` が必須。
- optimizer state精度は `optim.state_precision: fp32 | int8 | bf8` で選ぶ。
  小さい層を含む全parameterへ一様に適用し、別精度への暗黙フォールバックはしない。
  `int8`はdynamic符号帳版。旧linear-int8 checkpointは非互換として明示エラーになる。
  BitNet本体は引き続きW1.58/A8 + floating shadow weightのQATであり、
  integer Parameterへ置換しない。

HF Hub から本走データを streaming する環境では、未認証アクセスだと rate limit /
timeout で止まりやすい。`configs/arbor.yaml` は fineweb-2 / wikipedia /
fineweb-edu / fineweb / finemath など複数 dataset の parquet を起動直後に解決するため、
RunPod 等の別環境では先に token と timeout を設定する:

```bash
export HF_TOKEN=hf_...
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
```

`Warning: You are sending unauthenticated requests to the HF Hub` が出る場合は
`HF_TOKEN` が見えていない。`The read operation timed out` が出る場合はネットワーク
または Hub 側の応答待ちなので、token 設定後に再実行する。

`Killed` だけが出て traceback が無い場合、Python 例外ではなく OS / コンテナ側の
強制終了であることが多い。まず CPU RAM の OOM kill を確認する:

```bash
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
dmesg -T | tail -50 2>/dev/null || true
free -h
```

`oom_kill` が増えている場合は、VRAM が足りていてもホストRAMが足りない。1B 本走は
32GB級GPUに加えて十分な CPU RAM と安定した外向きネットワークが必要。小さな環境では
`--dry-run` で1 stepだけ実行し、必要なら `speed.micro_batch_size` を下げる。

検証済み環境: Python 3.12 / torch 2.11+cu128 / transformers 4.57+ / datasets 4.8
(RTX 4090, WSL2)。Python 3.13.15ではproject sourceの`compileall`を確認済みで、
CUDA学習依存は上記手順で別途導入する。`source scripts/env.sh` で venv +
CUDA アロケータ設定 (expandable_segments) + inductor 設定が入る。

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

# 1B / 8K CUDA本走 (sm89+)
python -m src.train.train --config configs/arbor.yaml

# 1 stepだけ確認
python -m src.train.train --config configs/arbor.yaml --dry-run

# 最新 checkpoint から再開
python -m src.train.train --config configs/arbor.yaml --resume latest

# checkpoint の optimizer state は維持し、LR の基準値だけ現在の config に変える
python -m src.train.train --config configs/arbor.yaml --resume latest --rebase-lr-on-resume
```

設定ファイルは意図的に2本だけにしている:

- `configs/arbor.yaml`: Arbor本体。既定は `arbor2_1b_8k_filter`。
- `configs/entropy_lm.yaml`: entropy patching境界判定用ByteLM。

`speed.device: auto` はCUDA→MPS→CPUの順に選ぶ。MPSではモデル形状・データ混合・
optimizer state精度を変えず、gradient checkpointingを有効化し、
micro-batchを1にしてgrad accumulationを増やすことで実効batchを維持する。

- `Ctrl+C` (SIGINT) / `kill -TERM` で次 step 境界に安全保存して終了。二度押しで強制終了。
- checkpoint は 1000 step ごとに `./checkpoints/step_XXXXXXXXXX/` へアトミック保存
  (weights safetensors + optimizer + scheduler + RNG + dataloader 位置 + 実効 config)。
  `latest` / `best` / `final` symlink は prune から保護される。
- **resume の正確性**: HF streaming の位置は `datasets` の state_dict API で復元する
  (最初から流し直して skip しない)。RNG・dataloader 位置も復元。
- **resume 時の config 不一致はエラー** (checkpoint 内 config.yaml と model 節を照合)。
  意図的に変える場合のみ `--allow-config-mismatch`。
- **resume は optimizer/scheduler state も復元する**。そのため途中で config の
  `optim.lr` を変えただけでは、checkpoint 内の LR が優先される。LR だけを変えて
  Adam の momentum 等は引き継ぎたい場合は `--rebase-lr-on-resume` を付ける。
  optimizer / scheduler を完全に初期化して重みだけ使う場合は `--init-from`。
- checkpoint のロードは strict (部分ロードを黙って通さない)。保存は compile 前の
  モデルで行うので `_orig_mod.` prefix 問題も起きない。
- `best` は **train loss の EMA** が最良だった checkpoint (validation best ではない)。
- `speed.cuda_prefetch: true` で次 batch を別 CUDA stream で GPU へ先行転送する。
  prefetched batch は checkpoint state に同梱されるため、resume で 1 batch 欠落しない。
- `speed.bitlinear_compute_mode: bwd` は sm89+ CUDA で BitLinear の backward GEMM を
  FP8化する。非対応deviceではエラーになり、暗黙に無効化しない。forwardまで
  FP8化する`full`は追加丸めと速度低下があり得るため既定では使わない。
- `speed.bitlinear_compute_mode: ternary` は実験的な学習経路。optimizer step後に
  forward用とdX用のternary weightをそれぞれ2bit（4 weights/byte）へpackし、
  Triton kernel内でdecodeする。`kmajor_single_dot` は packed weight を
  `[K/4,N]` のGEMM向けlayoutから1回loadし、4 weightへregister内decode後、
  dense INT8 fragmentへinterleaveして `tl.dot` を1回だけ呼ぶ。
  `speed.bitlinear_ternary_backend` は計算backendだけを固定し、
  `speed.bitlinear_ternary_tuning: auto` が実行GPU上でlaunch tileを測定する。
  結果はGPU/torch/CUDA/Triton/kernel-versionを含むfingerprintで
  `${XDG_CACHE_HOME:-~/.cache}/arbor/packed_ternary_autotune.json` に保存される。
  `torchrun` で初期化済みのprocess groupでは、global rank 0だけがこの事前測定と
  cache書き込みを行い、他rankは同期後に同じcacheを再読込する。この連携はautotune
  の重複実行を防ぐためだけのもので、モデルをDDP/FSDPでwrapしたり、勾配同期・
  データ分割を行う完全な分散学習を意味しない。
  `fixed`（`bitlinear_ternary_fixed_tile: BM,BN,BK,WARPS,STAGES` 必須）と
  conservative configを使う`off`もdebug/reproduction用に選べる。
  現行mean-pooling構成の実学習A-B-B-Aでは、`custom_op + auto`をCUDA Graphsで
  captureする経路がlegacy/defaultより約19%高いthroughputを再現したため、これを
  既定にする。`legacy_raw`は旧shape heuristicをraw Tritonで再現するrollback経路、
  `raw`は固定tileの境界A/B用である。cache生成後は`raw_plan`を
  選ぶと、fingerprintが一致する全shapeの固定launch planをcompile前に読み込み、
  hot path内のcustom opとresolverを外せる。plan missは暗黙fallbackせずエラーにする。
  設定契約は`legacy_* = dot_current + off`、`raw = fixed`、
  `raw_plan = auto + cache`である。
  `dot_current` は旧vectorized decode後に `tl.dot` でINT8 Tensor Coreを使う。
  `kmajor_current` はlayout単独比較、`dot` は4-way grouped decode比較用。dWは
  `speed.bitlinear_ternary_wgrad_backend: int8|fp8|auto` で選択できる。
  `int8` は `Q(dY)^T Q(X)` のdense INT8 GEMM、`fp8` はtensorwise FP8 GEMM、
  `auto` は現在の代表shape測定に基づき `N>=K` でFP8、それ以外でINT8を使う。
  現行1B/8k構成ではpacked ternaryと`fp8` dWを既定にする。
- `model.global_attn_impl: flex` は CUDA + `torch.compile` 必須。条件を満たさない
  場合はエラーになり、SDPAへ暗黙フォールバックしない。
- `optim.state_precision: fp32` が既定。実データ1000-stepでloss 1.89まで安定して低下。
  `int8`はblockwise scale + 非線形dynamic符号帳で、同じ1000-stepをloss 1.90で完走。
  無スケール`bf8`は発散を確認しており、実験用途以外では使わない。
- `optim.fp32_backend: auto` は連続CUDA tensorのFP32-moment AdamWをTritonで融合する。
  中間tensorとkernel起動を削減し、moment精度・BF16/FP16 parameterの丸め位置を維持する。
  GPU機種判定やTensor Core専用命令は使わない。CPU/MPS/ROCm・非連続tensorは同じAdamWの
  従来演算を使う。`eager`で従来版、`triton`で融合必須（非対応入力はエラー）。
  backendはcheckpointに固定されず、resume時のconfigで選ぶ。
  単体比較: `python -m scripts.bench_adamw --config configs/arbor.yaml`。
- `speed.sync_each_step: false` が既定。毎 step の `torch.cuda.synchronize()` は行わず、
  ログ/保存など scalar 化が必要な箇所でのみ同期する。
- 性能A/Bには `--benchmark-steps N` を使う。指定optimizer step数だけ実行し、
  checkpoint、validation、sampling、probeを省略する。終了時にCUDA peak
  allocated/reserved memoryも表示し、metricsは通常runと混ぜず
  `logs/benchmark_*.jsonl`へ保存する。
- ログの throughput は `bytes/s`。entropy/space patching では `patches/s`,
  `bytes/patch`, `patches/seq`, `max_patch/seq`, `patch_headroom` も出す。
  `ByteLM_ms` / `patching_ms` / `Arbor_ms` は `profile_sections_every_steps` 間隔で
  `ByteLM_ms` / `patching_ms` を no-grad probe で同期計測し、`Arbor_ms` は
  compiled forward 時間からの概算として出す。

CUDA計算部分だけを合成データで比較する場合:

```bash
python -m scripts.bench_cuda \
  --seq 8192 --micro-batch 2 --grad-accum 32 \
  --compile --global-attn-impl flex --bitlinear-fp8 bwd \
  --weight-cache full --production-optimizer
```

### entropy patching を使う手順 (区切り用 LM の学習)

entropy モードは「次バイトの予測しにくさ」を測る小型バイト LM (ByteLM) を
**事前に別途学習**して凍結利用する (本体と同時には学習しない。境界判定は
離散なので勾配が流れず、判定基準が動くと本体の学習も不安定になるため)。

```bash
# 1. 区切り用 ByteLM を学習 (データ混合は本走と同じにすること)
python -m src.train.train --config configs/entropy_lm.yaml

# 2. configs/arbor.yaml の model.patching_mode を entropy に変更
# entropy_lm_config: entropy_lm.yaml からByteLM構成とcheckpointを自動解決するため、
# Arbor側へ同じByteLM model定義を複製しない。
python -m src.train.train --config configs/arbor.yaml
```

thresholdを校正する場合:

```bash
python scripts/calibrate_entropy_threshold.py \
  --config configs/arbor.yaml \
  --checkpoint checkpoints/entropy_lm/latest \
  --target-bytes-per-patch 5.0 --write
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
生成は既定でフルフォワード方式を使う。単発の品質確認では、KV cache 経路の
数値差より checkpoint 本体の出力を優先するため。BitLinear は推論凍結
(dequant キャッシュ) を使う。packed ternary Triton 経路は
`ARBOR_PACKED_BITLINEAR_INFERENCE=1` の明示指定時だけ有効にする。2 階層 KV cache
(global は patch 確定ごとに追記、local は patch 内のみ再計算) は
`--cache` を指定した場合だけ使う実験的高速化。

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
