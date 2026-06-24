# arbor-llm 調査引継ぎメモ

## 背景

対象 repo:

```bash
https://github.com/hika019/arbor-llm
```

byte-level 階層 Transformer + BitNet b1.58 の自前 LLM 事前学習。

現状 checkpoint:

```bash
checkpoints/arbor2_1b_8k/latest -> step_0000009479
```

モデル概要:

```text
params 約 959M
patching=static
patch_size=8
max_bytes=8192
BitNet=ON
BitLinear layers=182
```

学習 loss はかなり低い:

```text
step 8780〜9360 あたり
loss ≈ 0.95〜1.05 nats
bpb ≈ 1.36〜1.50
ema_bpb ≈ 1.43〜1.47
```

しかし、生成品質・文脈追従が弱い。

## 症状

### 生成例

```bash
python -m src.infer.generate \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "日本の四季は" \
  --max-new-bytes 80 \
  --temperature 0
```

出力例:

```text
日本の四季は、大分の大分の大です。
この大分の大きな大きな大きな大�
```

別 checkpoint / seed では:

```text
日本の四季は、その天然記念物になるという。

天皇の天皇は、天皇の天皇には、天皇の天皇には...
```

強めの prompt でも greedy が反復 collapse する。

```text
日本には春、夏、秋、冬という四つの季節があり、
→ その夏は、夏の夏の夏には、その夏の夏には...
```

装置 prompt:

```text
この装置には赤、青、緑、黄色という四つのボタンがあり、
→ そのほかには、このようなものがある。
このようなも�������のようなものは...
```

文中に連続 `�` が出るケースがあり、単なる `max_new_bytes` 末尾打ち切りだけではない可能性がある。

## 既に確認済み

### cache / no-cache

```bash
python -m src.infer.generate \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "日本の四季は" \
  --max-new-bytes 80 \
  --temperature 0

python -m src.infer.generate \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "日本の四季は" \
  --max-new-bytes 80 \
  --temperature 0 \
  --no-cache
```

結果は一致。

少なくともこの範囲では KV cache バグの可能性は低い。

### sampling

```bash
for temp in 0.4 0.6 0.8; do
  python -m src.infer.generate \
    --ckpt latest \
    --ckpt-dir checkpoints/arbor2_1b_8k \
    --prompt "日本の四季は" \
    --max-new-bytes 160 \
    --temperature $temp \
    --top-p 0.9 \
    --seed 42
done
```

greedy の反復は軽減するが、意味的には依然として破綻。

例:

```text
temp=0.4:
日本の四季は、日本のコンパクトなものである。また、日本のコンパクトなものは...

temp=0.6:
日本の四季はあっという間においしくなった。
新鮮なお店は日本の大きさであるが...
```

## probe 結果

手元で `scripts/probe_generation_quality.py` を作成して、prompt + good target + bad target の byte NLL / bpb を比較した。

### Probe 1: 「日本の四季は」

```bash
python scripts/probe_generation_quality.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "日本の四季は" \
  --target "、春夏秋冬それぞれに異なる気候と風景を持っている。" \
  --bad-target "、日本のコンパクトなものである。" \
  --bad-target "あっという間においしくなった。" \
  --bad-target "、大分の大分の大です。"
```

結果:

```text
good bpb = 1.4212

bad[0] 、日本のコンパクトなものである。 bpb = 0.7785
bad[1] あっという間においしくなった。 bpb = 0.6349
bad[2] 、大分の大分の大です。 bpb = 1.5198
```

bad が good よりかなり低い。

「日本の四季は」の後で、意味的に正しい続きより、高頻度 Web っぽい文を好む。

### Probe 2: 強めの季節 prompt

```bash
python scripts/probe_generation_quality.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "日本には春、夏、秋、冬という四つの季節があり、" \
  --target "それぞれに異なる気候と風景を楽しむことができる。" \
  --bad-target "日本のコンパクトなものである。" \
  --bad-target "あっという間においしくなった。" \
  --bad-target "天然記念物になるという。"
```

結果:

```text
good bpb = 0.9579

bad[0] 日本のコンパクトなものである。 bpb = 0.8820
bad[1] あっという間においしくなった。 bpb = 0.8561
bad[2] 天然記念物になるという。 bpb = 0.8619
```

強めに条件づけても bad が勝つ。

### Probe 3: 装置・ボタン prompt

```bash
python scripts/probe_generation_quality.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --target "それぞれに異なる機能が割り当てられている。" \
  --bad-target "それぞれに異なる気候と風景を楽しむことができる。" \
  --bad-target "あっという間においしくなった。" \
  --bad-target "天然記念物になるという。"
```

結果:

```text
good bpb = 0.8511

bad[0] 気候と風景... bpb = 0.9888
bad[1] あっという間においしくなった。 bpb = 0.7225
bad[2] 天然記念物になるという。 bpb = 0.9945
```

文脈は完全には死んでいない。

「装置・ボタン」文脈では「機能」は「気候」「天然記念物」には勝つ。

ただし、短く高頻度な「あっという間においしくなった」が依然として強すぎる。

## context sensitivity

手元で `scripts/probe_context_sensitivity.py` を作成し、次 byte 分布を比較。

```bash
python scripts/probe_context_sensitivity.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --a "日本には春、夏、秋、冬という四つの季節があり、" \
  --b "この装置には赤、青、緑、黄色という四つのボタンがあり、"
```

結果:

```text
L1_prob = 0.246058
KL(a||b) = 0.054800
KL(b||a) = 0.050019
```

文脈で分布は変わっている。

ただし top byte はどちらも UTF-8 lead byte ばかりで、byte-level では意味診断が粗い。

## source_byte_ratios

`metrics.jsonl` より:

```json
{
  "fineweb2_ja": 0.464998,
  "wikipedia_ja": 0.189999,
  "aozora_modern": 0.020002,
  "japan_law": 0.005,
  "fineweb_edu_en": 0.119999,
  "fineweb_en": 0.050001,
  "finemath_4plus": 0.06,
  "github_code_filtered": 0.079999,
  "opc_code_web": 0.010001
}
```

日本語比率は高いが、`wikipedia_ja` が約 19% と強い。

出力 collapse は `〜である`, `天然記念物になるという`, `天皇`, `このようなもの`, `そのほかには` など Wikipedia / Web 定型に寄っている。

## 現在の仮説

優先度順。

### H1: train bpb が misleading

byte-level なので、日本語 UTF-8 の continuation byte を当てるだけでも bpb が下がる。

overall train bpb=1.45 は見た目ほど意味的日本語能力を保証しない。

必要な追加指標:

```text
UTF-8 lead byte bpb
UTF-8 continuation byte bpb
ASCII bpb
固定日本語 holdout bpb
good/bad target margin
```

### H2: データ分布が Web / Wikipedia 定型に寄りすぎ

`wikipedia_ja=19%` は強い。

`fineweb2_ja` も Web 定型・boilerplate を含む可能性あり。

生成が百科事典風 / 定型句 / 反復に落ちやすい。

試す案:

```text
wikipedia_ja を 0.19 -> 0.05 程度へ
fineweb2_ja を増やす
law は外すか極小
code はやや下げる
日常説明文・自然文 holdout を validation に入れる
```

### H3: frozen BitLinear inference path が怪しい

generate / probe は `load_inference_model()` 経由で `freeze_bitlinear_for_inference(model)` を呼んでいる。

そのため、今見ている生成・probe は学習時 forward ではなく packed ternary inference path。

文中に `�������` が出ているので、まず frozen inference を切って比較したい。

修正案:

`src/infer/generate.py` の `load_inference_model()` に `freeze_bitlinear: bool = True` を追加し、CLI に `--no-freeze-bitlinear` を追加。

`probe_generation_quality.py` も同じ flag を受け取る。

期待する比較:

```bash
python -m src.infer.generate \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --max-new-bytes 120 \
  --temperature 0 \
  --no-freeze-bitlinear
```

判定:

```text
no-freeze で大幅改善:
  packed / frozen inference path バグ濃厚。src/model/bitlinear.py を調査。

no-freeze でも同じ:
  モデル分布・データ・config 側。
```

### H4: global context の寄与が弱い

context sensitivity はゼロではないが、greedy は局所反復に落ちる。

調べたい値:

```text
byte embedding RMS
global context projection RMS
decoder input RMS
global / byte RMS ratio
```

ログを入れる場所:

```text
src/model/arbor.py の ArborModel.forward()
local decoder 入力を作る直前
decoder_input = byte_emb + projected_global_context 付近
```

環境変数で 1 回だけ出せる形にしたい。

例:

```python
if os.environ.get("ARBOR_DEBUG_CONTEXT", "0") == "1":
    byte_rms = byte_hidden.float().pow(2).mean().sqrt()
    global_rms = global_byte_context.float().pow(2).mean().sqrt()
    print(
        f"[ctx] byte_rms={byte_rms.item():.4f} "
        f"global_rms={global_rms.item():.4f} "
        f"global_byte_ratio={(global_rms / byte_rms.clamp_min(1e-8)).item():.4f}",
        flush=True,
    )
```

もし `global / byte < 0.05` なら global 経路が弱すぎる疑い。

### H5: BitNet training path / config 問題

BitNet b1.58 + 1B で train loss は下がるが、意味条件づけが育ちにくい可能性。

小モデルで controlled overfit をやるべき。

## Codex にやってほしいこと

### 1. `--no-freeze-bitlinear` を実装

対象:

```text
src/infer/generate.py
scripts/probe_generation_quality.py
```

`load_inference_model(..., freeze_bitlinear=True)` を追加し、flag で freeze を無効化。

その後、以下を比較:

```bash
# frozen
python scripts/probe_generation_quality.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --target "それぞれに異なる機能が割り当てられている。" \
  --bad-target "あっという間においしくなった。" \
  --bad-target "天然記念物になるという。"

# no-freeze
python scripts/probe_generation_quality.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --target "それぞれに異なる機能が割り当てられている。" \
  --bad-target "あっという間においしくなった。" \
  --bad-target "天然記念物になるという。" \
  --no-freeze-bitlinear
```

### 2. probe を train loop に統合

対象:

```text
src/train/train.py
```

checkpoint 保存直後、既存の `sample_at_checkpoint(saved_dir, global_step)` の近くに入れる。

欲しい metrics:

```text
probe_name
good_bpb
bad_min_bpb
margin = bad_min_bpb - good_bpb
```

JSONL に追記:

```json
{
  "step": 10000,
  "probe": {
    "ja_seasons": {
      "good_bpb": 0.9579,
      "bad_min_bpb": 0.8561,
      "margin": -0.1018
    }
  }
}
```

probe は config で on / off できるようにしたい。

例:

```yaml
probes:
  enabled: true
  max_new_bytes: 120
  items:
    - name: ja_seasons
      prompt: "日本には春、夏、秋、冬という四つの季節があり、"
      target: "それぞれに異なる気候と風景を楽しむことができる。"
      bad_targets:
        - "あっという間においしくなった。"
        - "天然記念物になるという。"
    - name: ja_buttons
      prompt: "この装置には赤、青、緑、黄色という四つのボタンがあり、"
      target: "それぞれに異なる機能が割り当てられている。"
      bad_targets:
        - "あっという間においしくなった。"
        - "天然記念物になるという。"
```

### 3. UTF-8 / debug decode を改善

`generate.py` は今 `errors="replace"` で不正 byte を `�` にしてしまう。

`--decode-errors` を追加したい。

選択肢:

```text
replace
backslashreplace
strict
ignore
```

さらに `--debug-utf8` で生成 byte の hex と replacement count を出したい。

目的:

```text
末尾の � が max_new_bytes 打ち切りなのか
文中の ������� が不正 byte 連発なのか
```

を切り分ける。

### 4. context contribution probe を追加

対象:

```text
src/model/arbor.py
scripts/probe_context_sensitivity.py または新規 scripts/probe_context_contribution.py
```

見たい値:

```text
byte_emb_rms
global_context_rms
decoder_input_rms
global_context / byte_emb
```

環境変数か debug flag で 1 回だけ出す。

training loop に常時入れる必要はない。

### 5. tiny overfit 実験を作る

目的: 実装 / architecture / BitNet / config の切り分け。

データ:

```bash
mkdir -p data/probes
cat > data/probes/tiny_ja.txt <<'TXT'
朝から雨が降っていたので、駅まで歩くのをやめてバスに乗った。
日本には春、夏、秋、冬という四つの季節があり、それぞれに異なる気候と風景を楽しむことができる。
この装置には赤、青、緑、黄色という四つのボタンがあり、それぞれに異なる機能が割り当てられている。
彼は傘を持っていなかったので、雨に濡れてしまった。
冷蔵庫に牛乳が残っていると思っていたが、今朝すでに使い切っていた。
TXT
```

作る config:

```text
configs/tiny_ja_overfit_fp.yaml     bitnet: false
configs/tiny_ja_overfit_bitnet.yaml bitnet: true
```

判定:

```text
fp で overfit できない:
  label / attention / architecture 実装バグ

fp でできるが bitnet でできない:
  BitNet training path / LR / quantization config 問題

両方できる:
  本走のデータ mix / scale / optimization / evaluation 問題
```

### 6. データ mix 実験 config

別 run で 1k〜3k step だけ比較。

本走を止める前に margin が改善するかを見る。

案:

```text
wikipedia_ja: 0.19 -> 0.05
fineweb2_ja: 0.46 -> 0.60
japan_law: 0.005 -> 0 または維持
code total: 0.09 -> 0.05〜0.08
aozora_modern: 0.02 -> 0.05
```

比較指標:

```text
train bpb
ja_probe_margin
greedy repetition
UTF-8 replacement count
```

## 現時点の暫定結論

単独の原因ではなさそう。

```text
高確度:
  train bpb は意味的生成品質を過大評価している
  Web / Wikipedia 定型への吸い込みが強い
  greedy 反復 collapse がある

中確度:
  global context の寄与が弱い
  データ mix が目的に合っていない
  BitNet / config が意味条件づけを育てにくい

未切り分け:
  packed / frozen BitLinear inference path
  文中の連続 � の原因
```

最初にやるべきは:

```text
1. --no-freeze-bitlinear 比較
2. train loop probe margin 追加
3. tiny overfit fp vs bitnet
4. context contribution RMS probe
```

## 2026-06-25 Codex 追記

実装済み:

```text
src/infer/generate.py
  --no-freeze-bitlinear
  --decode-errors backslashreplace
  --debug-utf8
  generate_byte_stream()

scripts/probe_generation_quality.py
  --no-freeze-bitlinear

scripts/probe_context_sensitivity.py
  --debug-context
  --no-freeze-bitlinear

src/eval/probes.py
  checkpoint-time probe と手動 probe で共有する scorer

src/train/train.py
  config probes.enabled/items を checkpoint 保存直後に実行
  metrics.jsonl に {"step": ..., "probe": ...} を追記
  step dir に probes.json を保存

src/model/arbor.py
  ARBOR_DEBUG_CONTEXT=1 で byte/global/decoder RMS を 1 回だけ表示

configs/tiny_ja_overfit_fp.yaml
configs/tiny_ja_overfit_bitnet.yaml
data/probes/tiny_ja.txt
  tiny overfit 切り分け用
```

検証:

```bash
.venv/bin/python -m py_compile \
  src/infer/generate.py src/eval/probes.py scripts/probe_generation_quality.py \
  scripts/probe_context_sensitivity.py src/train/train.py src/model/arbor.py tests/test_generate.py

.venv/bin/python -m pytest tests/test_generate.py -q
# 11 passed

source scripts/env.sh && .venv/bin/python -m src.train.train --config configs/smoke.yaml --dry-run
source scripts/env.sh && .venv/bin/python -m src.train.train --config configs/tiny_ja_overfit_fp.yaml --dry-run
source scripts/env.sh && .venv/bin/python -m src.train.train --config configs/tiny_ja_overfit_bitnet.yaml --dry-run
```

### frozen / no-freeze BitLinear 比較

対象:

```text
ckpt=checkpoints/arbor2_1b_8k/latest -> step_0000009479
prompt="この装置には赤、青、緑、黄色という四つのボタンがあり、"
target="それぞれに異なる機能が割り当てられている。"
bad_targets:
  "あっという間においしくなった。"
  "天然記念物になるという。"
```

結果:

```text
frozen:
  good_bpb=0.8445
  bad[0]_bpb=0.7235  delta=-0.1209
  bad[1]_bpb=0.9886  delta=+0.1441
  greedy="そのほかには、このようなものがある。\nこのようなも"

no-freeze:
  good_bpb=0.8522
  bad[0]_bpb=0.7236  delta=-0.1286
  bad[1]_bpb=0.9667  delta=+0.1145
  greedy="そのほかには、このようなものがある。\nこのようなも"
```

結論:

```text
packed / frozen BitLinear inference path が主因である可能性は低い。
no-freeze でも good/bad margin と greedy collapse は実質同じ。
```

### context contribution

```bash
python scripts/probe_context_sensitivity.py \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --a "日本には春、夏、秋、冬という四つの季節があり、" \
  --b "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --debug-context
```

結果:

```text
[ctx] byte_emb_rms=0.030982 global_context_rms=0.324780 decoder_input_rms=0.326178 global_byte_ratio=10.482713
L1_prob=0.276349
KL(a||b)=0.062713
KL(b||a)=0.056696
```

結論:

```text
少なくとも単純な「global context が小さすぎる / ほぼ 0」という状態ではない。
文脈差で次 byte 分布も変化している。
```

### UTF-8 debug

```bash
python -m src.infer.generate \
  --ckpt latest \
  --ckpt-dir checkpoints/arbor2_1b_8k \
  --prompt "この装置には赤、青、緑、黄色という四つのボタンがあり、" \
  --max-new-bytes 120 \
  --temperature 0 \
  --decode-errors replace \
  --debug-utf8
```

結果:

```text
replacement_count=8
... e3 82 82 82 82 82 82 82 82 e3 81 ae ...
```

結論:

```text
文中の ������� は末尾打ち切りだけではない。
UTF-8 continuation byte 0x82 が lead byte なしで連続生成されている。
```

注意:

```text
nvidia-smi は GPU を見ているが、この追記時点の Python/Torch では
torch.cuda.is_available() が false になっていた。smoke dry-run の初回は CUDA で動いたため、
WSL/CUDA ランタイム状態の揺れの可能性がある。
```

### tiny overfit 追加結果

Codex sandbox では Torch から CUDA が見えなかったため CPU で 200 step だけ確認。

FP:

```text
step=100 loss=2.2122
  ja_seasons good=3.6989 bad_min=3.5803 margin=-0.1186
  ja_buttons good=3.1796 bad_min=3.6356 margin=+0.4560

step=200 loss=0.4284
  ja_seasons good=2.4620 bad_min=4.0652 margin=+1.6032
  ja_buttons good=0.5689 bad_min=3.4212 margin=+2.8523
```

BitNet:

```text
step=100 loss=2.2915
  ja_seasons good=3.6712 bad_min=3.5924 margin=-0.0788
  ja_buttons good=3.2664 bad_min=3.4689 margin=+0.2025

step=200 loss=0.5104
  ja_seasons good=2.5912 bad_min=3.6329 margin=+1.0417
  ja_buttons good=0.6277 bad_min=3.2381 margin=+2.6104
```

結論:

```text
小さい固定データなら FP も BitNet も good/bad margin を正にできる。
したがって label shift / causal mask / global context 完全欠落のような致命的実装バグは薄い。
本走の問題は scale・目的関数・データ分布・patch 圧縮側が濃い。
```

### 2026-06-25 追加実装: UTF-8 / patch / metrics / config

実装:

```text
src/infer/generate.py
  UTF-8 constrained decoding を既定 ON
  --no-utf8-mask で旧挙動に戻せる
  残り生成 byte 数で完結不能な UTF-8 lead byte も禁止

src/model/arbor.py
  patching_mode: utf8 を追加
  UTF-8 文字先頭 byte だけを境界候補にする dynamic patching
  min_patch_len=8 / max_patch_len=12 / max_patches=1024 なら static 8 bytes/patch に近い粒度

src/train/train.py
  logging.byte_kind_metrics=true で ascii / utf8_lead / utf8_cont / other の bpb を metrics.jsonl に出す
  GPU 同期を避けるため interval 内は tensor のまま集計し、ログ時だけ CPU 化

configs/arbor_1b_8k_utf8.yaml
  data mix は static 版と同じ、patching だけ utf8

configs/arbor_1b_8k_utf8_mix.yaml
  utf8 patching + 日本語自然文寄せ mix
  fineweb2_ja=0.599, wikipedia_ja=0.05, aozora=0.05, law=0.001,
  fineweb_edu_en=0.11, fineweb_en=0.04, math=0.06, github_code=0.07, opc_code=0.02

configs/tiny_ja_overfit_utf8_bitnet.yaml
  tiny overfit の utf8 patching 比較用
```

検証:

```text
pytest tests/test_generate.py tests/test_train.py tests/test_arbor.py -q
=> 74 passed, 2 skipped

tiny_ja_overfit_utf8_bitnet --dry-run
=> patching=utf8, bytes/patch=8.83, patches/seq=29, max_patch/seq=29
```

既存 1B checkpoint への UTF-8 constrained decoding:

```text
prompt="この装置には赤、青、緑、黄色という四つのボタンがあり、"
max_new_bytes=120 temperature=0 debug_utf8

旧: replacement_count=8
新: replacement_count=0
```

出力はまだ反復する:

```text
そのほかには、このようなものがある。
このようなものは、このようなものがある。
こ 
```

結論:

```text
UTF-8 不正 byte は decoding constraint で潰せる。
ただし意味的 collapse / 高頻度句反復は残るので、次は
  1. arbor_1b_8k_utf8.yaml
  2. arbor_1b_8k_utf8_mix.yaml
を 1k〜3k step 比較し、probe margin / byte_kind_bpb / greedy repetition を見る。
```
