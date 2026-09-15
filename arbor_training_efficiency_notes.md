# 学習効率化メモ (GPU 増設以外の手口)

作成: 2026-09-13。公開ラボ (DeepSeek / Moonshot / Meta / Google / Ai2 / Microsoft) の技術報告と
nanoGPT スピードランから、arbor (4090 ×1、1B BitNet b1.58、byte 直、8k context) に
効きそうなものを抽出した。OpenAI / Anthropic 自身は学習効率の論文を出していない
(OpenAI が公開しているのは GPT-4 の loss を 1/10000 計算量のプロキシから予測した話のみ)。

分類:

- **A. 同じ loss に到達する token 数を減らす** (token efficiency)
- **B. token あたりの計算量を減らす**
- **C. 小さい実験で当てて、大きい run で外さない** (experiment efficiency)

## 1. batch size warmup (A) — 実装済み、ByteLM A/B で効きを確認し本走に採用 (2026-09-14)

`speed.grad_accum_steps` に [[step, accum], ...] (`src/train/grad_accum.py`)。詳細は configs/arbor.yaml のコメント参照。
lr schedule (cosine / decay_start / decay_end) の進行は消費 bytes 割合で測る (accum が変わっても
固定 accum と同じ bytes で同じ lr。最初の A/B はこれが step 割合だったため無効になった)。

- **A/B 結果** (BitNet ByteLM 19M、本走 mix、8k、lr 8e-4、同 983M bytes):

  | run | 最終 ema | 最終 mean_bpb |
  |---|---|---|
  | accum 1 固定 (65.5k B/update, 15000 step) | **0.976** | **1.476** |
  | 1→2→4 warmup, lr √(accum/4) 補正 (8750 step) | 1.010 | 1.531 |
  | 1→2→4 warmup, lr 補正なし (8750 step) | 1.021 | 1.540 |
  | accum 4 固定 (262k B/update, 3750 step) | 1.041 | 1.574 |

  - warmup は「最初から大 batch」に bpb −0.043 (2.7%) で勝つ。Ai2 の主張を方向として再現。
  - 小 batch を高 lr で最後まで回すのが最強 = 19M × 1GB では臨界 batch が 262k bytes に届かない。
    1B × 105GB は後半で臨界を超えるので schedule が要る (切替点は bytes 比率で仮置き)。
  - lr 補正: `none` は序盤 (bytes 1/3 で bpb 1.679 vs 1.721) 有利だが最終は `sqrt` が僅差で勝つ
    (1.531 vs 1.540)。**`sqrt` を採用**。
- 本走 (arbor.yaml): `grad_accum_steps: [[0, 8], [20000, 16], [50000, 32]]`、`sqrt`、total_steps 230k
  (105GB 維持)。

- 出典: Ai2 "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch
  Language Model Training" (arXiv 2505.23971) / [Ai2 blog](https://allenai.org/blog/critical-batch-size)。
  OLMo 1B で同 loss を **43% 少ない optimizer step** で達成。batch を倍にするとき lr を √2 倍
  (square-root scaling rule)。臨界 batch は checkpoint から batch 違いの枝を Δstep 学習して
  「小さい batch 全てに loss で劣らない最大の batch」として測る (branched training)。
- 関連: "How to Set the Batch Size for Large-Scale Pre-training" (arXiv 2601.05034) —
  WSD スケジュールと組み合わせ、固定 batch より増加 batch の方が loss/下流とも良い。
- arbor での注意: 序盤の bytes/step が減るので `total_steps` の bytes 換算が変わる (再計算済み)。

## 2. Muon optimizer (A) — 実装済み、ByteLM A/B で AdamW に負け → 本走は AdamW のまま (2026-09-14)

- 出典: Moonshot "Muon is Scalable for LLM Training" (arXiv 2502.16982)。AdamW 比
  **約 2× の計算効率**、3B/16B MoE (Moonlight) を 5.7T token で実証。スケールさせるのに
  必要だった 2 点 = weight decay の追加と、更新の RMS を AdamW に合わせる per-parameter
  スケール (`0.2·√max(rows, cols)` 相当)。Kimi K2 も採用 (MuonClip)。
- 出典: "Practical Efficiency of Muon for Pretraining" (arXiv 2505.02222) — 大 batch でも
  データ効率を保つ、µP との併用。
- **リスク**: "Bit-by-Bit: Progressive QAT Strategy ... for Stable Low-Bit LLMs"
  (arXiv 2604.07888) は **超低 bit QAT で Muon が AdamW に一貫して勝てない** (収束速度・
  最終 ppl 同等かやや悪い、一部層で短期振動) と報告。ternary + STE との相性は未知。
- Muon 自体の state 量子化: "Effective Quantization of Muon Optimizer States"
  (arXiv 2509.23106)、"MuonQ" (arXiv 2605.11396)。直交化が特異ベクトル方向の量子化誤差を
  増幅するので、8bit blockwise までは安全、4bit は工夫が要る。
- 実装 (82ab40f): `optim.optimizer: muon` (`src/train/optim.py` の `Muon`)。
  momentum (fp32) → Newton-Schulz 5 反復 (bf16) → `0.2·√max(rows, cols)` の RMS 合わせ →
  既存 `_apply_update` (stochastic rounding)。対象は transformer `Block` 内の 2D 重みだけで、
  embedding / head / RMSNorm / `patch_proj` / `global_to_local` は fp32 state の AdamW
  (param_groups[1])。`muon_momentum` / `muon_nesterov` / `muon_ns_steps` / `muon_adamw_lr`。
  lr / wd は AdamW と共有 (Moonshot のスケールの狙い)。`state_precision` は fp32 のみ。
  副産物: 二次モーメント不要で optimizer state が半分 → 24GB の VRAM に効く。
  コスト見積: 1B の Newton-Schulz は ~53 TFLOP/step ≈ +0.35 s (5.2 s/step の +7%)。
- A/B (config は削除済み。再現は entropy_lm.yaml の model に `bitnet: true` + arbor.yaml の data mix):
  BitNet ByteLM 19M、8k / micro 8 / lr 8e-4 / wd 0.1 / stochastic rounding、15k step (≈40 分)。
  差分は optimizer だけ。判定は train loss ema と validation (ja_web / english / code) の bpb。
- **結果 (2026-09-14): Muon は AdamW に一貫して僅差で負け。** 「Bit-by-Bit」の報告どおり。

  | step | AdamW ema | Muon ema | AdamW mean_bpb | Muon mean_bpb |
  |---|---|---|---|---|
  | 500 | 1.997 | 2.462 | — | — |
  | 1000 | 1.465 | 1.746 | — | — |
  | 2500 | — | — | 1.791 | 1.818 |
  | 5000 | 1.061 | 1.064 | 1.617 | 1.623 |
  | 10000 | 0.992 | 0.999 | 1.498 | 1.509 |
  | 15000 | **0.976** | 0.982 | **1.476** | 1.488 |

  - 序盤 (warmup 500 step 前後) は Muon が大きく遅れる (+0.47 @500)。直交化した更新は
    全特異方向を同じ大きさで動かすため、三値化の閾値付近にある latent weight が一斉に
    符号を跨ぎやすく、初期の ternary 構造が定まるまで荒れる (推定)。step 2000 で追いつき、
    step 3000 で一度だけ逆転 (-0.003) するが、以後は +0.005〜0.007 / bpb +0.010 で固定。
  - 最終 mean_bpb 1.488 vs 1.476 (+0.8%)、3 ドメインとも Muon が悪い。速度も Newton-Schulz
    分だけ遅い (234 vs 159 ms/step、19M では opt が 53 ms)。
  - 結論: ternary + STE では Muon の「2× 効率」は出ない。本走は AdamW のまま。実装は残す
    (`optim.optimizer: muon`) が、再挑戦するなら (1) Muon 用に lr を別に振る (RMS 合わせは
    FP 前提の係数)、(2) warmup を長くする / 序盤だけ AdamW で始めて切り替える、(3) momentum
    0.95 → 0.9、あたりが候補。ただし本走への投入条件は「ByteLM で明確に勝つこと」。
- 制約: 本走中は VRAM 22.5/23GB なので ByteLM でも同時実行は WSL 落ちの危険
  (推論同時実行で落ちた前例と同じ機構)。本走停止中に回す。

## 3. midtraining / annealing の複数回 + 重み平均 (A)

- 出典: Ai2 "2 OLMo 2 Furious" (arXiv 2501.00656)。stage 2 で高品質 mix (Dolmino) に
  切り替え lr を線形に 0 へ。**同じ anneal を data 順序を変えて 3 回回し、重みを平均** (souping)
  して local minimum を改善。7B は 50B token ×3、13B/32B は 100B/300B。
- 出典: Meta "The Llama 3 Herd of Models" (arXiv 2407.21783) — 事前学習末尾で高品質
  データに anneal (annealing データで下流 benchmark が上がる。少量の高品質データを終盤に入れる
  のが効く)。
- 出典: "Mid-Training of Large Language Models: A Survey" (arXiv 2510.06826)。
- arbor: `configs/arbor_1b_8k_cpt.yaml` が既に stage 2 相当 (高品質 mix、lr 1e-4)。
  残りは seed 違いで 2〜3 本回して safetensors を平均するスクリプト。GPU コストは anneal N 回分
  (各 20k step) なので「次に CPT を回すときに seed を変えた 2 本目を足す」が現実的。
  平均は BitLinear の shadow weight (bf16) 同士で取り、平均後に再量子化される点に注意
  (ternary 化は forward 時に absmean で行われるので平均自体は素直に効く見込み)。

## 4. model growth: 次の 2B を 1B base から成長させる (A)

- 出典: "Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM
  Pre-Training" (arXiv 2405.15319, NeurIPS 2024)。4 種の成長演算子のうち **depth 方向の
  stacking (G_stack)** が最良。7B で 300B token 学習相当の loss に **194B token で到達
  (54.6% 高速化)**、750B token まで有効性を確認。成長比率とタイミングのガイドラインあり。
- 出典: Apple "Scaling Smart: Accelerating Large Language Model Pre-training with Small
  Model Initialization" (HyperCloning, arXiv 2409.12903)。width 方向に関数保存で拡張し
  **2.2〜4× 速く収束**。linear / attention / norm を関数保存に写す。
- 関連: "Weight subcloning" (arXiv 2312.09299, 逆方向 = 大→小)。
- arbor: `--init-from` は重みのみロード (strict) なので、層マッピング付きのロード
  (例: 20 層 global → 40 層は `layer[i] ← base[i % 20]` or `base[i // 2]`) を足せば
  G_stack になる。ByteLM 側の成長も同じ仕組みで可能。2B の形状設計と VRAM の壁
  (2k context 化 / micro_batch) が先。

## 5. 長さ curriculum / 長さ bucket packing (B) — arbor には当てはまらない (2026-09-13 計測)

- 出典: Apple "Dataset Decomposition: Faster LLM Training with Variable Sequence Length
  Curriculum" (arXiv 2405.13226)。文書を長さ (2^k) の bucket に分解し、bucket 単位で batch を
  組む (文書跨ぎの attention を無くす) + 短→長の curriculum。**8k context を 2k と同コストで
  学習、最大 6× 高速**。
- 出典: "Sequence Length Warmup" 系 (Li et al., "The Stability-Efficiency Dilemma",
  arXiv 2108.06084) — 序盤の長系列は勾配分散を増やすので短く始めると安定。
- 出典: "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics"
  (arXiv 2601.21698) — 難易度 curriculum の結果は **まちまち**。長さ以外の curriculum は期待薄。
- **arbor での結論: 見送り**。論文の 6× は連結文書に O(T²) attention を掛ける無駄を消した分だが、
  arbor は local attention が patch 内 (16 byte)、global attention は 512 patch しかなく flex の
  BlockMask で既に文書ごと block 対角。per-byte コストは BitLinear GEMM で決まり系列長に線形。
  実測 (2026-09-13, `scripts.bench_cuda`, 合成データ, compile 込み, 同一 16k bytes/micro):
  **seq 2048×micro 8 = 86.1k bytes/s、seq 8192×micro 2 = 85.6k bytes/s** (差 0.6% = 誤差)。
  2k で回しても速くならず、文脈が短い分 loss/byte が悪くなるだけ。長さ bucket packing も
  狙う無駄 (文書跨ぎ attention) が既にほぼゼロなので不要 (形状可変化のコストだけ残る)。
- 残る用途は **2B を 24GB に収めるために 2k で始める** (メモリの壁、速度ではない)。その場合の切替:
  - batch 由来の lr 補正は不要 (2k×micro8 と 8k×micro2 は同じ 524k bytes/update)。
  - `rope_theta_global` は両フェーズで同じ値 (10000) に固定。theta を変えると学習済みの位置感覚が
    壊れる。切替は「global 位置 128〜511 が未見」の純粋な長さ拡張にとどめる。
  - `--init-from` は step/optimizer/scheduler を新規にするので、8k 側 config は
    `lr` = 2k フェーズ終了時点の lr、`warmup_steps` 500〜1000 (Adam モーメントがゼロから始まる
    ので短い再 warmup で loss の跳ねを吸収)、`total_steps`/`min_lr_ratio`/`decay_end_ratio` は
    残り step と新 `lr` 基準で「元の 1 本のスケジュールの続き」になるよう換算する。
  - 8k フェーズは 20〜30% 以上取る。global の位置 128〜511 はここで初めて学習されるので、
    Llama 3 の文脈拡張 (終盤の数%) では足りない。

## 6. µP + プロキシ実験 (C) — 方針として

- 出典: Microsoft "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
  Hyperparameter Transfer" (µTransfer, arXiv 2203.03466)。GPT-3 6.7B の HP を 40M で決めて
  **全学習の 7% のコスト**。
- 出典: Cerebras "The Practitioner's Guide to the Maximal Update Parameterization"、
  "Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale MoE"
  (2026, 10T token run の lr を 1/98 コストのプロキシで決定)。
- 出典: OpenAI "GPT-4 Technical Report" (arXiv 2303.08774) — 1/1000〜1/10000 計算量の
  モデルから最終 loss を予測 (predictable scaling)。
- arbor: `ByteLM` (`arch: byte_lm`, `bitnet: true` 可) がプロキシ。lr / schedule / Muon 可否 /
  小技の ablation は全部そこで決めて 1B に転移する。**1B で試行錯誤しない**。
  µP を厳密に入れる場合は BitLinear の absmean/absmax スケーリングと width 依存の整合が要る
  ので、まずは同 width の ByteLM で「相対比較」に使うところから。

## 7. スピードランの小技 (A) — ByteLM で ablation する候補

- 出典: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (124M, 8×H100,
  45 分 → 81 秒)、[WR #82 解説](https://djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html)、
  "The Automated LLM Speedrunning Benchmark" (arXiv 2506.22419)。
- 候補: value embeddings / value residual、U-net 型 skip、logit softcap、QK-norm (BitNet は
  SubLN あり)、momentum warmup、long-short sliding window attention + window warmup。
  ReLU² は採用済み。byte vocab 260 なので FP8 head / tied embedding は効かない。
- 各 1〜3% 程度。ByteLM で 1 時間/run の ablation 対象。

## 8. patch_pooling: concat vs mean (A) — 小型 Arbor A/B で concat が勝ち、本走に採用 (2026-09-14)

`model.patch_pooling` は `concat | mean | max` の 3 値 (旧 `legacy` は廃止。static=concat /
dynamic=max に化ける二重挙動だったため、concat は static 専用として動的モードでは起動時エラー)。
問い: mean は local encoder → patch_proj (dl → dg) で patch 内 16 byte の情報を 1 ベクトルに
潰しており、global 側の学習を妨げていないか。

- **A/B 結果** (config は削除済み。再現は arbor.yaml の model で global を 768/12 heads/kv4/ffn2048/8 層に縮小し patch_pooling だけ変える、optim/schedule は bytelm_ab と同じ: 小型 Arbor、local 側は本走と同一
  (dl 768 / enc 1 / dec 2 / patch 16 / 8k)、global 768×8 層、本走 mix、lr 8e-4、同 983M bytes):

  | step | mean | concat | Δ |
  |---|---|---|---|
  | 2500 | 2.347 | 2.300 | −0.047 |
  | 5000 | 2.152 | 2.118 | −0.034 |
  | 10000 | 1.970 | 1.938 | −0.031 |
  | 15000 (最終) | 1.938 | **1.907** | **−0.031** |

  - 全 5 ドメインで concat が良い (ja_web −0.029 / wiki −0.021 / en −0.036 / code −0.037 / math −0.032)。
    train ema も 1.244 → 1.222。差は step 5000 以降 −0.03 で安定 (縮まらない)。
  - 注意: concat は patch_proj が p·dl→dg で、プロキシでは +8.8M params (73.2M → 82.0M、+12%)。
    1B では +25M (2.5%) なので、小型での差の一部はパラメータ増の寄与を含む。ただし FP の射影 1 枚で
    global 20 層の BitLinear と同じ 0.03 bpb を稼げるなら安い。
  - 1B 本走 config での実測 (`scripts.bench_cuda`, micro2×accum8, compile default):
    mean 780 ms / 10.4 GiB → concat 790 ms / 10.6 GiB (+1.2% 時間、+0.2 GiB)。
- **本走 (arbor.yaml) は `patch_pooling: concat` に変更**。旧 checkpoint (filter/cpt/sft) は
  config に patch_pooling キーが無く既定 concat で従来どおりロードされる。
- entropy patching に戻す場合は mean/max しか使えない (patch 長可変)。concat 相当が欲しければ
  「max_patch_len へ右 pad して concat」を別途実装する (未実装)。

## 9. GPU 側の無駄取り (B) — torch 2.14 の rms_norm 罠、dW 直接累積、量子化配管の融合 (2026-09-15)

nsys (`scripts/profile_training_nsys.sh`、arbor.yaml、random_bytes、accum 8 フェーズ、2 update) で
1 update の GPU 時間を kernel 名で分解し、「行列積以外」の帯域往復を減らした。
4090 固有の tile チューニングは対象外 (packed GEMM / flex の kernel 効率は触っていない)。

| 状態 | GPU ms/update | bytes/s (合成) | 主な変化 |
|---|---:|---:|---|
| torch 2.14 + 修正前 | ~750 (step_ms) | 172k | autocast 下の `F.rms_norm` が fp32 を返し全 BitLinear 入力が fp32 経路 |
| RMSNorm 修正 (6765fac) | 550 | 233k | 2.11 と同じ bf16 経路に戻る |
| (a) dW 直接累積 (26aee4e) | 486 | 254-262k | `CUDAFunctor_add` ×2080 (60ms) が消える |
| (b) 量子化配管の融合 (653208a) | 447 | 276-292k | A8 量子化 / dY amax が producer に融合、x 側 amax 廃止 |

実データ (HF streaming) でも同じ値 (269k、batch_ms 35-40ms は GPU と重畳)。loss は全段で 4 桁一致。

### (a) 勾配累積を dW GEMM の epilogue に (−60ms)

AccumulateGrad の bf16 add は param 950M × 2B × 3 (新 dW 読み・累積読み・書き) = 5.7GB/micro-step
で帯域上限 (760GB/s) に張り付いていた。`torch._scaled_mm` は beta=0 固定なので cuBLASLt を
直接呼ぶ C++ 拡張 (`src/model/csrc/fp8_wgrad_lt.cpp`、host API のみ、nvcc 不要、header/lib は
pip の nvidia/cu13) を追加し、FP8 A/B + BF16 C/D + beta=1 で累積 buffer へ足し込む。
BitLinearGroup は member の weight.grad を 1 本の連結 buffer の行 slice にする (GEMM 出力が
連続領域である必要)。beta=1 の GEMM 時間は +1.3ms/update だけ。数値は fp32 累積 + bf16 1 回丸め
(従来より丸めが 1 回少ない)。`speed.bitlinear_fused_grad_accum: false` で従来経路。

### (b) 同じ activation を別 kernel で読み直す pass を消す (−39ms)

- A8 量子化 (forward): 専用 Triton kernel → torch op。inductor が RMSNorm / ReLU² の kernel に
  融合し、正規化済み bf16 の書き出し+読み直しが消える (fwd 113.6 → 99.6ms、kernel 単体 6.7ms より大)。
- dY の行 amax (backward pass 1): Triton → torch reduction。dY の producer (residual add /
  ReLU² backward) に融合。bit 一致 (max は順序非依存)。
- x 側 FP8 tensorwise scale: x_int8 の amax/amin 2 pass → `127 * max(inv_sx)` (per-token absmax
  量子化では各行の最大が必ず ±127 なので厳密に等しい。全行が |x|<1e-5 の下限 clamp のときだけ上界)。

### (c) local 層の pointwise — (b) の融合で大半が吸収された

(b) 前の local 層 pointwise 38ms (ReLU²+norm 8.2 / A8 3.9 / dY amax 8.4 / ReLU² bwd 9.5 / norm bwd 8.0)
→ (b) 後 23ms (ReLU²+norm+A8 6.6 / ReLU² bwd+amax 9.3 / norm bwd 7.5)。残りは全て 650-870GB/s で
帯域に張り付いており、減らすには tensor を跨ぐ融合 (SubLN backward + ReLU² backward を 1 kernel、
d_h の往復 134MB ≈ 3.6ms) や local attention 専用 kernel (fmha fwd+bwd 11.8ms → ~6ms) の
custom kernel が要る。各 1% 前後なので保留。

試して効かなかったもの:
- global attention を flex → cuDNN SDPA (密 doc マスク + native GQA): 単体では fwd+bwd 255 → 168µs/層
  だが、in-model では flex backward が RMSNorm backward の template に融合されていて分離すると
  fwd +6 / bwd +7ms の逆効果 (削除済み)。
- flex の BLOCK_M/N kernel_options: 全組み合わせで差なし (latency bound は block 形状で解けない)。

### 残りの内訳 (447ms/update、accum 8)

packed ternary GEMM 33% / FP8 dW GEMM 15% / optimizer (AdamW+clip、update あたり固定) 8% /
global flex bwd (norm bwd と融合) 7% / local 層 pointwise 5% / local attention 3% / その他 pointwise。
optimizer は accum 32 で 2% に下がる。GEMM 側 (三値 ~190-250 TOPS、FP8 280 TFLOPS) は tile の話で
4090 固有チューニングの領域。

## 見送り

- **Multi-token prediction**: DeepSeek-V3 (arXiv 2412.19437) 採用。ただし "Pre-Training
  Curriculum for Multi-Token Prediction" (arXiv 2505.22757) は **1B 以下では NTP に負ける**
  (curriculum が要る) と報告。byte 直なら「次 patch の bytes」で自然だが 1B では期待薄。
- **蒸留**: Gemma 3 (arXiv 2503.19786) 1B / Llama 3.2 は大モデルから蒸留。byte 出力の強い
  teacher が存在しないので不可。
- **MoE**: Moonlight / DeepSeek。4090 の壁は VRAM (shadow weight + optimizer state) で、
  MoE は params を増やす方向なので不適。
- **データ品質分類器** (FineWeb-Edu / DCLM / Nemotron-CC, arXiv 2412.02595): 効果は大きいが
  日本語で分類器を作る手間が大きく、bpb では効きが見えにくい (下流で効く)。
- **FP8 / NVFP4 / DualPipe** (DeepSeek-V3): GPU 側。FP8 は実装済み (bitlinear-fp8-gemm)。

## 優先順位

1. 次の 2B は G_stack で 1B base から
2. anneal ×N + 重み平均

(1 の batch size warmup は ByteLM A/B で効きを確認し本走 config に採用済み。)

(2 の Muon は ByteLM A/B で AdamW に負けたため、5 の長さ curriculum / bucket packing は
計測の結果 arbor では効かないため、それぞれ優先順位から外した。)

共通のエッセンス: **大きい run は一発勝負にして決め事は全部小さいモデルで済ませる**、
**序盤は小 batch で無駄を出さない** (短文脈は arbor では効かない)、**前のモデルを捨てずに次の初期値にする**。
