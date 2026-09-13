# 学習効率化メモ (GPU 増設以外の手口)

作成: 2026-09-13。公開ラボ (DeepSeek / Moonshot / Meta / Google / Ai2 / Microsoft) の技術報告と
nanoGPT スピードランから、arbor (4090 ×1、1B BitNet b1.58、byte 直、8k context) に
効きそうなものを抽出した。OpenAI / Anthropic 自身は学習効率の論文を出していない
(OpenAI が公開しているのは GPT-4 の loss を 1/10000 計算量のプロキシから予測した話のみ)。

分類:

- **A. 同じ loss に到達する token 数を減らす** (token efficiency)
- **B. token あたりの計算量を減らす**
- **C. 小さい実験で当てて、大きい run で外さない** (experiment efficiency)

## 1. batch size warmup (A) — 実装済み

`speed.grad_accum_schedule` (`src/train/grad_accum.py`)。詳細は configs/arbor.yaml のコメント参照。

- 出典: Ai2 "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch
  Language Model Training" (arXiv 2505.23971) / [Ai2 blog](https://allenai.org/blog/critical-batch-size)。
  OLMo 1B で同 loss を **43% 少ない optimizer step** で達成。batch を倍にするとき lr を √2 倍
  (square-root scaling rule)。臨界 batch は checkpoint から batch 違いの枝を Δstep 学習して
  「小さい batch 全てに loss で劣らない最大の batch」として測る (branched training)。
- 関連: "How to Set the Batch Size for Large-Scale Pre-training" (arXiv 2601.05034) —
  WSD スケジュールと組み合わせ、固定 batch より増加 batch の方が loss/下流とも良い。
- arbor での注意: 序盤の bytes/step が減るので `total_steps` の bytes 換算が変わる。予算を
  bytes で持つなら再計算する。本走投入前に ByteLM (`bitnet: true`) で固定 accum vs schedule を
  1 時間ずつ比べて効きを確認する。

## 2. Muon optimizer (A) — ByteLM で検証してから

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
- arbor での手順:
  1. `src/train/optim.py` に `Muon` を追加 (~80 行): momentum → Newton-Schulz 5 反復 →
     Moonshot のスケール → 既存 `_apply_update` で stochastic rounding 込みの書き戻し。
  2. 2D の BitLinear 重みだけ Muon。embedding / head / RMSNorm / `patch_proj` /
     `global_to_local` は AdamW のまま (Moonshot / スピードランと同じ分担)。
  3. `configs/entropy_lm.yaml` を `bitnet: true` にして adamw vs muon を同 step 数で比較
     (19M, 15k step ≈ 1 時間/run)。
  4. 副産物: 二次モーメント不要で optimizer state が半分 → 24GB の VRAM に効く。
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

1. batch size warmup (実装済み) → ByteLM で効き確認 → 次 run に投入
2. Muon を ByteLM で検証 (本走停止中)
3. 次の 2B は G_stack で 1B base から
4. anneal ×N + 重み平均

(5 の長さ curriculum / bucket packing は計測の結果 arbor では効かないため優先順位から外した。)

共通のエッセンス: **大きい run は一発勝負にして決め事は全部小さいモデルで済ませる**、
**序盤は小 batch で無駄を出さない** (短文脈は arbor では効かない)、**前のモデルを捨てずに次の初期値にする**。
