# BitNet × BLT プロジェクト仕様書

## 概要
BLT (Byte Latent Transformer) アーキテクチャに BitNet b1.58 (ternary量子化) の考え方を統合し、トークナイザ不要・ternary量子化のLLMを RTX 4090 一枚でゼロから学習することを目標にしたプロジェクト。

現状の `BitLinear` は CUDA forward に Triton の packed ternary/int8 kernel を持つ。量子化テンソルは int8、重みは 4 ternary values / byte で保持する。CPU または Triton 不可の環境では PyTorch 参照実装へ fallback する。backward はまだ STE の参照実装で、optimizer state や勾配まで含めた完全な fused BitNet training stack ではない。

この文書は目標仕様を含む。現時点の実装済み挙動は README、`src/`、`configs/` を正とする。
`model.backend: blt` で BLT 本体の import / 構築に失敗した場合、既定ではエラーにする。stub は `model.backend: stub` または `model.allow_stub_fallback: true` を明示した場合だけ使う。

---

## 決定事項

### モデル
| 項目 | 値 |
|---|---|
| モデル種別 | BitNet b1.58系を目標にした参照実装（W1.58A8: weight 1.58bit, activation 8bit） |
| ベースアーキテクチャ | BLT (Byte Latent Transformer) |
| BitLinear適用範囲 | **Global Latent Transformerのみ**（Local Encoder/Decoderはfull precision維持） |
| 量子化方式 | absmean量子化（重み）、absmax量子化per-token（活性化） |
| tokenizer | 不要（バイト直） |
| パッチング | エントロピーベースの動的パッチング（BLT標準） |

### サイズ・規模
| 項目 | 値 |
|---|---|
| モデル規模 | **1B parameters 目標**。現状の検証は smoke 小モデル |
| context長 | **2048 bytes** |

### アーキテクチャ詳細（MS BitNet b1.58 2B4T 準拠）
- Transformer-based decoder
- Position encoding: RoPE
- FFN activation: squared ReLU (ReLU²)
- Normalization: subln（biasなし）
- Linear層: biasなし
- アーキテクチャパラメータ（1B版、要設計）:
  - 参考: BitNet 2B4Tは24層 / hidden 2048 / FFN intermediate 5632
  - 1B版はこれを縮小、または BLT の Global Latent 部分を1B規模に設計

### BitLinear化マップ
| 層 | 扱い |
|---|---|
| Global Latent Transformer: Attention Q/K/V/O projection | **BitLinear（W1.58）** |
| Global Latent Transformer: FFN（gate/up/down） | **BitLinear（W1.58）** |
| Global Latent Transformer内: 活性化 | int8（A8） |
| Local Encoder（byte→patch） | FP維持 |
| Local Decoder（patch→byte） | FP維持 |
| Cross-attention（patch↔byte境界） | FP維持 |
| Byte embedding / 出力head | FP維持 |
| LayerNorm / subln | FP維持（biasなし） |

### 学習方針
| 項目 | 値 |
|---|---|
| 学習方法 | ゼロから事前学習 + 蒸留併用 |
| マスター重み | BF16（学習時のシャドウ重み） |
| 勾配推定 | STE（Straight-Through Estimator） |
| データ | Common Crawl系（FineWeb-Edu等） |
| 教師モデル候補 | Llama, Qwen, DeepSeek, Gemma（要ライセンス精査） |

### ハードウェア
| 項目 | 値 |
|---|---|
| GPU | RTX 4090 ×1（24GB VRAM） |
| 必須最適化 | Flash Attention 2/3, BF16 mixed precision, torch.compile, gradient checkpointing, 8-bit Adam（bitsandbytes） |

### 公開
| 項目 | 値 |
|---|---|
| 公開先 | HuggingFace |

### 開発環境
| 項目 | 値 |
|---|---|
| Python仮想環境 | **venv**（`python -m venv .venv`、リポジトリ直下に作成） |
| Python | 3.10 以上（PyTorch 2.x / flash-attn 互換） |
| 依存管理 | `requirements.txt` は下限バージョン中心。再現性が必要な実験では別途 lock/constraints を生成 |
| OS | Linux（WSL2 含む）。Flash Attention は Linux 限定 |
| fork対象 | **BLT（facebookresearch/blt）を fork**。BitNet 側からは BitLinear 実装のみ移植 |

### データ取り扱い方針（メモリ常駐禁止）
| 項目 | 値 |
|---|---|
| ロード方式 | **streaming のみ**（HF datasets `streaming=True`） |
| 全件メモリ展開 | **禁止**（Common Crawl 規模を一括展開しない） |
| ローカルキャッシュ | 使う場合は **numpy memmap / mmap** で参照のみ、コピー禁止 |
| シャッフル | streaming shuffle のバッファ（既定 10000 件）に限定 |
| 文書境界 | リングバッファに `\n` を挿入し、context_length で切り出し |
| iterator state | `(samples_emitted, shard_index, byte_offset)` を保存・復元可能 |

### 学習時間短縮（速度最適化、まとめ）
| 項目 | 値 |
|---|---|
| 精度 | BF16 mixed precision（マスター重みも BF16） |
| Attention | Flash Attention 2/3 |
| Compile | 現状 config は `torch.compile(mode="default", dynamic=True)` を採用 |
| 行列乗算 | TF32 有効化、`cudnn.benchmark = True` |
| メモリ節約 | gradient checkpointing、8bit Adam（state を 1/4） |
| バッチ | micro batch + gradient accumulation で実効バッチ確保 |
| カーネル | RMSNorm/subln, attention は fused 採用、可能なら squared ReLU も fused |
| I/O | DataLoader `num_workers>0` + `prefetch_factor=4` + `pin_memory=True` |
| Checkpoint | 安全優先の同期保存、safetensors |
| ロギング | tokens/sec を直近 N step 移動平均で監視 |

### 学習の中断・再開
| 項目 | 値 |
|---|---|
| チェックポイント方式 | **stepベース**（N step毎に保存） |
| 保持ポリシー | 直近 K 個 のローテーション + **best loss** + **final** + 長期保存を保持 |
| 保存先 | リポジトリ外の外部ディレクトリ（`CHECKPOINT_DIR` で指定、デフォルト `./checkpoints/`） |
| 保存内容 | model（BF16シャドウ重み）、optimizer state、scheduler state、RNG state（torch/cuda/numpy/python）、データローダのiterator state、global step、wandb run id、設定ハッシュ |
| アトミック保存 | `<step>.tmp/` へ書き出して fsync 後に rename。同一 step の上書きは禁止 |
| 再開 | `--resume from <ckpt>` または `--resume latest` で完全復元（同seed・同loss曲線継続） |
| 中断トリガ | SIGINT / SIGTERM を捕捉し、現stepで安全保存してから終了 |

---

## 未定事項

| カテゴリ | 内容 |
|---|---|
| 学習バイト数 | 未定 |
| 蒸留の比重 | 事前学習混入 / SFT のみ / 比率 |
| 教師モデルの最終選定 | ライセンス考慮の上で絞り込み（Llamaは命名制約あり、Gemma 2/3は蒸留制約あり、Gemma 4/Qwen/DeepSeekは寛容） |
| データ配分 | FineWeb-Edu / RedPajama / 日本語コーパス / 蒸留データの比率 |
| 自モデルのライセンス | MIT / Apache 2.0 / OpenRAIL のどれか |
| 自モデル名 | 教師選定後 |
| クラウド併用 | 4090一枚で完結 or クラウドH100併用 |
| 最終目的 | 汎用チャット / 特定ドメイン / 日本語特化 / 研究目的 |

---

## 実装タスク（Codex向け）

### Phase 1: 環境構築
1. **venv 作成**
   ```bash
   cd /mnt/d/develop/arbor-llm
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -U pip wheel setuptools
   ```
2. **PyTorch 2.x + CUDA インストール**（CUDA 12.1想定。WSL2 の場合は Windows 側 NVIDIA driver があれば OK）
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. **依存ライブラリ**: `transformers`, `flash-attn`（事前に `pip install packaging ninja`）, `bitsandbytes`, `accelerate`, `datasets`, `wandb`, `safetensors`, `sentencepiece`（teacher側で必要な場合）
4. `requirements.txt` は下限バージョン中心で管理、`.venv/` は `.gitignore` に追加
5. BLT公式リポジトリ参照: https://github.com/facebookresearch/blt

### Phase 2: BLT基盤のfork・再現
1. **BLT公式（facebookresearch/blt）を fork** し、自リポジトリにサブモジュールまたはコピーで取り込む
   - ライセンス（FAIR Noncommercial が含まれる場合あり）を確認の上で派生物のライセンスを決定
2. 1B規模へ縮小（layer / hidden / FFN ratio を再設計）
3. 動作確認（小データでoverfit確認、NaN/OOM確認）

### Phase 3: BitLinear統合
1. Global Latent Transformer内の `nn.Linear` を `BitLinear` に置換
2. BitLinear実装:
   - forward時にabsmean量子化でternary化
   - 活性化をabsmaxでint8量子化
   - backwardはSTE
   - シャドウ重みはBF16保持
3. Local Encoder/Decoder は標準 `nn.Linear` のまま

### Phase 4: 学習パイプライン
1. データローダ（FineWeb-Edu等のpre-tokenize不要、バイト直読み、resumable な iterator）
2. シーケンスパッキング実装
3. **速度最適化（RTX 4090で1B学習を現実的時間に収めるため必須）**:
   - Flash Attention 2/3
   - gradient checkpointing（VRAM-throughput トレードオフ、layer 単位）
   - 8-bit Adam（bitsandbytes、optimizer state を 1/4 に圧縮）
   - BF16 mixed precision（マスター重みも BF16、A100 以降の TF32 は不要）
   - **torch.compile**（現状は mode="default", dynamic=True。max-autotune は未採用）
   - **TF32 有効化 / cudnn benchmark = True**
   - **micro-batch + gradient accumulation**（global batch を 4090 のVRAMに合わせ分割）
   - **fused kernels**（RMSNorm/subln, squared ReLU を可能なら fused 化）
   - データ I/O は **mmap + 非同期 prefetch**、CPU 側でボトルネック化させない
   - 4090 は **NVLink なし**のため、将来分散時は ZeRO-2 程度を上限と想定
4. context長 2048 bytes
5. ロギング（wandb）、**定期チェックポイント（step ベース）**

### Phase 4.5: チェックポイント・中断/再開
1. **保存トリガ**
   - 毎 `save_every_steps`（例: 1000 step）
   - SIGINT (Ctrl+C) / SIGTERM 受信時に即時セーフセーブ
   - best loss 更新時は best 候補として扱う。実保存は定期 / 中断 / 最終 step に限定
2. **保存形式**
   - 重みは `safetensors`、その他状態は `torch.save` で `state.pt`
   - 1 checkpoint = 1 ディレクトリ:
     ```
     checkpoints/
       step_000010000/
         model.safetensors          # BF16 シャドウ重み（ternary化前）
         optimizer.pt               # 8bit Adam state
         scheduler.pt
         rng.pt                     # torch/cuda/numpy/python の RNG state
         dataloader.pt              # iterator 位置・shard index
         meta.json                  # global_step, wandb_run_id, config_hash, git_sha
       step_000010000.tmp/          # 書き込み中（完了で rename）
       best/                        # symlink → step_xxx
       latest/                      # symlink → step_xxx
     ```
   - **アトミック保存**: `<step>.tmp/` に書いて fsync 後に `os.rename` で確定。同一 step の上書きは禁止
   - 保持ポリシー: 直近 K=3 + best + final + 1万step毎の長期保存
3. **再開**
   - `--resume latest` / `--resume best` / `--resume <path>` を CLI で受ける
   - 復元順: config → model → optimizer → scheduler → RNG → dataloader → global_step
   - 再開後の loss が中断前と連続することを smoke test で確認
4. **シグナルハンドラ**
   - `signal.signal(SIGINT/SIGTERM, handler)` で flag を立て、次の step 境界で安全保存→ exit
   - 二重 Ctrl+C で強制終了パス

### Phase 5: 評価
1. perplexity測定
2. 既存ベンチマーク（バイトレベル対応のもの）
3. 1B BitNet b1.58 2B4T 等との比較

---

## ディレクトリ構成（想定）

```
arbor-llm/
├── .venv/                          # venv（git ignore）
├── .gitignore
├── requirements.txt
├── bitnet_blt_project_spec.md
├── README.md
├── src/
│   ├── model/
│   │   ├── bitlinear.py            # BitLinear (W1.58 / A8 / STE)
│   │   ├── global_latent.py        # BitLinear 統合済み Global Latent Transformer
│   │   ├── local_encoder.py        # FP維持
│   │   ├── local_decoder.py        # FP維持
│   │   └── arbor_blt.py            # 全体組み立て
│   ├── data/
│   │   ├── byte_dataset.py         # バイト直 + resumable iterator
│   │   └── packing.py
│   ├── train/
│   │   ├── train.py                # エントリポイント
│   │   ├── checkpoint.py           # アトミック保存・復元
│   │   ├── signals.py              # SIGINT/SIGTERM ハンドラ
│   │   └── optim.py                # 8bit Adam ラッパ
│   └── eval/
│       └── perplexity.py
├── configs/
│   └── arbor_1b.yaml               # layers/hidden/FFN/lr/save_every_steps 等
└── checkpoints/                    # 既定の外部保存先（git ignore、別ディスク推奨）
```

---

## 参考資料
- BitNet b1.58 2B4T Technical Report: https://arxiv.org/abs/2504.12285
- BitNet (元論文): https://arxiv.org/abs/2310.11453
- BLT論文: https://arxiv.org/abs/2412.09871
- BLT公式実装: https://github.com/facebookresearch/blt
- MS BitNet公式: https://github.com/microsoft/BitNet
- HuggingFace BitNet 2B4T: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

---

## 注意点

### 事実
- BLT × BitNet の組み合わせを採用した既存公開モデルは見当たらない
- BitNet b1.58の学習にはBF16のシャドウ重みが必須（量子化はforward時のみ、勾配計算用に高精度重みを保持）
- MS BitNet b1.58 2B4Tの推論時メモリは約0.4GB、これは推論用packed weightのサイズであり学習時のメモリとは別

### 推測・要検証
- RTX 4090 ×1 での1B BitNet+BLT学習の所要時間（実測されていない、雑計算ベース）
- 小規模BitNetでhidden sizeを倍にする必要があるとの報告は他研究で言及されているが、1B規模・BLT組み合わせでの妥当性は未検証
- BLT × BitNet の組み合わせの収束性・性能特性
- `torch.compile` と BitLinear（量子化 forward）の相性 — graph break が出る可能性、出た場合は eager fallback
- WSL2 環境での Flash Attention 2/3 ビルドの可否（CUDA toolkit / nvcc バージョン整合）
- BLT 公式のライセンス（FAIR Noncommercial Research License）— HuggingFace公開時の派生物ライセンス選定に影響
