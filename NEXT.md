# 次の続き (2026-06-03 更新)

## ★解決済み: 1B 本走 起動成功 (RAM 17GB + dynamic compile)

### 結論 (2026-06-03 実走で確定)
RAM を 9GB→**17GB** に増設後、`compile_dynamic: true` で **1B 本走の起動に成功**。
- loss 下降確認: step20=5.76 → step160=2.94 (順調)
- 定常 **~31k tok/s** / RAM 空き ~1.5GB で安定 / recompile 1件で固定 / GPU util ~89%
- VRAM ~15.6GB (dynamic は static より多いが 24GB 内で余裕)

### ★compile 戦略の実測結論 (重要・旧教訓を更新)
1B BLT では下記が確定。**起動env**: `TORCHINDUCTOR_COMPILE_THREADS=12` + `TORCHINDUCTOR_FX_GRAPH_CACHE=1`。
| 戦略 | 結果 |
|---|---|
| **static** (compile_dynamic:false) | **起動不能**。`patch_ids_from_lengths` 等が patch_lengths 形状(~200種)で毎step recompile → cache64 溢れ → **step20 すら未到達** |
| **dynamic** (compile_dynamic:true) | **○ 起動成功・定常 ~31k tok/s**。symbolic shape で recompile ほぼ消滅(1件のみ eager fallback)。**現行採用** |
| **eager** (torch_compile:false) | 未実走だが bench no-compile bs=8 = **35.5k** → dynamic より速い見込み。待ちゼロ・最シンプル |

- 旧教訓「dynamic は初回15分超で非現実的」は **9GB/20スレ時代の話**。17GB/12スレでは数分で起動でき逆転。
- **意外な点: dynamic(31k) は eager(35.5k bench) より遅い**。symbolic kernel が static特化より非最適 + 1フレーム eager fallback のため。
  → **次に速度を詰めるなら eager を実走比較する価値あり** (compile 待ちもゼロ)。bench 59.6k は固定形状での値で BLT 動的patchingでは出ない。
- mini (run_blt_fineweb.yaml) は hidden256 と小さく既に学習成功済み (loss 5.49→2.79)。

### 起動コマンド (再現用)
```bash
cd /mnt/d/develop/arbor-llm && source scripts/env.sh
export TORCHINDUCTOR_COMPILE_THREADS=12   # 9GB時代に20が枯渇主因。17GBでは12が安全
export TORCHINDUCTOR_FX_GRAPH_CACHE=1     # 2回目以降の起動を短縮 (/tmp/torchinductor_hi)
python -u -m src.train.train --config configs/arbor_1b.yaml > /tmp/run_1b_dyn.log 2>&1 &
```
確認ポイント: `step=` 出力 / RAM 空き(`free -g`、~1.5GB は正常) / recompile 数(`grep -c cache_size_limit`、1で固定が正常) / tok/s。

### 現状の arbor_1b.yaml (要コミット)
- model: hidden 2048 / 22層 / GQA(kv4) / context 2048
- data: num_workers 0, shuffle 4000 (RAM 17GB でも暫定維持中。余裕あれば 4→2workers / shuffle 上げ検証可)
- speed: torch_compile true / mode default / **compile_dynamic true** / cache 64 / bs 8 / accum 8
- optim: 8bit adamw, lr 3e-4, total_steps 200000

### 残課題
- RAM 17GB は spec 目標 24GB に未達。`num_workers>0` を試すなら 1.5GB しか空きが無い点に注意 (枯渇リスク)。
- 速度を詰めるなら **eager 実走で dynamic(31k) と比較** → 速ければ eager に切替。

---

## ★データ構成: 日本語主軸の混合に変更 (2026-06-03)

### 方針 (ユーザ指示)
**日本語を半分以上 + news を含める**。蒸留は別フェーズ (教師選定が前提) として後回し。

### 確定した混合 (configs/arbor_1b.yaml の `data.sources`)
| ソース | 言語/種別 | weight | 備考 |
|---|---|---|---|
| `HuggingFaceFW/fineweb-2` (name `jpn_Jpan`) | 日本語 web | **0.60** | NHK等 news 含む・高品質 |
| `vblagoje/cc_news` | 英語 news | 0.15 | CommonCrawl News |
| `HuggingFaceFW/fineweb-edu` | 英語 edu | 0.25 | 高品質土台 |

- weight は確率に正規化し `interleave_datasets(probabilities=..., stopping_strategy="all_exhausted")` で**行レベル混合** (byte_dataset.py)。
- **実測 byte 比率: 日本語 66.1%** (row重み60%より高い。日本語UTF-8が約3byte/文字でバイト寄与大)。狙い通り半分以上。
- バイト直読みなので **vocab 変更不要** (日本語UTF-8も 0-255 バイトに収まる)。

### 実装 (src/data/byte_dataset.py)
- `ByteStreamDataset` に `sources: list[dict]` を追加 (単一 `source` str も後方互換)。
- 各ソースを `rename_column→select_columns(["text"])` で正規化してから interleave (スキーマ衝突回避)。
- 各 dict は `{path, name?, weight?, text_column?, split?}`。
- `build_byte_dataloader` は `cfg.get("sources")` 優先、無ければ従来 `cfg["source"]`。
- seed=42 固定なので skip ベース resume でも同じ混合順序を再現。

### smoke test 済み (認証不要 streaming OK)
- ✅ fineweb-edu / fineweb-2 jpn_Jpan / cc_news / range3/cc100-ja
- ❌ llm-jp-corpus-v3 (存在せず) / OSCAR-2301 (gated 要認証) / izumi-lab/cc100-ja (存在せず)
- 注: cc_news で稀に `[Errno 9] Bad file descriptor` → HF datasets が自動リトライ(5回)で無害。

### RAM 注意
- 3ソース分の shuffle バッファを持つため `shuffle_buffer 4000→2000` に縮小。`num_workers 0` 維持。
- 起動後 `free -g` で枯渇しないか要確認 (日本語ページは1行が大きい)。

### 起動 (英語版を停止してから fresh restart)
```bash
pkill -9 -f src.train.train; pkill -9 -f compile_worker   # 英語版停止
source scripts/env.sh
export TORCHINDUCTOR_COMPILE_THREADS=12 TORCHINDUCTOR_FX_GRAPH_CACHE=1
python -u -m src.train.train --config configs/arbor_1b.yaml > /tmp/run_1b_ja.log 2>&1 &
```
- checkpoint dir は `./checkpoints` (flat, 元のまま)。英語版 step_1000 は削除済みなので衝突しない。
- 英語版は step1080 (loss 1.44, 英語のみ) で停止・破棄 (checkpoint 5GB も削除)。checkpoint保存/prune は step1000 で正常動作を確認済み。

### 未実施 / 次の判断
- **日本語版 1B 本走の起動はまだ (ユーザ確認待ち)**。起動可なら上記コマンド。
- 蒸留 (distillation): 教師モデル未選定 (spec: Qwen/Gemma系が license寛容)。やるなら別途 KD パイプライン実装が必要。
- 日本語 news 専用コーパスは streaming 可能な ungated なものが見つからず → 日本語 news は fineweb-2 ja 内に含まれる web news で代替。

---

## 環境立ち上げ
```bash
cd /mnt/d/develop/arbor-llm
source scripts/env.sh    # venv + micromamba gcc + PYTHONPATH + expandable_segments
```
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を env.sh と各エントリ
  (train/bench/check_fallback) の torch import 前に設定済み (VRAM 断片化抑止)。
- System Memory Fallback は OFF 確定 (超過確保は WSL2 では device-not-ready で失敗)。

## 確定した実測 (RTX 4090, 1B BLT × BitLinear, seq=2048, grad_ckpt+sdpa)
| 設定 | tok/s | VRAM |
|---|---|---|
| no-compile bs=6 | 33.6k | 14.6 GB |
| no-compile bs=8 | 35.5k | 18.0 GB (grad_ckpt上限近辺) |
| **compile=default bs=8** | **59.6k** | **11.1 GB** (採用・効率最良) |
| compile=default bs=16 | 61.9k | 18.5 GB (最速だが余裕少) |
| compile=default bs=24 | OOM | — |
| max-autotune | default 比 +3%のみ・autotune遅い → default 採用 |

- **torch.compile だけで +78% (33.6k→59.6k) かつ VRAM も減** → 目標 50k 達成。
- bs=10/12/24 は容量不足。grad_ckpt 無し compile 込みの上限は bs=16 (18.5GB)。

## config (確定済み)
- `arbor_1b.yaml`: compile=default, micro_batch=8, accum=8 (実効64)
- `run_blt_fineweb.yaml`: compile=default (旧 OFF→ON), micro_batch=8
  - compile は初回コンパイルで数分待ち。不安定なら false に戻す。

## flash-attn → 速度目的では不要 (調査結論)
- 2.8.3 をソースビルド成功・インストール済み (venv)。
- **SDPA(`F.scaled_dot_product_attention`) が BLT 形状(head_dim=96,bf16,causal)で
  既に FlashAttention(FA2相当) backend を使用** (`flash_sdp_enabled=True` 確認済)。
  → 明示 flash_attn を base_transformer に挿しても同カーネルを呼ぶだけで利得ほぼ無し。
  base_transformer.py は sdpa/xformers/flex のみ対応 (fmha 分岐なし、追加には fork 改造要)。
- 再検討の価値があるのは varlen packing / sliding window を入れる時だけ。

## ビルド環境 (導入済み)
- nvcc 12.0 (apt `nvidia-cuda-toolkit`), host compiler は gcc-12/g++-12。
- flash-attn ビルド時の決まり: env.sh は使わず venv だけ activate し
  `CUDA_HOME=/usr CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12`
  `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12" FLASH_ATTN_CUDA_ARCHS=89 MAX_JOBS=1`
  (RAM 9GB / swap 9GB なので MAX_JOBS=1 必須。env.sh の gcc15 だと nvcc12 が弾く)

## 次にやること（優先度は目的次第）

### A. 学習を回したい → 実走 (FineWeb)
- **mini 実走 (run_blt_fineweb.yaml, 200 step) 成功済み**:
  loss 5.49→2.79 と下降、tok/s ~45k、checkpoint/正常終了 OK。
- 次は **1B 本走** `python -m src.train.train --config configs/arbor_1b.yaml`
  - 初回 compile の数分待ち。SIGINT で安全保存。HF は初回 shard DL + shuffle
    buffer 充填で最初の step まで数分の CPU バウンド待ちがある (固まりではない)。

#### 本走前に対処価値のある課題
1. **torch.compile の動的形状 recompile** (mini で顕在化 → 機能対策済み):
   BLT の動的パッチング (patching_mode=space) で seq 長が毎 step 変動し、
   dynamo が `cache_size_limit (8)` に到達して一部 eager fallback していた。
   対策を config 化 (train.py): `compile_dynamic` (None/true/false) と
   `dynamo_cache_size_limit`。両 config を **compile_dynamic: true / cache 64** に確定。
   - mini 検証: dynamic=true / static+cache64 とも **recompile 警告 0・loss 同一・正常終了**。
   - ただし mini の tok/s は **HF streaming の I/O 律速でノイズだらけ**(3パターン
     とも似た乱高下波形)で、compile 戦略の速度差は判定不能。
   - **速度の最終判断は 1B 本走 (計算律速) で dynamic vs static を実測して決める**。
     1B は patch 数の値域が広い(~300-512≈200種)ので形状非依存の dynamic が有利な見込み。
2. **checkpoint prune バグは修正済み** (b952df3):
   `_prune` の resolve 不一致で keep 対象を誤削除していた + 終了前に prune
   daemon を join するよう修正。単体テスト PASS。

#### 既知の終了時クラッシュ (無害)
- `--dry-run` (1 step 即終了) 時のみ `PyGILState_Release` Fatal error。
  HF datasets streaming の非同期スレッドが finalize に残るため。通常の本走
  (200 step 完走) では発生せず、checkpoint もアトミックなので実害なし。

### B. 推論・省メモリ重視 → packed ternary kernel
- BitNet b1.58 = 重み ternary {-1,0,+1} (1.58bit)。
- 現状は reference 実装 (bf16 シャドウ→量子化→通常 bf16 matmul、利得なし)。
- packed kernel: ternary を ~2bit/5値1byte に詰め、乗算なし(加減算/LUT)の専用
  CUDA kernel で matmul → VRAM ~8x減・高速。microsoft/BitNet, T-MAC 参考。
- **注意: 学習は STE で bf16 シャドウ重み必須なので利得は主に推論/メモリ**。
  学習スループット向上目的なら優先度低い。

## 既知の落とし穴
- /mnt/d は Windows FS → torch import 遅い (10-20s)
- `tee` は block buffering。background は `python -u` + redirect。
- 学習中の signal は CUDA カーネルでブロックされ handler 起動が数 step 遅延 (許容)
- BLT は `non_linearity="swiglu"` 固定 → ReLU² は `src/model/ffn.swap_swiglu_to_relu2` で差し替え
- WSL2 で VRAM 超過確保は OOM ではなく `device not ready` で出る
