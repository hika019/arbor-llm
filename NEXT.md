# 次の続き (2026-06-03 更新)

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
