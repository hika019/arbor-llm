# 再起後の続き

## 環境立ち上げ
```bash
cd /mnt/d/develop/arbor-llm
source scripts/env.sh    # venv + micromamba gcc + PYTHONPATH 一発設定
```

## 直前までの状態（コミット履歴）
```
b5ff535 FineWeb-Edu streaming + 8bit Adam + 200 step + 1B 試算
8aaddcc 実 BLT × BitLinear smoke
c476451 end-to-end smoke (train+resume+eval+SIGINT)
bd0cd9f initial skeleton
```
未コミット: ReLU² FFN (`src/model/ffn.py`)、`arbor_blt.py` の swap 統合、
bench_1b.py の変更、scripts/size_1b.py。先に一度コミットすると良い。

## 直前までの実測 (RTX 4090, 1B BLT × BitLinear, seq=2048)
| 設定 | tok/s | VRAM |
|---|---|---|
| sdpa + grad_ckpt + bs=1 | 9.6k | 5.54 GB |
| sdpa + grad_ckpt + bs=2 | 16.4k | 7.25 GB |
| sdpa + grad_ckpt + bs=4 | 25.9k | 10.80 GB |
| sdpa + grad_ckpt + bs=8 | **33.6k** | **18.17 GB** |
| sdpa + bs=1 (no ckpt) | 7.0k | 5.50 GB |
| sdpa + grad_ckpt + bs=1 + compile | 17.1k (+78%) | 5.48 GB |
| xformers | エラー (sliding_window assert) |
| bs=12 / compile=max-autotune | 計測中断 (ベンチ kill) |

## System Memory Fallback の確認結果
- 26 GB tensor 確保→成功 (fallback 一部効いている可能性)
- 40 GB tensor 確保→**OOM** (driver メッセージ: "GPU 0 has a total capacity of 22.49 GiB")
- 再起後にもう一度 `python scripts/check_fallback.py` で再確認するべき
  (check_fallback.py をこの後作成。再起前のスクリプトを残す)

## 再起後やること（優先度順）

### 1. fallback OFF の再確認
```bash
source scripts/env.sh
python scripts/check_fallback.py
```
40GB が即 OOM なら OK。26GB で時間かかるようなら fallback まだ on。

### 2. 未コミットを保存
```bash
git -C /mnt/d/develop/arbor-llm add -A
git -C /mnt/d/develop/arbor-llm commit -m "feat: ReLU² FFN を BLT global に統合 + 1B bench スクリプト"
```

### 3. 1B bench 再走 (compile + bs スケーリングの境界探し)
```bash
python scripts/bench_1b.py --seq 2048
```
- bs=8 + compile=default
- bs=8 + compile=max-autotune
- bs=12 (fallback OFF なら境界)

### 4. xformers attn_impl のエラー原因解消
`local_models.py:260` で `sliding_window is not None` assert に当たる。
回避案:
- BLT の `LocalModelArgs` に `sliding_window=2048` を渡す
- もしくは `attn_impl="sdpa"` で固定 (現状こちらで動作)

### 5. cross-attention 復活トライ
FlexAttention 必須なので, BLT 側コードを fork し
`assert mask is None or isinstance(mask, BlockMask)` を緩める
or BlockMask を構築する。優先度は速度最適化より下。

### 6. 1B 本走の準備
- packed ternary kernel (BitLinear の真の利得 8x を取りに行く)
  現状は BF16 シャドウ→量子化→fp matmul の reference 実装。
  microsoft/BitNet の C++ kernel を参考に CUDA kernel を書く。
- flash-attn pip install (SDPA より 1.5-2x 期待)
  ```bash
  pip install flash-attn --no-build-isolation
  ```

## 既知の落とし穴
- /mnt/d は Windows ファイルシステム → torch import 遅い (10-20s)
- `tee /tmp/file.log` は block buffering で、終わるまで file は空
- 学習中の signal は CUDA カーネルでブロックされるため、handler 起動は数 step 遅延 (許容)
- BLT は `non_linearity="swiglu"` 固定 (param のみ。実装は `F.silu(x1)*x3` ハード) →
  ReLU² 化は `src/model/ffn.swap_swiglu_to_relu2` で構築後に差し替える

## GPU 効率の評価メモ
- bs=8 で 33.6k tok/s → 約 50 TFLOPS → 4090 BF16 peak 165 TFLOPS の **MFU 30%**
- 改善余地: torch.compile (×1.5-2)、flash-attn (×1.2-1.5)、packed ternary (×2-4)
- 目標 20k tok/s は既に達成。50k tok/s が次の目安
