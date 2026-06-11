# Issues

対応済みの項目は削除済み。(2026-06-11 v2 移行時に全面整理)

## 生成がまだカーネル起動レイテンシ律速 (27 B/s)

### 状況

2 階層 KV cache (ArborByteGenerator) + BitLinear 推論凍結 + fused RMSNorm で
7.0 → 27.2 B/s (4090/WSL2, 1B)。プロファイル上、計算自体は軽く、残りは
「1 step あたり数百個の小さい CUDA カーネル起動 (WSL2 では 1 個 ~15-100us)
+ PyTorch の Python ディスパッチ」が支配的。RoPE (~430us/層) も手書きの
複数カーネル。

### 対応案

torch.compile / CUDA Graph で step 関数を融合する (初回コンパイル数分の
トレードオフ)。KV cache の cat を固定長バッファ + 書き込み位置方式に変えて
形状を固定すれば reduce-overhead (CUDA Graph) まで効かせられる。
Windows ネイティブや素の Linux なら起動レイテンシ自体も下がる。

## validation loss が無い (best = train loss EMA)

### 状況

`best` symlink は train loss の EMA (logging.best_ema_decay) で判定している。
汎化性能のチェックは samples.txt の目視のみ。

### 対応案

- held-out 用の HF streaming ソース (学習と重複しない split/shard) を
  `data.val_sources` として追加し、checkpoint 時に `src/eval/perplexity.py` で
  bpb を測って meta に記録する。

## BitNet 推論カーネル (packed ternary) 未実装

### 状況

学習は BF16 シャドウ重みの QAT で、推論も bf16 のまま on-the-fly 量子化している。
BitNet の本来の利点 (重み ~10x 圧縮・乗算無し matmul) は推論側で未回収。

### 対応案

- 学習完了後、ternary を 2bit pack した重み + int8 GEMM カーネルで推論専用化。
  microsoft/BitNet (bitnet.cpp) / T-MAC が参考。
- HF エクスポートとは別形式になる (transformers 互換は bf16 のまま維持)。

## BitNet training tips の 2 段階スケジュール未実装

### 状況

公式 training tips は「後半で weight decay を 0 にし LR を下げる」2 段階を推奨。
現在は cosine + 一定 wd 0.1。

### 対応案

loss が伸び悩んだ時点で `src/train/optim.py` の scheduler に 2 段階版を追加する。

## checkpoint 保存時間がログに出ない

### 状況

保存は同期 (async_save: false 相当の動作)。1B では optimizer state 込みで
数 GB 書くため、save step の tok/s が一時的に落ちる。

### 対応案

- save にかかった秒数をログに出す。
- 書き込み先が /mnt/d (Windows FS) だと遅い。ext4 側への退避も検討。

## HF streaming resume は shuffle バッファまでは復元しない

### 状況

`datasets` の state_dict API はストリーム位置を正確に復元するが、shuffle
バッファの中身は破棄され再充填される (ライブラリ仕様)。resume 直後の
データ順は中断しなかった場合と完全一致はしない (重複・欠落は無い)。

### 対応

仕様として許容。厳密一致が必要になったら shuffle を行レベル mix の外に出す。
