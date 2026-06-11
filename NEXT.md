# 次の続き (2026-06-11 更新: Arbor v2 全面書き換え)

## 現在の状態

- **旧アーキテクチャ (third_party BLT fork + 部分 BitNet) は全廃**。
  自己完結の Arbor v2 (静的 patching 階層 Transformer × BitNet b1.58 公式レシピ) に移行。
  旧 checkpoint (step 4492) も削除済み (ユーザー指示で互換性破棄)。
- **1B 本走 (952.8M params) を fresh start 済み** (`logs/run_arbor2_1b.log`)。
  config は `configs/arbor_1b.yaml`。synthetic 実測 51.2k tok/s / VRAM 16.1GiB (B=8)。
- 再開:
  ```bash
  source scripts/env.sh
  python -u -m src.train.train --config configs/arbor_1b.yaml --resume latest \
      > logs/run_arbor2_1b.log 2>&1 &
  ```

## v2 で直したこと (arbor_llm_code_review.md の指摘対応含む)

1. BitNet を公式レシピ化: SubLN 追加 / detach トリック STE (勾配を量子化後の値で計算) /
   全 Transformer 層 BitLinear 化 (旧: global のみ, 生値で勾配, norm 無し)。
2. 静的 patching (4B/patch, MegaByte 方式) で torch.compile 常時 ON。
   因果性は tests/test_arbor.py で検証 (未来バイト→過去 logits の漏れ無し)。
3. checkpoint: 保存は compile 前モデル (_orig_mod 問題根絶) / ロード strict=True。
4. best 判定を train loss EMA に (単発 step のノイズ排除。validation best ではない)。
5. data: HF streaming resume を datasets state_dict API で正確化 (再走査スキップ廃止) /
   local mmap の 1-byte stride → block stride。
6. stub fallback / 死に config (use_flash_attention 等) / BLT 依存 / 旧 bench を削除。

## 次にやる候補

- 学習を進めて samples.txt の品質を見る (1000 step ごと)。10k step 以降で export して
  transformers から品質確認。
- val bpb: 専用 held-out ストリームでの定期 eval は未実装 (best は train EMA)。
- 推論高速化: 2 階層 KV cache (global per-patch + local per-byte)。設計は素直にできる
  (静的 patching なので)。やるなら src/infer/generate.py に。
- BitNet 推論カーネル: ternary packed (2bit) 重みで VRAM ~10x 減 + 高速化。
  microsoft/BitNet (bitnet.cpp) / T-MAC 参考。学習には不要。
- BitNet training tips の 2 段階 LR/WD スケジュール (後半 wd=0) は未実装。
  loss が伸び悩んだら検討。

## 2026-06-11 の NaN 事故 (解決済み・重要)

初回の 1B 起動で step5 から loss=nan。原因は **`global_bos` のゼロ初期化**:
- 厳密ゼロの行は RMSNorm forward で 0 のまま (情報なし) だが、backward は
  `1/sqrt(eps)≈316` 倍の増幅になる (forward 利得 0 と非対称)。
- position 0 は residual 経由でも全層ゼロのままなので、20 層 × 複数 norm で
  複利増幅 → bf16 上限 (3.4e38) を超え inf → nan。
- 修正: trunc_normal 初期化 (tests/test_arbor.py に回帰テスト)。
- 切り分けの過程で SDPA の `enable_gqa=True` も疑い、KV repeat 方式 (Llama 流) に
  変更済み (枯れた経路、速度 -4% の 49k tok/s)。こちらは原因ではなかったが維持。
- 教訓: 「学習可能 embedding/トークンのゼロ初期化」は RMSNorm 系では禁止。

## 既知の落とし穴 (引き継ぎ)

- /mnt/d は Windows FS → torch import が遅い (起動 1-2 分は正常)。
- 初回 torch.compile ~2 分。TORCHINDUCTOR_FX_GRAPH_CACHE=1 (env.sh) で 2 回目以降短縮。
- WSL2 で VRAM 超過は OOM ではなく `CUDA driver error: device not ready` で出る。
  micro_batch 8 が安全 (16 は fp32 optimizer だと溢れた)。
- HF streaming の resume は shuffle バッファの中身までは復元されない (datasets の仕様)。
- `--dry-run` 終了時の PyGILState_Release Fatal error は無害 (HF datasets スレッド起因)。
