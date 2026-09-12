# 学習高速化の実装確認とAdamW融合 (2026-09-12)

## 実装監査で見つかった問題

`arbor_bitnet_launch_bound_investigation.md` のCUDA Graphs、固定grad buffer、
packed kernel / autotuneはコードに存在する。しかし学習起動時のimportに不具合があった。
`clear_observed_packed_tune_keys` と `observed_packed_tune_keys` は
`src.model.bitlinear_tuning` に定義されているのに、`train.main` は
`src.model.bitlinear` からimportしていた。広い `except Exception` がImportErrorを
握りつぶし、QKV/gate-up融合、低bit mode設定、重みcache設定をすべてスキップする。
そのためconfigがternaryでも実行は通常のfake-quant/BF16になる。

本変更では正しいmoduleからimportし、初期化を `configure_training_bitnet` にまとめた。
要求した演算を初期化できなければ例外を伝播する。CPUでもcache有効化を検証し、
import失敗を握りつぶさない回帰テストを追加した。起動ログの
`bitnet_weight_cache` と `bitlinear_fp8` で実際の経路を確認できる。

## AdamWの演算融合

既定のAdamWFP32はtensorごとに勾配cast、moment更新、weight decay、sqrt/div/add、
update cast、parameter更新を個別発行していた。融合版はmomentとparameter更新を
1 kernel/tensorで処理し、中間tensorのGPUメモリ往復とkernel起動を削減する。
Tensor Core、SM数、GPU機種名に依存せず、1要素ごとの算術をTritonで並列化する。
汎用の1024要素blockを使い、LRとbias correctionはruntime引数なので毎step再compileしない。

momentはFP32、parameterは元dtypeのまま。既存実装のweight decay直後の丸め、
updateをparameter dtypeへcastしてから減算する丸めも残す。
単純な標準fused AdamWへの置換では、この丸め契約やFP32 momentを保証できない。
FP32演算は除算実装による小さな丸め差を許容し、更新とmomentを複数step比較する。

`optim.fp32_backend: auto | eager | triton`。
autoは対応する連続CUDA tensorで融合し、それ以外は同じAdamWのeager実装を使う。
ROCmはCUDA用libdeviceのRN除算/sqrtがないため、autoではeagerを使う。
tritonを指定して非対応ならエラー。backendはcheckpoint stateに保存しないので、
既存checkpointをconfigで切り替えて再開できる。
resume検証で、旧load_state_dictがFP32 momentをいったんBF16へ丸めてからFP32へ
戻していた問題も修正した。保存された元のFP32値から復元する。

## 最新技術の調査と今回の採否

2026-09-12時点で公式release一覧の最新はPyTorch 2.14.0 (9月2日)。
ローカルの検証環境はPyTorch 2.11.0+cu128 / Triton 3.6.0。
以下は公式情報からの候補整理であり、この環境での性能測定値ではない。

| 技術 | Arborへの適用判断 |
|---|---|
| PyTorch 2.14 NVGEMM / CuTeDSL、epilogue融合、ATen/Tritonと併せたautotune | GEMM候補として有望。ただし独自packed-2bit custom opの中身は自動で置き換わらない。別環境で既存STE・FP8 dW・Graphの回帰確認が必要。 |
| PyTorch 2.12 accelerator.Graph、Graph trace annotation | 前者はdevice間のAPI統一、後者はGraph内の性能診断に有用。APIを変更するだけでGPU計算が速くなるわけではない。 |
| PyTorch 2.13 LinearCrossEntropyLoss | 大語彙でのlogits保存削減向け。Arborは語彙260なので、大語彙LLMと同じ効果は期待しない。 |
| Triton block-scaled FP4/FP8 | NVIDIA CC10 / AMD CDNA4の専用演算が前提。W1.58/A8学習の汎用高速化として一律採用しない。 |
| optimizerのcompile / elementwise融合 | メモリ往復・起動数の削減が直接効く箇所。今回は既存AdamWの丸め位置を明示した融合kernelを実装。 |

一次資料:

- [PyTorch 2.14 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)
- [PyTorch 2.13 release](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/)
- [Triton fused softmax: メモリtrafficを減らす原理](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton block-scaled matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [PyTorch optimizer compile](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html)

ライブラリは今回更新しない。カーネル改善とライブラリ更新の効果を混ぜず、
現環境で動作する改善として検証する。別GPUでの速度は未測定。

## 単体計測

`source scripts/env.sh; python -m scripts.bench_adamw --iters 30 --warmup 5`
実モデルからmeta deviceで260 tensor / 926,807,552 parameterのshapeを取得し、
BF16 parameter/gradient、FP32 momentで直列A-B-B-A実行。初期化/JITはwarmupに含め、
同期を含むwall timeを測定した。入力gradientは合成値で、更新を繰り返す。

| run | backend | optimizer wall ms |
|---|---|---:|
| A1 | eager | 106.009 |
| B1 | triton | 30.441 |
| B2 | triton | 30.335 |
| A2 | eager | 97.799 |

平均101.904 → 30.388 ms、3.35倍、時間70.2%削減。
これはoptimizerのみの結果で、学習全体が3.35倍になるという意味ではない。
raw log: `/tmp/arbor-adamw-micro.log`。

## 正しさの検証

- CUDA: `tests/test_optim.py tests/test_train.py` の90件合格。
  追加した「trainの初期化→ternary forward/backward→融合optimizer→cache再生成」も1件合格。
- CPU全suite: 244件合格、36件skip。Glooの1件はsandboxのsocket禁止で失敗したため、
  同一テストを制限外で再実行し合格（合計245件合格）。
- BF16/FP16/FP32、1/4097/131072要素、30更新、LR変更、WD切替、None gradient、
  非連続tensorの従来実装への切替、checkpoint再開を検証。
  BF16/FP16 parameterは対象テストで完全一致、FP32 parameterはrtol=1e-6/atol=1e-8、
  FP32 momentはrtol=1e-6/atol=1e-10内。
- Ruffおよび `git diff --check` 合格。

## 学習全体の計測条件

現行 `configs/arbor.yaml` の926.8M / context8192 / mean pooling / micro2 / accum32を使用。
外部streamingの速度変動を除くため、既存 `data/random_bytes.bin` (64MiB) を入力にする。
品質・収束のベンチマークではなく、実モデルのforward/backward/clip/optimizer/cache更新を
含む学習ループの計算速度ベンチマーク。validation/checkpoint/sample/probeは無効。
ログ10step間隔、1run60step、最初の20stepを除いて比較する。
CPUテストは初回のコンパイル・tuning中に終了し、集計対象stepとは重ならない。

再現用config: `/tmp/arbor-adamw-{eager,triton}.yaml`、
実行: `python -m src.train.train --config CONFIG --benchmark-steps 60`。
両configは `optim.fp32_backend` のみ異なる。初期化修正前のbaselineも別に保存し、
packed経路の復旧とoptimizer融合の効果を区別する。

## 学習全体の結果とconfig採用

| run | AdamW | bytes/s | step ms | optimizer+cache ms |
|---|---|---:|---:|---:|
| A1 | eager | 170193 | 3069.8 | 191.2 |
| B1 | triton | 177765 | 2939.2 | 119.2 |
| B2 | triton | 179810 | 2904.6 | 113.1 |
| A2 | eager | 173993 | 2997.1 | 190.5 |

A平均3033.484 ms → B平均2921.898 ms (`step time -3.68%`, `speedup 1.0382x`)。
ログのthroughput平均172,093 → 178,787 bytes/s (`+3.89%`)。
optimizer区間（AdamW + cache再生成 + scheduler + zero_grad）は190.885 → 116.171 ms、
39.1%短縮。実行順でAのstep平均は2.4%、Bは1.2%程度変動したが、Bの両runともAの
両runより速かった。残るforward/backwardの揺れも含む実測であり、厳密な信頼区間ではない。

A同士・B同士は集計対象のlossが各log pointで一致。A/B間の最大loss差は0.000062。
短い合成入力での検証なので、長期の実データ収束や他GPUの高速化率を保証する結果ではない。
peak allocatedは全run14.14GiB、peak reservedはA14.58 / B14.48GiB。

初期化不具合を残した変更前runは3034.897 ms/stepだった。この合成入力では
packed経路の復旧だけでは平均速度はほぼ変わらず、最終的な短縮は主にAdamW融合による。
不具合修正前はFP8 backwardも無効だったため、復旧前後を同じ数値演算の比較とは扱わない。

`configs/arbor.yaml` に `optim.fp32_backend: auto` を採用。
model・batch・precision・packed kernelの選択方法・学習データ設定は変更しない。
従来AdamWへの切替は `optim.fp32_backend: eager`。

Metrics:

- a1: `logs/benchmark_arbor-adamw-eager_20260912-170900_78341.jsonl`
- b1: `logs/benchmark_arbor-adamw-triton_20260912-172216_80623.jsonl`
- b2: `logs/benchmark_arbor-adamw-triton_20260912-172654_81057.jsonl`
- a2: `logs/benchmark_arbor-adamw-eager_20260912-173124_81354.jsonl`
- broken: `logs/benchmark_arbor-adamw-eager_20260912-165445_75894.jsonl`

## Nsight Systemsによる実学習の確認

Nsight Systems 2025.3.2、CUDA/NVTX trace、CUDA Graph node trace。
同じconfigで25更新待って26〜27更新目の2更新をcaptureした。
`arbor.optimizer` のNVTX区間内で発行したCUDA APIのcorrelation IDからGPU kernelを
対応付けて集計。Runtime/Driver APIを含め、correlation IDの重複を除く。
以下は1更新あたりで、cache再生成・zero_gradなども含む区間全体の値。

| 指標 | eager | triton |
|---|---:|---:|
| optimizer区間GPU kernel起動数 | 6,998 | 3,878 |
| optimizer区間GPU kernel時間合計 | 121.253 ms | 74.282 ms |
| 融合AdamW kernel起動数 | 0 | 260 |
| 融合AdamW kernel時間合計 | — | 29.841 ms |

起動数44.6%減、GPU kernel時間38.7%減を確認。
差分は3,120回/更新 = 260 parameter × 12回で、AdamWの13演算を1 kernelへ
融合した実装と一致する。残る3,618回はcache生成・grad zeroなど。
中間tensorのメモリ往復削減は実装構造に基づく説明で、DRAM転送量そのものの
ハードウェアcounterは今回測っていない。

Nsight有効時のCPU側optimizer区間は365→295 msで、通常測定の191→116 msより
大幅に長い。trace負荷があるため、このwall timeを学習速度の採否には使わない。

再現コマンド（CONFIGをeager/triton、NAMEを対応する名前に置換）:

```bash
scripts/profile_training_nsys.sh --config CONFIG --wait 25 --active 2 \
  --output /tmp/arbor-adamw-nsys-NAME -- --benchmark-steps 27
```

成果物:

- `/tmp/arbor-adamw-nsys-eager.nsys-rep` / `.sqlite`
- `/tmp/arbor-adamw-nsys-triton.nsys-rep` / `.sqlite`
- `/tmp/arbor-adamw-nsys-{eager,triton}-summary.json`
- 集計script: `/tmp/arbor-adamw-nsys-summary.py`

## 結論

学習全体の実測gainは約3.8%（throughput +3.89%、step time -3.68%）であり、残る時間の
大部分（backward主体で約70%）に、これ以上大きな汎用gainが入る確かな根拠はない。
RTX 4090向けのtile/batch追い込みは行わず、ここで打ち切る。他GPUでの高速化率と
長期実データ収束は未検証のまま残る。

AdamW融合後のportability硬化として、raw Triton kernel起動をparameter deviceで囲む
device guardを追加した（`src/train/adamw_triton.py`）。guard由来の最終sourceでの
オーバーヘッドは再ベンチマークしていない。

guard追加後の最終コードでCUDA有効環境から `tests/test_optim.py` と
`tests/test_adamw_triton.py` を実行し、40 passed / 1 skippedを確認した。
skipは2 GPUを必要とする実機テスト（手元は1 GPU）。変更対象全体のRuffと
`git diff --check` も合格した。
