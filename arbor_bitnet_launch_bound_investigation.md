# Arbor BitNet packed ternary: launch-bound 調査と CUDA Graphs 対応

- 作成日: 2026-09-11
- Codex 再検証・更新: 2026-09-12
- 対象ブランチ: `feat/packed-ternary-kmajor-layout`
- GPU: RTX 4090 (sm89), PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8
- 関連文書:
  - [arbor_packed_ternary_runtime_tuning_design.md](arbor_packed_ternary_runtime_tuning_design.md)
  - [arbor_packed_ternary_runtime_tuning_findings.md](arbor_packed_ternary_runtime_tuning_findings.md)
- 進め方の原則: RTX 4090 固有チューニングはしない。仕組み→一般論としての期待値
  →実測との差→合わないなら何がおかしいか、の順で調べる。

## 0. 現在の状態 (再検証後)

> 2026-09-12追記: 後続のHEAD `470bd3e` を実行すると、train.pyのimport先の誤りを
> `except Exception` が握りつぶし、設定したpacked演算・重みcache初期化がスキップされる
> 回帰を確認した。下記は過去の測定記録であり、現コードで同経路が動く保証ではない。
> 今回の修正と再測定は [学習融合の検証記録](arbor_training_fusion_20260912.md) を参照。

### コミット済み

- `6fcdc83 feat: add packed ternary runtime tuning diagnostics`
  - Codex が残していた runtime autotune / raw・custom 実行経路 / テスト / 設計文書。
  - 一時的だった config 変更 (`run_name`, `checkpoint.dir` の `_legacy` 付与、
    `patch_pooling: legacy`) はコミットに含めず元へ戻した。
  - 検証: CPU 218 passed / 23 skipped、CUDA 100 passed、Ruff・`git diff --check` 合格。
- `9a305c8 fix: create Nsight target parent directory`
  - `scripts/profile_training_nsys.sh` の初回実行バグ (コピー先の親ディレクトリ
    未作成) を修正。これが直前の作業停止点だった。

### 再開後に完成・再検証した変更

- `a35e81b feat: enable CUDA graphs for packed ternary training`
  - `src/train/train.py`: CUDA Graphs + gradient accumulation を成立させる修正
    (下記 5 章)。CUDA Graphs と `grad_accum_steps>1` の一律禁止を解除。
  - `--benchmark-steps N` を完成。benchmark時はcheckpoint/validation/sampling/probeを
    実行せず、終了時にCUDA peak memoryを表示する。
  - `tests/test_train.py`: 固定 grad buffer のテスト、禁止テストを許可テストへ置換。
  - `README.md`, `configs/arbor.yaml`: compile modeと検証済みdefaultの説明を更新。
  - `configs/arbor.yaml` の既定を、後述のA-B-B-Aに合格した
    `kmajor_single_dot + custom_op + auto + reduce-overhead`へ変更。
  - CPU全suite `223 passed / 24 skipped`、CUDA対象suite `92 passed`、Ruff、
    `git diff --check`に合格。

## 1. 結論 (要約)

1. 新 packed kernel (`kmajor_single_dot`) は実学習のpacked GPU時間を約60%短縮する。
   Fuguの元reportは消失したが、Codexが同条件を再profileして独立再現した。
2. 非Graphではkernelを速くしてもwall timeは1.6%しか縮まず、GPU idleは
   0.254→0.945 s/step、100µs超gapは約100→932/stepへ増えた。kernel高速化後に
   CPU launchが律速になるという診断を直接再現した。
3. `custom_op + auto + CUDA Graphs` は現行mean-pooling構成のA-B-B-Aで、
   legacy/default比 `bytes/s +18.98%`, step time `-16.09%`。run間差も約0.2%以下だった。
4. CUDA Graphs + gradient accumulation のクラッシュは、graph外の固定grad bufferと
   step境界通知で解消した。小規模compiled testと926.8M modelのaccum=32で再確認した。
5. 同じsteady intervalで `1f42ae7` は64.3k bytes/s、現行defaultは186.0k bytes/sだった。
   「約320k」は別の21.0M parameter小型ベンチの値で、現行1B級モデルの過去値ではない。

## 2. 仕組みと期待値 (原理)

- packed 2bit ternary decode + INT8 `tl.dot` は、weight のメモリ traffic を大きく
  減らす。memory-bound 領域では forward/dX が速くなる**はず**であり、これは
  アーキ非依存の一般論。
- 期待: 単体で速いなら、実モデルの forward/dX 合計でも非劣化〜改善する。
- 非Graph経路の現実: 単体でもフルでも packed kernel 時間は縮む (期待通り) が、
  step wallは縮まない (期待と不一致)。この不一致の原因を特定するのが本調査。

## 3. Nsight による実測 (Fugu集計、2 optimizer steps, wait=25/active=2)

> 再検証上の制約: 下表の元reportは `/tmp` にのみ置かれ、Codex再開時には存在しなかった。
> コード・commit・実行条件との整合は確認したが、数値は再集計できていない。

### 3.1 packed kernel 単体 (legacy vs raw_plan, 非 CUDA Graph)

| 指標 | legacy | raw_plan (`kmajor_single_dot`) |
|---|---:|---:|
| packed kernel 総時間 | 2.941 s | **1.149 s** |
| packed kernel 中央値 | 218 µs | **75 µs** |
| packed kernel 呼び出し | 11,776 | 11,776 |

→ kernel 選択の考え方は正しい。実 call mix でも約 61% 短縮。

### 3.2 GPU idle (step 区間の GPU 空き時間)

| 構成 | idle/step | 100µs 超 gap 数 |
|---|---:|---:|
| legacy | 0.28–0.38 s | 94–312 |
| raw_plan | 1.19–1.33 s | 1263–1625 |

→ kernel を速くしたことで GPU が待つ時間が増えた = launch-bound 化。

### 3.3 実行経路別の wall time (Nsight, ms/step)

| 構成 | step | fwd | bwd | opt |
|---|---:|---:|---:|---:|
| legacy + default compile | 3508 | 1101 | 1787 | 232 |
| raw_plan + default compile | 3557 | 1151 | 1758 | 266 |
| legacy + CUDA Graphs | **4111** | 224 | 2892 | 489 |
| raw_plan + CUDA Graphs | **3057** | 198 | 2089 | 380 |
| custom_op+auto + CUDA Graphs | **3093** | 235 | 2034 | 409 |

観察:
- 非 Graph では kernel が速い raw_plan でも step は縮まない (launch-bound)。
- CUDA Graphs は legacy には逆効果 (+17%)、raw_plan には有効。
- raw_plan+Graph は legacy/default 比で約 12.9% 短縮。
- `custom_op+auto`+Graph は raw_plan+Graph とほぼ同等 (差 1.2%)。

> 注意: 各構成 1 回計測の Nsight であり、run 間分散は未評価。default 変更前に
> 交互順の長めの A/B が必要 (7 章)。

### 3.4 現行configのA-B-B-A再測 (120 optimizer steps/run)

2026-09-12、現行 `patch_pooling: mean`、同じseed/data/model、micro batch 2、
gradient accumulation 32で実行した。各runのstep 1--20を除き、step 21--120の
100 stepsを集計した。

| run | 構成 | 開始温度 | bytes/s | step ms | fwd ms | bwd ms | opt ms |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | legacy/default | 40°C | 155,915 | 3354.0 | 1050.8 | 2093.4 | 173.7 |
| B1 | custom+auto/Graph | 50°C | 185,569 | 2820.8 | 602.2 | 1994.8 | 184.6 |
| B2 | custom+auto/Graph | 47°C | 185,817 | 2808.9 | 609.6 | 1974.5 | 185.8 |
| A2 | legacy/default | 47°C | 156,235 | 3355.4 | 1048.0 | 2092.3 | 179.8 |

- A平均: 156,075 bytes/s, 3354.7 ms/step。
- B平均: 185,693 bytes/s, 2814.8 ms/step。
- B/A: `bytes/s +18.98%`, `step_ms -16.09%`。
- 同一構成のrun間差: A throughput 0.21%、B 0.13%。
- loss/EMAは全runの全log pointで一致。100-step平均loss `3.737610`、EMA `4.496888`。
- peak allocated: A 10.94 GiB / B 10.92 GiB。peak reserved: A 11.14 GiB /
  B 14.45 GiB。Graph poolにより予約量は3.31 GiB増えるが、23 GiB GPU内で成立する。

旧findingsの約9 s/step・約58k bytes/sの比較は、step 20 lossが `5.81`で現行meanの
`5.72`と一致せず、中断メモにも一時的な `patch_pooling: legacy` 変更が記録されている。
したがって別モデル条件の測定であり、現行defaultの採否には使わない。

### 3.5 CodexによるNsight再取得・因果再検証

2026-09-12に現行mean-pooling条件で、wait 25 / active 2 optimizer stepsを
Nsight Systems 2025.3で再取得した。GPU busyはstep内のkernel/memcpy/memset区間を
unionし、Graph構成ではgraph trace区間も加えた。idleはstep wallとの差、gapはunion後の
空白区間から算出した。

| 構成 | wall/step | GPU busy/step | idle/step | >100µs gap/step | launch API calls |
|---|---:|---:|---:|---:|---:|
| legacy/default | 3.273 s | 3.019 s | 0.254 s | 99.5 | 133,432 |
| raw_plan/default | 3.220 s | 2.275 s | 0.945 s | 932.0 | 133,800 |
| custom+auto/Graph | 2.717 s | 2.119 s | 0.598 s | 72.5 | 38,942 |

packed kernelはlegacy/raw_planとも11,776 calls。総時間は2.803→1.130 s (`-59.7%`)、
中央値は206.0→70.3 µs (`-65.9%`)。それでもraw_planのwall改善は1.6%に留まり、
idleは0.691 s/step増え、長いgapは約9.4倍になった。よってlaunch-bound診断は正しい。
Graph構成はlaunch API callを70.8%減らし、Nsight wallもlegacy比17.0%短縮した。
これは非Nsight A-B-B-Aのstep time `-16.09%`と整合する。

Graph reportではchild kernelが `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` にaggregateされるため、
Graph内packed kernelの名前別時間はこの取得方式では集計しない。

### 3.6 `1f42ae7` との同条件比較

commit `1f42ae7a8f3a7ad26dfe1217ac825c0712eff68d` を別worktreeへ展開し、
現行と同じRTX 4090・データ・seed・926.8M model・`patch_pooling: mean`・
micro batch 2・gradient accumulation 32で120 optimizer stepsを実行した。
checkpoint保存だけを無効化し、計算内容は変えていない。step 60/80/100/120の
log interval平均を比較する。

| revision / 構成 | bytes/s | step ms | fwd ms | bwd ms | opt ms |
|---|---:|---:|---:|---:|---:|
| `1f42ae7` / INT8 | 64,340 | 8129.3 | 2685.4 | 5053.2 | 129.0 |
| 現行 / legacy packed | 155,406 | 3368.4 | 1055.7 | 2100.3 | 177.1 |
| 現行 / custom+auto/Graph | **186,033** | **2812.9** | **606.4** | **1982.6** | 184.8 |

現行defaultは `1f42ae7` の約2.89倍のthroughputで、step timeは65.4%短い。
現行のlegacy packedだけでも約2.42倍であり、`1f42ae7` が速かったという事実は
再現しなかった。旧版の通常サイズINT8 GEMMはPyTorchの `torch._int_mm` を使っており、
RTX 4090専用tileを固定した実装ではない。configには4090実測で選んだFlexAttentionの
注記があるが、その設定は現行にも残っているため、今回の速度差の説明にはならない。

### 3.7 「約320k bytes/s」の出所

過去の実行記録に300k超の値は実在した。ただし2026-09-07の小型合成ベンチで、
現行フルモデルの過去baselineではない。

| 項目 | 小型ベンチ | 現行フルモデル |
|---|---:|---:|
| parameter数 | 約21.0M | 926.8M |
| global model | hidden 512 / 2 layers | hidden 2048 / 20 layers |
| local model | hidden 768 / encoder 1 + decoder 1 | hidden 768 / encoder 1 + decoder 2 |
| 入力 | synthetic random batch | production data mixture |

小型ベンチの実測は、同じeffective batch 16で `micro=4/accum=4` が
379--385k bytes/s、`micro=8/accum=2` が332--333k bytes/s、
`micro=16/accum=1` が274--299k bytes/sだった。「320k前後」という記憶は正しいが、
parameter数が約44分の1の別ベンチなので、926.8M modelの性能比較には使えない。

この差が単なるmicro-batchの違いかも確認するため、フルモデルのeffective batchを64に
保ったまま `micro=4/accum=16` で120 stepsを追加実行した。step 40--100は
181.7--187.4k bytes/sで、既定の `micro=2/accum=32` の平均185.7kに対して改善しない。
さらにstep 101--120は50.7k bytes/s、10.33 s/stepへ急落した。実行中に観測したGPU
memory使用量は22.1 / 23.0 GiBで、100-step区間全体の実効throughputは約120.5k bytes/s。
容量限界に近く不安定なため、このbatch形状は既定値へ採用しない。

## 4. 一般化の方向 (4090 固有にしない)

`raw_plan` は事前 cache 必須で利用者 default に不向き。代わりに:

```
custom_op + auto
  → 実行 GPU 上で候補を実測
  → fingerprint 付き cache
  → 選択済み kernel を CUDA Graph へ capture
  → steady replay では Python resolver を通らない
```

これは GPU ごとに実測選択するため、4090 固有 tile 固定にならない。現行configの
長時間A/Bにも合格したため、production defaultへ採用する。

## 5. CUDA Graphs + gradient accumulation クラッシュの原因と修正

### 症状

旧コードは CUDA Graph compile mode (`reduce-overhead`/`max-autotune`) と
`grad_accum_steps>1` を config 段階で一律禁止していた (PyTorch 2.5 時代の実測)。
禁止を解除して切り分けると:

- micro 0 forward/backward: 成功
- micro 1 forward: 成功
- micro 1 **backward で** `accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run`。

### 原因

最初の backward が `parameter.grad` を CUDA Graph の private pool 上に作り、次の
forward replay が同じ storage を再利用するため、累積途中の grad が壊れる。
activation ではなく **gradient accumulation の grad 生存契約**と Graph Trees の衝突。

### 修正 (`src/train/train.py`, `a35e81b`)

1. `prepare_cudagraph_gradient_buffers()` で `parameter.grad` を graph 外に事前確保。
2. model invocation ごとに `torch.compiler.cudagraph_mark_step_begin()`。
3. optimizer 後も `zero_grad(set_to_none=False)` で buffer アドレスを維持。
4. `uses_cudagraph_compile()` ヘルパで判定を一元化。

### 再確認済み (2026-09-12)

- CPU全suite: 223 passed / 24 skipped。
- CUDA `tests/test_train.py + tests/test_bitlinear.py`: 92 passed。
- 小規模 `torch.compile(mode="reduce-overhead")` で3 micro-step x 3 optimizer step完走。
- RTX 4090、926.8M model、accum=32が1 optimizer step完走。
- 固定 grad buffer: 260 tensors / 1.73 GiB。
- A-B-B-Aの全log pointでloss/EMA一致。

## 6. 変更・生成物の場所

- 修正コード: `src/train/train.py` (`a35e81b`)
- テスト: `tests/test_train.py` (`a35e81b`)
- 診断用 config (再現用、リポジトリ外 `/tmp`):
  - `/tmp/arbor-nsys-raw-plan.yaml` (kmajor_single_dot + raw_plan + auto)
  - `/tmp/arbor-cudagraph-raw-plan.yaml` (上記 + reduce-overhead)
  - `/tmp/arbor-cudagraph-legacy.yaml` (legacy + reduce-overhead)
  - `/tmp/arbor-cudagraph-custom-auto.yaml` (custom_op + auto + reduce-overhead)
- Nsight report (`/tmp`): `arbor-nsys-legacy`, `arbor-nsys-raw-plan`,
  `arbor-nsys-cudagraph-legacy`, `arbor-nsys-cudagraph-raw-plan`,
  `arbor-nsys-cudagraph-custom-auto` の `.nsys-rep` / `.sqlite` / `-stats_*.csv`
  (Fugu終了時には存在したがCodex再開時には消失)。
- A-B-B-A log/metrics: `/tmp/arbor-abba-{a1,b1,b2,a2}.log` および
  `/tmp/arbor-abba-{a1,b1,b2,a2}/metrics.jsonl`。
- 再取得Nsight: `/tmp/arbor-verify-nsys-{legacy,raw-plan,default}.nsys-rep/.sqlite`、
  summary CSVは `/tmp/arbor-verify-{legacy,raw-plan,default}_*.csv`。
- autotune cache: `~/.cache/arbor/packed_ternary_autotune.json` (16 entries)。

## 7. 次にやること

1. [完了] `--benchmark-steps N`、A-B-B-A、correctness、VRAM peakを確認。
2. [完了] 合格した `custom_op + auto + CUDA Graphs` をproduction defaultへ変更。
3. [完了] 新default、legacy、raw_planのsteady timelineを再取得し、launch-boundの
   因果を独立再現。
4. `raw_plan + Graph` はproduction必須ではない。custom opとの差を再確認する場合のみ、
   同じmean-pooling条件でA/Bする。
5. 新GPU/driver更新時はautotune cacheを再検証し、reserved memoryが搭載量に収まることを
   確認する。VRAM不足時のrollbackはlegacy/defaultを使う。

## 8. 低優先の別トラック: patch pooling

- `patch_pooling: mean` は encode 時に中央部分の情報が欠落気味。
- `legacy` にすると過剰圧縮が緩和されたという利用時観測 (未体系比較)。
- BitNet 性能調査とは混ぜず、品質指標・速度/memory trade-off を分離して追試する。
- 現在の標準 config は指示により `mean` のまま。未解決の低優先課題。

## 9. Gated-FFN fusion PoC (2026-09-12)

RTX 4090上で、`M×K×I = 1024/2048/4096 × 2048 × 5632`、BF16 output、
tile `64×64×64`、warps=4、stages=2、warmup=10/iters=100を単一GPUで直列計測した。
従来の `_packed_linear()` reference（runtime tuning/cacheを含み得る）とは別に、
同じprequantized A8 input、K-major packed weight、scale、固定tileを使う
`_packed_linear_execute()` ×2 + ReLU²/multiply の raw-vs-raw 比較を追加した。

| M | production-path speedup | fixed raw-vs-raw speedup |
|---:|---:|---:|
| 1024 | 2.210× | 1.080× |
| 2048 | 1.108× | 1.356× |
| 4096 | 3.032× | 1.487× |

M=1024 raw-vs-rawは再測でも1.189×（fixed ref 0.2836 ms、fused 0.2386 ms）であり、
5% kernel-level閾値は満たす。一方でfused対fixed rawのM=1024数値差は最大相対
2.027%、平均相対0.165%（最大absolute 8192）だった。これはBF16/accumulation順序の
影響と推測できるが、semantic proofではない。またPoCはforward-only raw kernelであり、
学習に必要なinput/weightのbackwardとSTEをまだ持たない。

したがって**現時点では学習経路へ統合しない**。次段階は既存BitNet pathに対する
許容誤差の定義、forward/backward/optimizer-step equivalence test、そしてend-to-end
training impactの測定である。ベンチはkernel ceilingを示すものでproduction speedupを
保証しない。
