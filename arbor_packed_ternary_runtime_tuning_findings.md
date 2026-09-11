# Arbor Packed Ternary Runtime Tuning: 調査結果と修正方針

- 作成日: 2026-09-11
- 対象ブランチ: `feat/packed-ternary-kmajor-layout`
- legacy 基準: `1140a479efe3e341e213d77147d8bbd1f0d75a27`
- 関連設計: [arbor_packed_ternary_runtime_tuning_design.md](arbor_packed_ternary_runtime_tuning_design.md)

## 結論

runtime autotune のデータモデル、候補生成、fingerprint 付き cache、失敗policyは概ね妥当である。加えて、cache済みの `TuneKey -> PackedLaunchConfig` をcompile前に読み、compiled hot pathから`custom_op`、resolver、I/O、計測を外す`raw_plan`経路まで実装した。機能とcorrectnessは成立している。

一方、**性能改善としては未採用**である。旧commitのAに対し現行`custom_op + auto`のB3は`bytes/s -7.14%`だった。さらに同一worktreeでA2/B1/B2を120 stepずつ実行すると、A2と旧A自体に`-5.82%`のrun間差があり、A2比のB1は`+2.17%`、B2は`+1.57%`となった。現状の差はrun間変動を超える再現性のある改善ではない。

`custom_op`境界はmicrobenchmarkで測定可能なoverheadを持つが、実学習B1は同一backend/tileのraw A2より逆に速かった。したがって、compile visibilityや単発dispatch差だけを理由にraw化をproduction採用しない。また、shape別単体計測で`kmajor_single_dot + raw_plan`はforward/dXともlegacyより大幅に速いが、実学習B2では同等の改善を再現しない。次の主対象は、単体kernel選択ではなくmodel全体のcompiled scheduling、launch gap、実行call mixである。

短期方針は、既存tuner/cache、`legacy_raw`、`legacy_custom_op`、`raw_plan`を診断経路として維持し、同一条件の交互順A/Bを取ることである。合格まではproduction defaultを`dot_current + legacy_raw + tuning=off`から切り替えない。`triton_op`や新layoutを追加する前に、現在の4経路で原因を独立に測る。

## 1. 今わかっている事実

### 1.1 実装状況

現在の変更には以下が含まれる。

- `src/model/bitlinear_tuning.py` に `PackedLaunchConfig`、`TuneKey`、GPU/software/kernel fingerprint、候補生成、CUDA Event 測定、process cache、JSON 永続 cache がある。
- tuning mode は `auto`、`fixed`、`off` を持つ。
- autotune 候補は shape 固有 winner table ではなく構造的な tile 集合から生成される。既知の基準と同一コード上で A/B できるよう、旧 shape heuristic は `_legacy_packed_launch_spec()` として明示的な rollback 経路にのみ復元した。
- training 設定と `scripts/bench_bitlinear_kernels.py` は同じ tuning policy を使用する。
- `src/model/bitlinear.py` の通常 packed 経路は `arbor::packed_ternary_linear` という `torch.library.custom_op` を呼び、その実装内で config 解決と Triton kernel launch を行う。
- `_packed_linear_execute()` は primitive な固定 launch spec を受け、tuning/cache/logging を含まない raw Triton data plane として分離済みである。
- `speed.bitlinear_ternary_execution: raw` と `--execution-path raw` で、この raw 経路を `TernaryBitLinearSTE` から使用できる。現段階では誤った動的利用を避けるため `tuning=fixed` を必須としている。
- `raw_plan` は起動時に永続cacheをGPU/software/kernel fingerprintで検証し、選択済みtileをprimitive tupleへ変換する。compiled実行でのplan missは、autotuneやfallbackをせず必要keyを表示して停止する。
- `legacy_custom_op` は`dot_current`と旧shape heuristicを保ったまま、実行境界だけを`custom_op`にするB1用経路である。
- `legacy_raw` は旧 `dot_current` shape heuristic を raw Triton で再現する基準経路である。実学習受け入れ完了までは `dot_current + legacy_raw + tuning=off` を config と省略時の既定値にした。
- benchmark の `--boundary-ab` は同一 process/input/backend/layout/tile で raw と現行 custom op を比較する。`--compile-boundary-ab` は両方を `torch.compile(fullgraph=True)` 下でも比較する。
- `scripts/env.sh` が選ぶ現在の環境は Python `3.12.3`、PyTorch `2.11.0+cu128`、CUDA build `12.8`、Triton `3.6.0` である。

### 1.2 検証済みの correctness

直前の実装確認では以下が完了している。

- CPU test suite: `212 passed, 19 skipped`
- CUDA の `tests/test_bitlinear.py`: `40 passed`
- tuning/config 関連: `48 passed, 2 skipped`
- `torch.compile` を含む packed ternary の CUDA test が通過
- 代表 shape の backend 間比較で `max_abs_diff=0`
- Ruff と `git diff --check` が通過

これは数値意味と基本的な compile 可否が維持されていることを示す。一方、`torch.compile` test が通ることは、compile 後の実行効率が同等であることを保証しない。

作業再開後の最終確認では、GPUを隠した全suiteが`218 passed, 23 skipped`、CUDAの固定4経路が`4 passed`である。さらに実際にautotuneしてcacheを生成し、同cacheから`raw_plan`を読み、`torch.compile(fullgraph=True)`でforward/dX/dWを実行するtestが`1 passed`である。Ruffと`git diff --check`も通過した。従来のCUDA testは既定`legacy_raw`のままでautotuneを実行していなかったため、このライフサイクルtestへ修正した。

### 1.3 単体 benchmark と autotune

既存設計時の RTX 4090 測定では、代表 shape において `kmajor_single_dot` と適切な tile が `dot_current` より速かった。現在の tuner も代表 shape で有効な config を選び、独立した再測定で候補集合内の上位 config であることを確認済みである。

ただし tuner が記録した CUDA Event の値と、同期条件の異なる独立 benchmark の値には約 2 倍の絶対値差が見られた例がある。winner が明白に壊れている証拠ではないが、測定 protocol が十分安定しているとはまだ言えない。

最初の全 shape tuning を含む dry-run は約 217 秒を要した。永続 cache 作成後の実学習 A/B では、実際に使われた 14 個の tune key はすべて cache hit であり、定常区間に再探索は入っていない。現在の永続 cache 自体には 16 entry がある。

shape profilerでは1 micro-batchあたりforward/dXがそれぞれ92 call、8 unique shapeだった。同call数をgradient accumulation 32回に展開した単体中央値の概算で、`kmajor_single_dot + raw_plan`はforward `666.4 ms`、dX `683.7 ms`、legacyはforward `1933.9 ms`、dX `2025.5 ms`だった。これは選択kernelが単体で速いことは示すが、後述の実学習結果を予測しない。また、profilerのdW counterが旧関数をhookして0件になる不備と、benchmarkが実経路と異なる再構成を含む不備は、実際の`_ternary_wgrad_from_a8()`をcount/measureするよう修正した。

### 1.4 実学習 A/B

比較条件は同じ RTX 4090、model/config、dataset、seed、micro batch `2`、gradient accumulation `32`、context `8192`、static patch `16`、FP8 dW、120 optimizer stepsである。step 1-20をwarmupとし、10 stepごとのlog 10区間、すなわちstep 21-120の100 stepsをsteady集計した。

旧commitのAと現行`custom_op + auto`のB3比較は次の通りである。

| 指標 | A: 旧commit legacy | B3: current custom auto | B3 / A |
|---|---:|---:|---:|
| `bytes/s` | 61,793.9 | 57,383.1 | **-7.14%** |
| `step_ms` | 8,481.179 | 9,132.946 | **+7.68%** |
| `fwd_ms` | 2,326.403 | 2,882.212 | **+23.89%** |
| `bwd_ms` | 5,228.316 | 5,250.214 | +0.42% |

その後、現行worktree上で実行境界とbackend/configを分けた。

| Case | backend/config | 実行経路 | `bytes/s` | `step_ms` | `fwd_ms` | `bwd_ms` | A2比bytes/s |
|---|---|---|---:|---:|---:|---:|---:|
| A2 | `dot_current` + legacy heuristic | `legacy_raw` | 58,198.2 | 9,007.913 | 2,807.492 | 5,349.402 | baseline |
| B1 | A2と同一 | `legacy_custom_op` | 59,461.9 | 8,808.950 | 2,632.358 | 5,389.854 | **+2.17%** |
| B2 | `kmajor_single_dot` + cached winner | `raw_plan` | 59,110.2 | 8,866.072 | 3,046.306 | 4,985.336 | **+1.57%** |
| B3 | B2と同一 | `custom_op` | 57,383.1 | 9,132.946 | 2,882.212 | 5,250.214 | -1.40% |

全5 runの全log pointでlossとEMAは一致した。steady平均はloss `3.7675656`、EMA `4.5460965`、step 120はloss `3.722799`、EMA `4.017266`である。

注意すべき点は、同じlegacy rawの旧Aと今回A2に`bytes/s -5.82%`、`step_ms +6.21%`の差があることである。また、B1はA2よりforwardが`6.24%`速く、B2はforwardが`8.51%`遅い一方でbackwardが`6.81%`速い。この大きさと符号の変動から、次が確実に言える。

- 数値的correctnessは維持されている。
- B3は旧Aに対し受け入れ失敗であり、production defaultにできない。
- B1/B2のA2比1--2%改善は、観測済みrun間変動5.8%を下回るため、性能採用の根拠にできない。
- `custom_op`境界が実学習regressionの単独主因という仮説は、B1の結果と整合しない。
- `raw_plan`はresolverとopaque opをhot pathから外す機能目的を満たしたが、実学習での非劣性はまだ証明されていない。

## 2. どこが悪い可能性があるか

以下は可能性順であり、事実と推測を混同しない。

### P0: 単体kernel目的関数とmodel全体実行の不一致

shape/call countを収集し、raw固定planのB2まで実行した。`kmajor_single_dot`はforward/dXの単体合計でlegacyより大幅に速いにもかかわらず、full trainingのB2はA2と実質同等であった。したがって、次に測るべきは単一launchではなく、compiled region全体のCUDA kernel列、CPU launch gap、allocation、synchronization、周辺kernelとの重なりである。単体tunerは候補排除には使えるが、production winnerの最終判定器にはできない。

### P1: `custom_op` と resolver による hot-path overhead

**事実:** 通常経路は `_packed_linear_custom_op()` を通り、その内側に output allocation、tuning lookup、Triton launch がある。PyTorch 公式は、`custom_op` の中身を `torch.compile` が trace しないと明記している。

**測定済み:** `M=64,K=128,N=64`、tile `32,64,32,4,2` では同一 process の eager A/B が raw `0.083--0.142 ms`、current custom op `0.161--0.218 ms` だった。`M=2048,K=2048,N=11264`、tile `128,64,64,4,2` では raw `0.321 ms`、current custom op `0.378 ms` だった。後者の `0.057 ms/call * 320 calls` は約 `18.2 ms/update` である。

**注意:** current custom op 側には固定 config resolver も含まれるため、これは dispatcher だけでなく「現行 opaque boundary + resolver」の合算差である。またゲームと同時実行した測定で分散が大きい。small shape の compiled A/B でも raw に対し current custom op が `1.49x` だったが、raw の p10/median が `0.011/0.249 ms` と不安定であり定量判断には使わない。

**結論:** overheadの存在は確認したが、実学習B1はraw A2より速かった。raw化は原因分離とcompile visibilityのための有用な経路だが、現時点で性能修正そのものとは見なさない。

### P1: 単体 kernel の目的関数と実学習の目的関数が異なる

autotuner は 1 shape、1 kernel launch の中央値を最小化する。一方、実学習は多数の shape、forward/dX、周辺 kernel、memory allocation、cache、stream scheduling の合計で決まる。

単体では速い `kmajor_single_dot` が、実モデルの call mix、layout、cache behavior、周辺 kernel との並びでは利点を失うことがB2で観測された。backend内tile選択というPhase 1の範囲は妥当だが、「backendが既に勝者」という前提は成立していない。

### P1: autotune 測定 protocol の不安定さ

現在の tuner は全候補を compile/warmup した後、CUDA Event で複数 launch を enqueue し、最後に同期して中央値を取る。benchmark CLI は iteration ごとに同期する経路があり、絶対値が一致しない。

候補間の差が大きい場合は winner に影響しないが、近接候補では clock、thermal state、測定順、queueing が選択を変え得る。永続 cache はその一度の winner を長期間固定する。

### P1: output allocation と memory planning

`torch.empty((m, n))` が不透明な custom op 実装内にある。PyTorch allocator 自体が遅いと断定はできないが、compile graph が buffer lifetime と再利用を把握できないことは確認対象である。

### P0: A/B の実行順と thermal/clock noise

同一legacy rawの旧AとA2に5.82%の差があり、境界だけが異なるA2/B1ではmicrobenchmarkと逆符号になった。個別runはGPU温度、clock、Windows/WSL側の利用、compile/cache状態の影響を除外できない。A/BはA-B-B-Aの交互順、開始温度/P-state/clock記録、各case最低2 run、中央値と分散付きに修正する。

### P2: cache winner の陳腐化または fingerprint 不足

cache key は GPU、compute capability、memory、Torch、CUDA、Triton、kernel version、shape、dtype、scale mode を含み、設計としては十分保守的である。ただし driver version、compile flags、power/clock policy は入っていない。過去の thermal/clock 状態で選んだ winner が現在も最適とは限らない。

## 3. 既存実装を活かして修正する方針

### 3.1 原則

`bitlinear_tuning.py` を control plane、Triton kernel launch を data plane として明確に分ける。

- control plane: 候補生成、測定、failure handling、cache、logging
- data plane: frozen config を受け取り、allocation と Triton launch だけを行う traceable な関数
- steady-state の compiled graph 内で Python の cache lookup、file I/O、CUDA Event 測定をしない。
- correctness が通るだけでなく、legacy 比の実学習性能 gate を必須にする。

### 3.2 最初に行う切り分け

同じ seed/config で次を比較し、1 回に 1 要因だけ変える。

| Case | backend/tile | 実行境界 | 目的 |
|---|---|---|---|
| A | `dot_current` + legacy heuristic | legacy raw Triton | 基準 |
| B1 | A と同一 config | current `custom_op` | 境界だけを測る |
| B2 | current selected config | raw Triton | kernel/config だけを測る |
| B3 | current selected config | current `custom_op` | 現状再現 |

B1 が遅ければ境界、B2 が遅ければ backend/config、両方なら複合原因である。各 case は実行順を交互にした最低 2 run を取り、同一 run 内の steady 100 steps で比較する。

現在は4経路の実装、micro A/B、shape/call-weighted profile、実学習A2/B1/B2/B3まで完了した。結果は単一原因を支持せず、run間変動より大きな改善も示していない。次の決定点は、A-B-B-A交互順の再測とsteady-state timeline profileである。

併せて以下を行う。

- `TORCH_LOGS=graph_breaks,output_code` と `torch._dynamo.explain` で graph break、custom op node、生成コードを比較する。
- Nsight Systems の CUDA/NVTX trace で CPU launch gap、GPU idle、kernel call count、allocation/sync を比較する。
- 同一 tile の `_packed_linear` を多数回呼ぶ小規模 test で raw Triton と custom op の dispatch overhead を比較する。
- call shape ごとの回数と総 GPU time を集計し、単一代表 shape ではなく call-weighted cost を作る。

### 3.3 推奨する実装修正

第一案であった、既存tuner/cacheを維持して`custom_op`をhot pathから外す最小修正は`raw_plan`として実装済みである。

1. 使用 shape を preflight で tune し、`TuneKey -> PackedLaunchConfig` を freeze する。
2. compile 対象関数には選択済み `BLOCK_M/N/K`、`num_warps`、`num_stages` を静的値として渡す。
3. raw Triton kernel を直接 launch し、Inductor から実装を見える状態に戻す。
4. raw Triton で subsystem integration が不足する場合だけ、`torch.library.triton_op` と `torch.library.wrap_triton` を使う。
5. runtime missはcompiled hot path内でautotuneせず、現在は明示エラーにする。

現在のconfig/cache API、candidate generator、benchmark CLI、failure policyは再利用した。planのPython辞書lookupはDynamo trace時に解決され、Triton meta parameterは固定値になる。CUDA lifecycle testとfull training B2でこの経路の成立を確認した。ただし、性能非劣性は確認できていない。

代替案として Triton の `@autotune` を kernel に直接付ける方法もある。`torch.compile` は `configs`、`key`、`restore_value`、`reset_to_zero` をサポートし、Triton は `cache_results=True` で disk timing cache を持てる。ただし現在の fingerprint schema、詳細 log、failure policy、独自 pruning をどこまで維持できるかを先に小さく検証する。全面置換より、まず 1 backend/1 shape family の試作が適切である。

### 3.4 当面の production policy

- 修正後 A/B が合格するまでは `kmajor_single_dot + auto` を production default にしない。
- rollback 用の `off` と `fixed` は残す。
- 一時的 default は実測済みの legacy `dot_current` 相当とする。
- cache schema は実行境界または kernel code が変わるたびに version を上げる。
- cache hit でも optional な再検証 command を用意し、selected config が同一 run の measured best の `1.20x` 以内か確認できるようにする。

### 3.5 受け入れ条件

最低条件は以下である。

- 全 correctness、compile、cache tests が従来どおり通る。
- `torch.compile(fullgraph=True)` または graph inspection で、意図しない graph break がない。
- representative shape の selected config が同一 run の measured best の `1.20x` 以内。
- 同一 machine/config の交互順 A/B で、steady 100 steps の `bytes/s` が legacy より悪化しない。
- `fwd_ms`、`bwd_ms`、VRAM peak、loss/EMA を併記し、throughput だけで判断しない。
- cold-start tune 時間は steady throughput と分けて報告する。

## 4. 既存をすべて無視できる場合の方針

greenfield でも、packed ternary kernel を直ちに CUTLASS/CuTe へ全面移植するのが最善とは限らない。現在の toolchain と custom ternary decode を考えると、まず Triton 中心で実行境界を設計し直す方が低リスクである。

### 4.1 二段構成

**Build/preflight phase**

- model shape、micro batch、patch policy から実際の operation signature を列挙する。
- tile だけでなく backend、layout、decode algorithm を同じ探索空間で評価する。
- 単発 latency と call count から、model 全体の call-weighted objective を作る。
- candidate を事前 compile し、hardware/software/kernel fingerprint 付きの tuning artifact を生成する。
- top 1 だけでなく top 2-3 と測定分散を保存し、僅差 winner の再検証を可能にする。

**Execution phase**

- runtime では immutable な tuning artifact を読むだけにする。
- graph ごとに config を定数化し、raw Triton または traceable `triton_op` を実行する。
- file I/O、benchmark、Python callback、同期を steady path に入れない。
- 未知 shape は明示的な再計画か conservative config へ進み、暗黙の backend fallback はしない。

### 4.2 kernel 境界

greenfield では「1 packed GEMM」を API 境界に固定せず、BitLinear の実際の dataflow を境界候補にする。

- A8 row quantization
- packed ternary decode + INT8 dot
- activation/weight scale 適用
- 必要なら bias/epilogue

row reduction と matmul を無理に単一 kernel へ詰めると、N tile ごとに reduction を重複する危険がある。まず 2 個の Triton kernel を 1 個の traceable `triton_op` 内に置き、中間 buffer と launch gap を profile する。その結果が支配的なら persistent kernel、producer/consumer fusion、専用 epilogue を検討する。

forward と dX は shape が違うだけという現設計の原則は維持できる。ただし、将来それぞれに異なる algorithm が勝つなら、operation role を key に足すのではなく、backend candidate の実装差として明示し、実測で選ぶ。

### 4.3 backend の選択

- **第一選択: Triton。** 現在の PyTorch/Triton stack と直接統合でき、packed decode を柔軟に書ける。
- **第二選択: CUTLASS/CuTe DSL。** 大規模な kernel rewrite と toolchain 更新を許容し、Triton で到達できない tensor-core pipeline、persistent scheduling、epilogue fusion を狙う場合に評価する。
- **CUDA Graphs:** kernel/API overhead が支配的と確認された後の段階で評価する。現在は `grad_accum_steps>1` と cudagraph mode の既知の correctness 問題があり、第一修正にはできない。

CUTLASS DSL 4.4 の公式要件は CUDA Toolkit `12.9` または `13.3` と対応 driver であり、現在の PyTorch CUDA build `12.8` からは環境変更になる。また一般的な GEMM profiler/autotuner が、そのまま 2-bit packed ternary decode を MMA 前に行う Arbor kernel の代替になるわけではない。採用判断には小規模 prototype が必要である。

## 5. 周辺技術調査

### 5.1 既存を修正する場合

| 技術 | 公式仕様上の性質 | Arbor への適合 | 判断 |
|---|---|---|---|
| raw Triton + `torch.compile` | user-defined Triton kernel は compile graph に trace 可能 | legacy に近い。既存 kernel/tuner を再利用しやすい | **短期の第一候補** |
| `torch.library.triton_op` + `wrap_triton` | 1 個以上の Triton kernel を trace 可能な op として統合 | tensor subclass 等との composability が必要なら有力 | **第二候補** |
| `torch.library.custom_op` | compile/export は実装内部を trace しない | correctness 境界にはなるが hot path 最適化を妨げる疑い | 現形のまま採用しない |
| `torch.compiler.disable` / graph break | 問題領域を eager 実行へ逃がす | 原因切り分けには使える | 恒久性能修正にしない |
| `torch.compiler.allow_in_graph` | Dynamo safety check を迂回する black-box API | soundness risk があり、今回の一般解ではない | 採用しない |
| Triton `@autotune` | config/key/pruning/benchmark/disk cache を標準提供 | 独自 tuner の一部を削減できる | 小規模 prototype 後に判断 |
| Nsight Systems | CUDA API、kernel、memory、CPU/GPU timeline、GPU gap を確認可能 | opaque boundary 仮説の検証に適する | 次回 A/B で使用 |

PyTorch は、単純さを優先するなら `torch.library` wrapper のない Triton kernel を推奨し、wrapper が必要な場合は Triton 実装に `custom_op` ではなく `triton_op` を使うよう案内している。`custom_op` は `torch.compile` から不透明、`triton_op` は内部を trace して最適化できる、という差は今回の設計判断に直接関係する。

### 5.2 既存を無視する場合

| 技術 | 強み | 制約 | greenfield での位置づけ |
|---|---|---|---|
| Triton + offline/preflight tuner | Python で custom decode、候補生成、compile 統合を一体化しやすい | 最良 pipeline は手動設計が必要 | **基準実装** |
| Triton `@autotune(cache_results=True)` | 標準の key/config 探索と disk cache | Arbor 固有 fingerprint、log、failure policy の差を埋める必要 | tuner 簡素化候補 |
| CUTLASS/CuTe DSL | hardware hierarchy、GEMM pipeline、JIT/autotuning を細かく制御 | toolchain 更新、学習コスト、ternary decode の独自実装が必要 | Triton の限界確認後 |
| native C++/CUDA extension | 最大の制御、AOT 配布が可能 | build/ABI/packaging 負担。Inductor からは通常 custom op 境界 | 大きな fused op に価値がある場合 |
| CUDA Graphs | workflow を事前定義し CPU launch cost を削減 | static address/control flow、capture safety、現行 grad accumulation 問題 | correctness 解消後の最終段階 |

CUTLASS/CuTe 側にも candidate の列挙、compile、benchmark、cache という標準的 autotuning pattern がある。ただし、framework integration と model 全体の objective は別途設計が必要であり、library を変えるだけでは今回の問題は解決しない。

## 6. 元設計書の妥当性評価

### 妥当な指針

- GPU SKU や shape 固有 winner table を production policy にしない。
- numerical semantics と hardware tuning policy を分離する。
- GPU/software/kernel fingerprint で cache を分離する。
- first-use だけ測定し、毎 step autotune しない。
- benchmark と production の candidate source を共有する。
- microbenchmark だけで完了にせず、実学習 steady 100 steps で A/B する。
- `selected <= 1.20 * measured_best` の相対評価を使う。

### 修正した指針と残課題

1. [反映済み] Section 17のcontrol planeとdata planeを分け、`raw_plan`でresolveをcompile前に移した。
2. [一部反映] `torch.compile` correctnessに加え、実学習throughput gateを追加した。steady-stateのgraph/timeline診断は残っている。
3. [反映済み] Section 39に`bytes/s`非劣性と交互順複数runを性能採用条件として追加した。
4. [反映済み] one-factor-at-a-timeのA2/B1/B2/B3を設計・実装し、1回目の120-step比較を行った。
5. [一部反映] call-weighted model costはprofile済みだが、測定分散とtop candidatesのartifact保存は未実装である。
6. [未実装] cache fingerprintにdriverと実効compile optionを追加する余地がある。power/clock stateはcache keyより再検証policyで扱う。

## 7. 推奨実施順序

1. [完了] production defaultを`dot_current + legacy_raw + off`に戻す。
2. [完了] A2/B1/B2/B3と`raw_plan`を実装し、correctnessと1回目の120-step比較を行う。
3. [完了] profilerのforward/dX/dW call counterを実経路に合わせ、shape別計測を可能にする。
4. A-B-B-A順の120-step比較を最低2 run/case行い、開始温度/P-state/clockと分散を保存する。
5. compileを含まないsteady intervalにNsight Systemsを当て、allocation、launch gap、kernel time、call countを比較する。
6. model全体で原因がopaque boundaryにあると確認できた場合だけ`triton_op + wrap_triton`をprototypeする。
7. kernel自体がボトルネックと確認した後に限り、`GROUP_M`、候補拡張、新layout、CUTLASS/CuTeを一項目ずつ比較する。
8. 最後にCUDA Graphsを検討する。

## 8. 外部実装をArborの具体的なレイヤーへ落とす

外部実装は丸ごと導入しない。以下の5層に分け、各要素を独立に取り込んでA/Bする。

```text
BitNet numerical semantics
        |
        v
weight/activation representation
        |
        v
packed/dense execution kernel
        |
        v
launch selection and cache
        |
        v
torch.compile / training integration
```

### 8.1 数値意味層

対象:

- `src/model/bitlinear.py::weight_quant`
- `src/model/bitlinear.py::ternary_quantize_int8`
- `src/model/bitlinear.py::_quantize_a8_rows`
- `src/model/bitlinear.py::_quantize_a8_rows_scaled`

TorchAO BitNet と Arbor は、weight の tensor-wise abs-mean ternary と activation の row-wise abs-max INT8 という基本式が一致している。この層は性能問題の原因候補ではなく、現在の実装を維持する。

追加するのは実装ではなく参照 test である。

```python
def torchao_style_reference(x, w):
    sw = w.float().abs().mean().clamp_min(1e-5)
    w_i8 = (w.float() / sw).round().clamp(-1, 1).to(torch.int8)
    sx = x.abs().amax(dim=1) / 127
    x_i8 = (x / sx[:, None].clamp_min(1e-5 / 127)).round()
    x_i8 = x_i8.clamp(-128, 127).to(torch.int8)
    return (x_i8.int() @ w_i8.int().T) * sx[:, None] * sw
```

`tests/test_bitlinear.py` に以下を追加する。

- scalar weight scale のforwardを上記referenceと比較する。
- `M/K/N`がtile非整列のshapeも比較する。
- forwardだけでなく、転置packed weightを使うdXも比較する。
- round-half-to-even、全zero row、極小scaleを個別testにする。

TorchAOをruntime dependencyにはしない。式をtest内の小さなreferenceとして持つことで、prototype API変更の影響を避ける。

### 8.2 weight表現・packing層

対象:

- `src/model/bitlinear.py::pack_ternary_weight`
- `src/model/bitlinear.py::pack_ternary_weight_kmajor`
- `src/model/bitlinear.py::_ternary_pack_fn`
- training cache内のpacked weight生成

Microsoft GPU kernelの要素はこの層に対応する。同実装は16x32 block、block内 permutation、4値同時decodeを採用している。ArborのK-major layoutも4値を1 byteへ格納しており、基本思想は既に近い。

直ちに既存formatを置換してはいけない。追加するなら第3のlayoutとして実装する。

```text
nmajor_v1       existing [N, K/4]
kmajor_v1       existing [K/4, N]
block16x32_i2s  experimental Microsoft-style interleaved block layout
```

具体的には次の変更になる。

1. backend名とlayout名を分離する。現在は`kmajor_single_dot`というbackend名がlayoutとdecode algorithmを同時に表している。
2. `TuneKey`へ`layout`と`decode`を追加し、cache schemaとkernel versionを更新する。
3. `pack_ternary_weight_block16x32()`と逆変換を追加する。
4. `tests/test_bitlinear.py`で全3 layoutのpack/unpack round-tripを比較する。
5. training cacheは選択layoutに必要なpacked tensorだけを保持し、同じweightの全layoutを常時保持しない。

ただし、この追加はcompile境界修正後である。現状のA/B regressionをpacking変更で同時に直そうとすると原因分離できない。

### 8.3 kernel実行層

対象:

- `src/model/bitlinear.py::_packed_bitlinear_kernel`
- `src/model/bitlinear.py::_launch_packed_linear`
- `src/model/bitlinear.py::_packed_linear_impl`
- `src/model/bitlinear.py::_packed_linear_custom_op`
- `src/model/bitlinear.py::_packed_linear`

最初に、config選択を一切含まない実行関数を作る。これは`_packed_linear_execute()`として実装済みである。

```python
def _packed_linear_execute(
    x_q,
    inv_sx,
    w_packed,
    sw,
    k,
    n,
    out_dtype,
    *,
    backend,
    launch_config,
):
    """Allocation + one raw Triton launch only. No cache, timing or logging."""
```

この関数に許可する処理は以下だけである。

- shape/layout validation
- output allocation
- backend flagの静的展開
- selected configによるraw Triton launch

以下は禁止する。

- `PackedTernaryTuner.resolve()`
- JSON cache read/write
- CUDA Event timing
- Python logging
- backend fallback

現在の`_packed_linear_impl()`はresolverとlaunchを両方持つため、次の2つに分割する。

```text
_resolve_packed_linear_eager()  control plane; compile前だけ
_packed_linear_execute()        data plane; compile graphからtrace可能
```

`_packed_linear_custom_op()`は性能比較用の一時経路として残してよいが、production pathにはしない。raw Tritonで必要なPyTorch subsystemと合成できない場合だけ、`torch.library.triton_op`の本体から`wrap_triton(_packed_bitlinear_kernel)`を呼ぶ。

#### TorchAOからkernelへ取り込む候補

TorchAOの`scaled_int8_mm`から、以下を1項目ずつA/Bする。

1. **Grouped program ordering:** 現在の2D `pid_m/pid_n` gridを1D gridへし、`GROUP_M=8`でM方向をまとめてL2 localityを改善する。
2. **Candidate拡張:** `BM/BN=256`、`num_warps=2`、`num_stages=4/5`をresource filter付きで候補へ加える。
3. **Even-K specialization:** `K % BLOCK_K == 0`をconstexprにし、整列shapeではK maskを除く。
4. **Scale epilogue:** scalar/per-output weight scaleとrow scaleは現在どおりstore直前に適用する。この部分は既にTorchAO型になっているため変更しない。

最初に`GROUP_M`だけを試す。candidate集合の拡張と同時に行うと、program orderingとtile変更の寄与を分離できない。

strideについて、TorchAOは`stride_ak/stride_bk`をautotune keyに含めるが、現在のArborは`x_q.contiguous()`と固定packed layoutを要求している。この契約を維持する限りstride keyは冗長である。非contiguous入力を許可する変更を行う場合だけ追加する。

### 8.4 launch選択・cache層

対象:

- `src/model/bitlinear_tuning.py::PackedLaunchConfig`
- `src/model/bitlinear_tuning.py::TuneKey`
- `src/model/bitlinear_tuning.py::packed_launch_candidates`
- `src/model/bitlinear_tuning.py::PackedTernaryTuner`

productionの`auto`について、次の2案を小さなprototypeで比較する。

#### 案A: native Triton autotune

`_packed_bitlinear_kernel`に`@triton.autotune`を付け、`M/N/K`とsemantic flagsをkeyにする。

```python
@triton.autotune(
    configs=packed_triton_configs(),
    key=["m", "n", "k", "SCALE_PER_OUTPUT", "K_MAJOR_LAYOUT", "DECODE_V2"],
)
@triton.jit
def _packed_bitlinear_kernel(...):
    ...
```

利点は、TorchAOと同様に選択がkernel launchへ統合され、Python resolverをhot pathから外せることである。欠点は、現在の独自fingerprint、failure log、cache schemaをそのまま使えない可能性があること。Triton自体は`cache_results=True`を持つが、PyTorch公式tutorialは`torch.compile`下でサポートするautotune引数を`configs`、`key`、`restore_value`、`reset_to_zero`に限定している。このためdisk timing cacheは別のcompatibility testを通すまで有効化しない。

#### 案B: preflight + frozen plan

既存tunerでcompile前に測定し、結果だけをimmutable planへ変換する。

```python
@dataclass(frozen=True)
class PackedPlanEntry:
    key: TuneKey
    config: PackedLaunchConfig


@dataclass(frozen=True)
class PackedExecutionPlan:
    fingerprint: TuneFingerprint
    entries: tuple[PackedPlanEntry, ...]

    def require(self, key: TuneKey) -> PackedLaunchConfig:
        # Miss is an error. No tuning or fallback in compiled execution.
        ...
```

`frozen=True`でも内部に`dict`を持てば内容は変更できるため、entryはtupleで保持する。lookupはcompile前だけなので、ここを最適化する必要はない。

`src/train/train.py`では、`bitlinear_ternary_execution: raw_plan`を選び、`configure_bitlinear_ternary_tuning()`でcompile前にcache済みplanを読む。現在は事前の`custom_op + auto`実行をpreflightとし、起動中に未知shapeを自動収集する機能は追加していない。

```text
build model
install projection fusions
build/refresh packed weight cache
prepare_packed_execution_plan      new; eager only
install primitive launch specs     new; module attributes
apply_compile_settings
training loop
```

shape収集は`scripts/profile_bitlinear_shapes.py`のcounterをlibrary化して再利用する。ただしproduction startupで毎回full forward/backwardを実行するのではなく、以下の優先順位にする。

1. cacheに全keyがあれば即座にfrozen planを作る。
2. static patchingでMを計算可能ならmodel dimensionsからrequired keyを列挙する。
3. dynamic shapeだけrecord-only eager passで収集する。
4. compiled pathで未知keyが出た場合は暗黙にtuneせず、keyを表示して停止するか、明示設定時だけconservative configを使う。

productionの最終形では、compile前のinstall処理で各`BitLinear`/`BitLinearGroup`へ次のprimitive tupleを設定する案がある。ただし現在の`raw_plan`はより小さな変更として、globalの`TuneKey -> primitive tuple`をDynamo trace時に検索させる。`fullgraph=True`のCUDA testで成立を確認しており、runtimeのresolver/I/O/timingは入らない。module attribute化は性能トレースでglobal guard/traceコストが問題と確認した場合に行う。

```python
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
module._packed_fwd_launch = (128, 64, 64, 4, 2)
module._packed_dx_launch = (128, 64, 64, 4, 2)
```

`TernaryBitLinearSTE.apply()`へこの2 tupleを非Tensor引数として渡し、forwardでは`fwd_launch`、`ctx`には`dx_launch`だけを保存する。`_packed_linear_execute()`の直前でtupleを`PackedLaunchConfig`へ戻さず、その5整数をそのままTriton meta parameterへ渡す。これによりdataclass生成とplan lookupはcompile graphに入らない。

static trainingで同一moduleが複数Mを取る場合は、MごとのPython辞書をmoduleへ持たせず、その構成をstatic対応外として明示する。dynamic shape対応が必要になった時点で、案Aのnative Triton autotuneまたはshape specializationを採用する。

既存を活かす修正では案Bを優先する。既存tuner/cacheを再利用でき、現在の14 keyが既知であるstatic training構成に適するためである。案Aはgreenfield時の簡素化候補、または案Bのshape収集が複雑になり過ぎた場合の比較案とする。両方をproductionに残して複雑化させない。

### 8.5 autograd・training統合層

対象:

- `src/model/bitlinear.py::TernaryBitLinearSTE`
- `src/model/bitlinear.py::BitLinear`
- `src/model/bitlinear.py::BitLinearGroup`
- `src/train/train.py::main`

`TernaryBitLinearSTE`の責務は以下に限定する。

```text
forward:
    A8 row quantize
    packed_linear_execute(frozen config)
    save x_int8/inv_sx/w_packed_t/row_scale

backward dX:
    scaled A8 row quantize
    packed_linear_execute(frozen config)

backward dW:
    existing INT8 or FP8 wgrad
```

TorchAOと同じく、forwardで作成した`x_int8`とscaleをbackwardまで保存する現在の構造は維持する。TorchAOのtensor subclassへ置換しない。Arborの`BitLinearGroup`、projection fusion、packed transpose cacheとの統合を壊すためである。

原因分離期間だけ、設定へ次を追加する。

```yaml
speed:
  bitlinear_ternary_execution: raw_plan  # legacy_raw | legacy_custom_op | custom_op | raw | raw_plan
```

これは恒久的な利用者向け機能ではなく、同一backend/configで実行境界だけを比較するdiagnostic switchである。A/B完了後はwinnerだけをproductionに残す。

### 8.6 benchmark・profile層

対象:

- `scripts/bench_bitlinear_kernels.py`
- `scripts/profile_bitlinear_shapes.py`

`bench_bitlinear_kernels.py`と`profile_bitlinear_shapes.py`へ`--execution-path`を追加し、同じ固定configまたはcache済みplanを複数経路で呼べるようにした。

```text
raw
raw_plan
legacy_raw
legacy_custom_op
custom_op
```

ここではautotune winnerを比較しない。同じbackend、layout、tile、input tensorを使い、境界 overheadだけを見る。

`profile_bitlinear_shapes.py`は既にpacked ternaryとdense INT8をshape別・call-weightedで比較できるため、この機能を新しいscriptへ重複実装しない。以下だけ拡張する。

- execution path別のfwd/dX総時間
- shape別call count x median/p10/p90
- selected configとmeasured bestの比率
- JSON出力

Hugging Face型の`unpack + F.linear`はperformance winner候補ではなく、correctnessと「packing自体の価値」を確認するreferenceとしてbenchmarkへ追加する。

### 8.7 test層

対象:

- `tests/test_bitlinear.py`
- `tests/test_bitlinear_tuning.py`
- `tests/test_train.py`

追加するtest matrix:

| Test | CPU | CUDA | 目的 |
|---|---:|---:|---|
| TorchAO式referenceとの量子化一致 | yes | yes | numerical semantics固定 |
| pack/unpack全layout round-trip | yes | yes | representation検証 |
| raw/legacy/custom_op出力一致 | no | yes | 実行境界のcorrectness |
| raw path `torch.compile(fullgraph=True)` | no | yes | traceability |
| first call tune、cacheからraw_plan | no | yes | autotune lifecycle |
| unknown frozen-plan keyの明示failure | yes | optional | hot pathで暗黙tuneしない |
| `GROUP_M=1`対`8`の出力一致 | no | yes | program ordering変更 |

性能testは絶対msでfailさせない。専用GPU runでのみ以下をgateにする。

```text
selected <= 1.20 * measured_best
steady bytes/s >= legacy bytes/s
loss/EMA identical under fixed seed
```

### 8.8 採用・不採用の一覧

| 外部実装の要素 | 判断 | Arborでの反映先 |
|---|---|---|
| TorchAOのBitNet量子化式 | 採用済み | reference testを追加 |
| TorchAOのsaved INT8 activation | 採用済み | `TernaryBitLinearSTE`を維持 |
| TorchAOのscale epilogue | 採用済み | packed kernel store前scaleを維持 |
| TorchAOのTriton autotune | prototype採用 | production auto案A |
| TorchAOの`GROUP_M` ordering | 個別A/B | `_packed_bitlinear_kernel` |
| TorchAOの広いtile候補 | resource filter後にA/B | `packed_launch_candidates` |
| TorchAOのtensor subclass全置換 | 不採用 | fusion/cacheとの不整合が大きい |
| TorchAOのlibrary op wrapper | 不採用 | opaque境界問題を再導入し得る |
| Microsoft 16x32 interleaved layout | 後段のbackend候補 | packing/kernel層 |
| Microsoft `dp4a` GEMVを直接移植 | 不採用 | 学習GEMMのMと目的が異なる |
| HF unpack + `F.linear` | referenceのみ | benchmark/correctness |
| T-MAC LUT kernel | 不採用 | CPU/NPU推論向け |

### 8.9 実装順序

各段階でA/Bを通し、複数要因を一度に変更しない。

1. [完了] `_packed_linear_execute()`を追加し、現configのraw固定経路を作る。
2. [完了] `--execution-path`と`--boundary-ab`でrawとcurrent custom opを同一tile比較する。
3. [完了] raw経路を`TernaryBitLinearSTE`から呼び、実学習A2/B1/B2/B3を行う。
4. [完了] 既存cacheからfrozen `raw_plan`を作り、resolverとcustom opをcompiled hot pathから外す。
5. native Triton autotuneはgreenfield比較案としてprototypeし、compile/cache要件を確認する。
6. `GROUP_M=8`だけを追加してA/Bする。
7. tile候補を段階的に広げる。
8. dense INT8とのcall-weighted比較を再実行する。
9. packedが依然負けるshapeに限り、Microsoft-style block layoutを試作する。
10. [未合格] 交互順のsteady 100-step A/B合格後にだけdefaultを変更する。

最初の変更単位だった「現在のpacked kernelをtrace可能なraw実行へ戻す」は実装済みである。それだけでは性能優位にならなかったため、外部実装の最適化要素を載せる前にmodel-level timelineで次のボトルネックを確定する。

### 8.10 非ブロッキングの周辺課題: patch pooling

利用時の観測として、`patch_pooling: mean` は encode 時に中央部分の情報が欠落気味になり、
`legacy` へ戻すと過剰な情報圧縮が緩和された。これは現時点では体系的な比較結果ではなく、
再現条件、品質指標、速度・memoryとのtrade-offを分離して追試する必要がある。

この課題は重要だが、現在の最優先はpacked ternary BitLinearのmodel-level性能不一致を
原理とtimelineから解明し、一般化可能な改善を行うことである。したがってpooling変更を
BitLinear A/Bへ混ぜず、独立した低優先トラックとして扱う。

## 9. 参照資料

- [PyTorch `torch.library` documentation](https://docs.pytorch.org/docs/main/library.html): raw Triton、`triton_op`、`custom_op` の compile visibility の違い。
- [Using User-Defined Triton Kernels with `torch.compile`](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html): raw Triton、`triton_op`、`wrap_triton`、`triton.autotune` の compile integration と制約。
- [TorchDynamo APIs for fine-grained tracing](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fine_grain_apis.html): `disable`、graph break、`allow_in_graph` の用途と注意。
- [`torch.compile` troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html): `TORCH_LOGS=graph_breaks` 等による診断。
- [Triton `triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html): configs、key、pruning hooks、benchmark callback、`cache_results`。
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html): GEMM autotune の標準的構成。
- [NVIDIA CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): graph による CPU launch cost 削減と workflow optimization。
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): CUDA API/kernel/memory trace。
- [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html): GPU gap、kernel duration、CUDA API/synchronization 分析。
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/): CUTLASS/CuTe の全体像。
- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html): GEMM candidate の列挙と性能検証。
- [Autotuning with the CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html): compile、benchmark、cache を分ける autotuning pattern。
- [CUTLASS DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html): 対応 OS、Python、CUDA Toolkit、driver 要件。
- [TorchAO BitNet training implementation](https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/bitnet.py): BitNet量子化式、saved tensor、FSDP2 2-bit packing、forward/backward構造。
- [TorchAO scaled INT8 matmul](https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/int8_mm.py): Triton autotune候補、`GROUP_M` ordering、scale epilogue。
- [Microsoft BitNet GPU kernel](https://github.com/microsoft/BitNet/blob/main/gpu/README.md): W2A8 GEMV、16x32 block permutation、4値decode、`dp4a`。
- [Hugging Face BitNet integration](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitnet.py): packed weightのunpack、fake quant、`F.linear` reference経路。
- [Microsoft T-MAC](https://github.com/microsoft/T-MAC): CPU/NPU向け低bit LUT inference。
