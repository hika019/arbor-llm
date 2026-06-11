# hika019/arbor-llm コードレビュー

対象リポジトリ: <https://github.com/hika019/arbor-llm>  
観点: コードとしての問題点、学習・評価・推論の実用性

## 結論

現状の公開リポジトリは、READMEで説明されているような「BLT + BitNet b1.58 の1B級 tokenizer-free LLM」として、そのまま学習・推論できる完成度ではありません。

実体としては、**研究案の骨組み + smoke test 用 stub 実装**に近い状態です。

特に問題なのは、READMEでは `third_party/blt` にBLT forkが入っている前提になっている一方で、実際の `third_party` には `.gitkeep` しかなく、BLT本体が含まれていない点です。さらに、`src/model/arbor_blt.py` はBLT importに失敗してもエラーで止まらず、小さいstub Transformerへ自動フォールバックします。

つまり、README通りに学習を走らせても、気づかないまま本命のBLT+BitNetではなく、stubモデルを学習している可能性があります。これは致命的です。

---

## 推論はできるか

現状では、**まともなテキスト生成用の推論コードはほぼありません**。

`src/eval/run_eval.py` はチェックポイントを読んでperplexityを測る評価スクリプトで、`byte_perplexity` を呼ぶだけです。生成、sampling、top-k/top-p、temperature、byte decode、CLIでのプロンプト入力、KV cacheなどは実装されていません。

理屈上は、モデルが `logits` を返すので、byte IDを1トークンずつ足していく簡易generate loopは書けます。ただし次の制約があります。

1. 学習済みチェックポイントが無ければランダム出力になる。
2. BLT本体が無いとstub推論になる。
3. tokenizer-freeのbyte入出力処理が推論用APIとして整っていない。
4. KV cacheが無いため、素朴な生成は毎stepで全コンテキストを再計算して遅い。
5. `torch.compile` 後のcheckpoint保存・読込問題で、学習済み重みが正しくロードされない危険がある。

したがって、**perplexity評価は一応できる設計だが、会話・文章生成としての推論は未実装に近い**という評価です。

---

## 主なコード上の問題点

### 1. README・specと実装が一致していない

specでは「BLT標準のentropy-based dynamic patching」と書かれていますが、実装側では `patching_mode` のデフォルトが `"space"` になっています。

またspecではpatch-byte間cross-attentionをFP維持すると説明していますが、実装では `cross_attn_encoder=False`、`cross_attn_decoder=False` になっており、cross-attentionを明示的に無効化しています。

これは細部の未実装ではなく、**アーキテクチャの中核がREADME/specとズレている**レベルです。

### 2. Local Encoder / Local Decoderが実装されていない

`src/model/local_encoder.py` と `src/model/local_decoder.py` は、実質的にdocstringとTODOだけです。

本命BLTが `third_party/blt` にある前提なら薄いwrapperでも成立しますが、その `third_party/blt` が存在しないため、現状のリポジトリ単体ではBLT系のlocal/global構成は成立していません。

### 3. Global Latent Transformerも本体実装ではない

`src/model/global_latent.py` はGlobal Latent Transformerの実装ではなく、`nn.Linear` を `BitLinear` に差し替えるutilityです。

つまり、このリポジトリ内にGlobal Latent Transformerを自前実装しているわけではありません。BLT本体への依存が前提です。

### 4. BitLinearは「BitNet風の参照実装」で、実際の高速・省メモリBitNetではない

`BitLinear` のdocstringにも「reference implementation」「packed ternary kernel later」と書かれており、現状は最適化kernelではありません。

forwardでは重みとactivationを量子化したあと、結局 `F.linear(x_q, w_q) * sx * sw` を使っています。

そのため、BitNet b1.58の売りである実際の高速化・省メモリ化は、この実装だけでは出ません。研究実験用のstraight-through estimatorに近いです。

細かい点では、activation quantizationのdocstringは `[-127, 127]` と言っていますが、実装は `[-128, 127]` にclipしています。大事故ではありませんが、仕様と実装の不一致です。

### 5. Flash Attention設定が実際には効いていない可能性が高い

configでは `use_flash_attention: true` になっています。

しかし `requirements.txt` では `flash-attn` がコメントアウトされており、実装側でもBLT構築時に `attn_impl="sdpa"`、`cross_attn_use_flex_attention=False` になっています。

つまりREADME/spec/config上は高速化前提に見えますが、実際にはPyTorch SDPA頼みで、FlashAttentionを確実に使う構成ではありません。

### 6. checkpointまわりが危険

configには `async_save: true` がありますが、checkpoint実装を見ると、実際に非同期なのは古いcheckpointのpruneだけです。

重み、optimizer、scheduler、dataloader state、RNG、metaの保存は同期的に行っています。つまり「async checkpoint save」というより「同期保存 + 非同期prune」です。大きいモデルでは保存時に学習が止まります。

さらに、checkpoint loadが `strict=False` です。missing/unexpected keyを表示するだけで処理を続行します。

特にtrain loopは `torch.compile` してからcheckpoint load/saveしています。PyTorchでは `torch.compile` 済みモデルの `state_dict` keyに `_orig_mod` が絡み、通常モデルへのロードでkey mismatchが起きる既知の問題があります。

この組み合わせだと、**保存したcheckpointをeval側の非compileモデルに読ませたとき、重みが一部または大部分ロードされず、それでも `strict=False` で通ってしまう**リスクがあります。

### 7. 「best checkpoint」がvalidation bestではない

train loopでは `best_loss` を更新してbest checkpointを保存していますが、見ているlossはvalidation lossではなく学習中のlossです。

README上は `--resume best` が用意されていますが、これは「汎化性能が最良」という意味ではありません。

現状では「train lossが一番低かったcheckpoint」です。

### 8. data streaming / resumeが弱い

dataset設計コメントでは `shard_index, byte_offset, samples_emitted` のようなresumable stateを想定しています。

しかしHF streaming側では、resume時に `skip = samples_emitted` として、ストリームを最初から流し直してyieldをskipする実装です。

長時間学習後のresumeでは、復帰までに大量のデータを再走査する可能性があります。

local mmap側も気になります。1サンプルを出すたびにcursorを1 byteだけ進めています。context長2048なら、連続サンプルがほぼ2047 byte重複します。

意図的なsliding windowならよいですが、普通のpretrainingではデータ多様性・I/O効率の面で疑問です。

### 9. DataLoaderのworker対応がない

configでは `num_workers: 0` なので今は問題が出にくいですが、DataLoader側は `num_workers` を受け取る設計です。

一方、IterableDataset側に `get_worker_info()` を使ったworker shardingが見当たりません。

`num_workers > 0` にすると、複数workerが同じstreamを重複して読む可能性があります。将来的にハマりやすいです。

### 10. CPU実行で壊れやすい

train loopはdeviceを `cuda` または `cpu` で選んでいますが、その後のautocastは `torch.autocast("cuda", dtype=torch.bfloat16)` で固定です。

eval側も同様に `torch.autocast("cuda", dtype=torch.bfloat16)` 固定です。

GPU前提ならよいですが、CPU fallbackのコードを書いているのにautocastだけCUDA固定なのは不整合です。

### 11. 依存関係が再現しづらい

`requirements.txt` は `transformers>=4.44`、`datasets>=2.20` など下限指定中心で、PyTorchも「assumed but intentionally not pinned」です。

研究コードとしては普通にありますが、`torch.compile`、`bitsandbytes`、FlashAttention、HF streamingあたりはバージョン差で壊れやすいので、1B学習を謳うならlock fileか検証済み環境が欲しいです。

---

## 優先して直すべきところ

### 1. BLTが無いときにstubへ黙って落ちる挙動をやめる

`backend: stub` を明示した時だけstubを使い、それ以外はhard errorにした方が安全です。

```python
if backend != "stub" and not (_BLT_PATH / "bytelatent").exists():
    raise RuntimeError(
        "third_party/blt is missing. "
        "Run `git submodule update --init --recursive` or install BLT explicitly."
    )
```

### 2. checkpoint保存はcompile済みwrapperではなく元モデルを保存する

```python
model_to_save = getattr(model, "_orig_mod", model)
state = model_to_save.state_dict()
```

ロード時も、基本は `strict=True` にするべきです。key変換が必要な場合だけ明示的に `_orig_mod.` prefixを処理します。

`strict=False` で雑に通すのは、この手のモデルでは危険です。

### 3. 最小のgenerate APIを追加する

最低限、以下のような簡易generate loopが必要です。

```python
@torch.no_grad()
def generate(model, prompt: str, max_new_bytes: int = 128, temperature: float = 1.0):
    model.eval()

    # dataset側の仕様に合わせるなら raw byte + offset 4
    ids = [b + 4 for b in prompt.encode("utf-8")]
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device)[None, :]

    for _ in range(max_new_bytes):
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        next_logits = logits[:, -1, :] / temperature

        # special ids 0..3を避けるならbyte領域だけ使う
        probs = torch.softmax(next_logits[:, 4:260], dim=-1)
        next_byte = torch.multinomial(probs, num_samples=1) + 4

        x = torch.cat([x, next_byte], dim=1)

    byte_values = [
        int(t) - 4
        for t in x[0].tolist()
        if 4 <= int(t) < 260
    ]
    return bytes(byte_values).decode("utf-8", errors="replace")
```

ただしこれは簡易版です。長文生成にはcontext truncation、EOS扱い、KV cache、top-k/top-p、repetition penalty、batchingなどが必要です。

---

## 総評

このリポジトリは、アイデアとしては面白いです。BLT、byte-level、BitNet b1.58、1B級モデルを4090で動かす、という方向性は分かります。

ただし現状のコードは、少なくとも公開されている範囲では次の評価になります。

- **「BLT+BitNet 1Bをそのまま学習できる実装」ではない**
- **「推論用LLMとして使える状態」でもない**
- **「README/spec/configと実コードの乖離が大きい」**

本当に動く研究コードにするなら、最初にやるべきことは、`third_party/blt` を正しくsubmodule化し、stub fallbackを禁止し、1つの小さいconfigで「本物のBLT pathがforward/backward/checkpoint/eval/generateまで通る」ことをCIかsmoke testで保証することです。

---

## 参照した主な箇所

- README: <https://github.com/hika019/arbor-llm>
- third_party: <https://github.com/hika019/arbor-llm/tree/main/third_party>
- `src/model/arbor_blt.py`: <https://github.com/hika019/arbor-llm/blob/main/src/model/arbor_blt.py>
- `src/model/local_encoder.py`: <https://github.com/hika019/arbor-llm/blob/main/src/model/local_encoder.py>
- `src/model/local_decoder.py`: <https://github.com/hika019/arbor-llm/blob/main/src/model/local_decoder.py>
- `src/model/global_latent.py`: <https://github.com/hika019/arbor-llm/blob/main/src/model/global_latent.py>
- `src/model/bitlinear.py`: <https://github.com/hika019/arbor-llm/blob/main/src/model/bitlinear.py>
- `src/train/train.py`: <https://github.com/hika019/arbor-llm/blob/main/src/train/train.py>
- `src/train/checkpoint.py`: <https://github.com/hika019/arbor-llm/blob/main/src/train/checkpoint.py>
- `src/data/byte_dataset.py`: <https://github.com/hika019/arbor-llm/blob/main/src/data/byte_dataset.py>
- `src/eval/run_eval.py`: <https://github.com/hika019/arbor-llm/blob/main/src/eval/run_eval.py>
- `src/eval/perplexity.py`: <https://github.com/hika019/arbor-llm/blob/main/src/eval/perplexity.py>
- `configs/arbor_1b.yaml`: <https://github.com/hika019/arbor-llm/blob/main/configs/arbor_1b.yaml>
- `requirements.txt`: <https://github.com/hika019/arbor-llm/blob/main/requirements.txt>
