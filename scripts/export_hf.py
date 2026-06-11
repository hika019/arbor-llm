"""checkpoint を HuggingFace 形式のモデルディレクトリに変換する.

使い方:
    python scripts/export_hf.py --ckpt latest --verify
    python scripts/export_hf.py --ckpt 4000 --out export/arbor-step4000

出力ディレクトリは transformers でそのまま読める:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(OUT, trust_remote_code=True,
                                                 dtype="auto").cuda()
    tok = AutoTokenizer.from_pretrained(OUT, trust_remote_code=True)
    ids = tok("日本の四季は", return_tensors="pt").input_ids.cuda()
    print(tok.decode(model.generate(ids, max_new_tokens=100)[0]))

Arbor は独自アーキテクチャ (バイトレベル階層 Transformer × BitNet b1.58) のため、
モデル定義コード (arbor_model/, torch のみ依存) を出力ディレクトリに同梱し
trust_remote_code でロードする。GGUF 変換が前提の ollama / LM Studio では
読めない点に注意 (詳細は出力される README.md)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.infer.generate import (  # noqa: E402
    load_checkpoint_config,
    resolve_checkpoint,
    _strip_compile_prefix,
)

_HF_TEMPLATES = _ROOT / "src" / "hf"

# config.json に書き出す model 設定のフィールド (modeling_arbor.py と一致させる)
# entropy_model_ckpt は除外: 重みは model.safetensors に同梱される
_ARBOR_FIELDS = (
    "vocab_size", "patch_size", "max_bytes",
    "patching_mode", "min_patch_len", "max_patch_len",
    "entropy_threshold", "entropy_model",
    "hidden_size", "num_heads", "num_kv_heads", "intermediate_size",
    "num_hidden_layers",
    "local_hidden_size", "local_num_heads", "local_num_kv_heads",
    "local_intermediate_size", "num_local_encoder_layers", "num_local_decoder_layers",
    "rope_theta", "norm_eps", "bitnet",
)


def _copy_arbor_model(out: Path) -> None:
    """src/model/ を arbor_model/ パッケージとして同梱 (import を付け替え)."""
    pkg = out / "arbor_model"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("bitlinear.py", "arbor.py"):
        text = (_ROOT / "src" / "model" / name).read_text(encoding="utf-8")
        (pkg / name).write_text(text.replace("src.model.", "arbor_model."), encoding="utf-8")


def _export_weights(ckpt_dir: Path, out: Path) -> int:
    from safetensors.torch import load_file as safe_load
    from safetensors.torch import save_file as safe_save

    state = _strip_compile_prefix(safe_load(str(ckpt_dir / "model.safetensors"), device="cpu"))
    # HF ラッパは self.model = <学習時の model> なので "model." を前置する
    state = {f"model.{k}": v.contiguous() for k, v in state.items()}
    safe_save(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    return sum(v.numel() for v in state.values())


def _write_config(cfg: dict, meta: dict, out: Path) -> None:
    m = cfg["model"]
    config = {name: m[name] for name in _ARBOR_FIELDS if name in m}
    config.update(
        model_type="arbor",
        architectures=["ArborForCausalLM"],
        auto_map={
            "AutoConfig": "modeling_arbor.ArborConfig",
            "AutoModelForCausalLM": "modeling_arbor.ArborForCausalLM",
        },
        torch_dtype="bfloat16",
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        code_root=str(out.resolve()),
        arbor_export={
            "global_step": meta.get("global_step"),
            "best_loss": meta.get("best_loss"),
            "git_sha": meta.get("git_sha"),
        },
    )
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generation_config = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        # 特殊 ID は学習で一度も target にならないため必ず抑制する
        "suppress_tokens": [0, 1, 2, 3],
        "use_cache": False,
    }
    (out / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2), encoding="utf-8"
    )

    tokenizer_config = {
        "tokenizer_class": "ArborByteTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_arbor.ArborByteTokenizer", None]},
        "model_max_length": m.get("max_bytes", 2048),
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2), encoding="utf-8"
    )
    (out / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}, indent=2),
        encoding="utf-8",
    )


def _write_readme(meta: dict, n_params: int, out: Path) -> None:
    step = meta.get("global_step")
    loss = meta.get("best_loss", float("nan"))
    (out / "README.md").write_text(f"""---
language: [ja, en]
pipeline_tag: text-generation
---

# Arbor (バイトレベル階層 Transformer × BitNet b1.58) — step {step}

tokenizer 不要のバイトレベル LLM。静的 patching (4 bytes/patch) の階層構造で、
Local/Global の全 Transformer 層が BitNet b1.58 (W1.58A8 BitLinear + SubLN,
ReLU² gated FFN)。パラメータ数 {n_params / 1e6:.0f}M / 学習 step {step} /
train loss (EMA) {loss:.3f}。

## 使い方 (transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "このディレクトリへのパス"
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype="auto").cuda().eval()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

ids = tok("日本の四季は", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=100)
print(tok.decode(out[0]))
```

## 必要環境

- torch >= 2.4 (CUDA 推奨), transformers >= 4.45
- モデル定義は同梱 `arbor_model/` (依存は torch のみ)

## 制限

- **ollama / LM Studio では動かない**: これらは llama.cpp (GGUF) の既知
  アーキテクチャのみ対応で、バイトレベル階層構造 + BitLinear には変換器が
  存在しない。transformers (Python) から利用すること。
- transformers の `generate()` は 1 バイトごとに全系列を再フォワードする (遅い)。
  高速版は同梱の 2 階層 KV cache 生成器を直接使う:
  ```python
  from arbor_model.arbor import ArborByteGenerator
  from arbor_model.bitlinear import freeze_bitlinear_for_inference
  freeze_bitlinear_for_inference(model.model)
  gen = ArborByteGenerator(model.model)
  logits = gen.prefill(tok("日本の").input_ids)  # 以後 gen.push(next_id)
  ```
- tokenizer はバイト直 (token = byte + 4)。chat template は無い (base model)。
""", encoding="utf-8")


def export(ckpt_dir: Path, cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ckpt_dir / "meta.json").read_text())

    shutil.copy2(_HF_TEMPLATES / "modeling_arbor.py", out / "modeling_arbor.py")
    shutil.copy2(_HF_TEMPLATES / "tokenization_arbor.py", out / "tokenization_arbor.py")
    _copy_arbor_model(out)
    n_params = _export_weights(ckpt_dir, out)
    _write_config(cfg, meta, out)
    _write_readme(meta, n_params, out)
    print(f"[export] {ckpt_dir} -> {out} ({n_params / 1e6:.0f}M params)")


def verify(ckpt_dir: Path, cfg: dict, out: Path) -> bool:
    """エクスポート結果を学習時モデルとロジット比較し、generate も流す."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.infer.generate import load_inference_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("[verify] loading reference model (training stack)...")
    ref = load_inference_model(ckpt_dir, cfg, device, dtype=dtype)
    print("[verify] loading exported model (transformers)...")
    hf = AutoModelForCausalLM.from_pretrained(
        out, trust_remote_code=True, dtype=dtype
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)

    g = torch.Generator().manual_seed(0)
    x = torch.randint(4, 260, (1, 96), generator=g).to(device)
    with torch.inference_mode():
        ref_logits = ref(x).logits.float()
        hf_logits = hf(x).logits.float()
    diff = (ref_logits - hf_logits).abs().max().item()
    print(f"[verify] logits max abs diff = {diff:.3e}")
    ok = diff < 1e-3

    decoded = tok.decode(tok("こんにちは, hello!").input_ids)
    roundtrip = decoded == "こんにちは, hello!"
    print(f"[verify] tokenizer roundtrip: {roundtrip} ({decoded!r})")
    ok = ok and roundtrip

    ids = tok("日本の", return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        gen = hf.generate(ids, max_new_tokens=24, do_sample=False)
    print(f"[verify] generate: {tok.decode(gen[0])!r}")
    ok = ok and gen.shape[1] == ids.shape[1] + 24

    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="latest", help="'latest' | 'best' | 'final' | step数 | パス")
    p.add_argument("--ckpt-dir", default="./checkpoints", type=Path)
    p.add_argument("--config", default=None, type=Path)
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--verify", action="store_true", help="エクスポート後にロジット一致を検証")
    args = p.parse_args()

    ckpt_dir = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    cfg = load_checkpoint_config(ckpt_dir, args.config)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    out = args.out or Path("export") / f"{cfg.get('run_name', 'arbor')}-step{meta['global_step']}"

    export(ckpt_dir, cfg, out)
    if args.verify:
        return 0 if verify(ckpt_dir, cfg, out) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
