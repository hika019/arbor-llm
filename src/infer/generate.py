"""Checkpoint からのバイト単位テキスト生成.

使い方:
    # 最新 checkpoint で 1 回生成
    python -m src.infer.generate --ckpt latest --prompt "日本の四季は"

    # 対話モード (プロンプトを繰り返し入力)
    python -m src.infer.generate --ckpt best --interactive

設計:
- モデル構成は checkpoint ディレクトリ内の config.yaml を正とする
  (学習時の実効 config が保存されているので、現在の configs/ と乖離していても安全)。
- BLT は KV cache を持たないため、1 バイト生成するごとに全系列を再フォワードする。
  単発のサンプル生成・動作確認用であり、スループットは求めない。
- 重みは生成中固定なので BitLinear の packed weight cache を有効化して再利用する。
- token = byte値 + 4 (0..3 は BLT の BOE/BOS/EOS/PAD 特殊 ID)。サンプリング時は
  特殊 ID を必ずマスクする。出力は UTF-8 incremental decoder で逐次復号する。
"""
from __future__ import annotations

import argparse
import codecs
import os
import sys
import time
from pathlib import Path
from typing import Iterator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BYTE_OFFSET = 4  # 生バイト b は token id (b + 4)。0..3 は特殊 ID
VOCAB_SIZE = 260


# ------------------------------------------------------------- model loading
def resolve_checkpoint(which: str, ckpt_root: Path) -> Path:
    """'latest' | 'best' | 'final' | step 数 | パス を checkpoint dir に解決する."""
    p = Path(which)
    if p.is_dir() and (p / "model.safetensors").exists():
        return p
    from src.train.checkpoint import CheckpointManager

    mgr = CheckpointManager.__new__(CheckpointManager)  # mkdir せず resolve だけ使う
    mgr.root = ckpt_root
    resolved = mgr.resolve(int(which) if which.isdigit() else which)
    if resolved is None or not (resolved / "model.safetensors").exists():
        raise FileNotFoundError(f"checkpoint not found: {which} (root={ckpt_root})")
    return resolved


def load_checkpoint_config(ckpt_dir: Path, config_path: Path | None = None) -> dict:
    """checkpoint 内の config.yaml を優先して読む。無ければ --config を要求する."""
    if config_path is not None:
        with config_path.open() as f:
            return yaml.safe_load(f)
    cfg_file = ckpt_dir / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{cfg_file} がありません。--config で学習時の config を指定してください。"
        )
    with cfg_file.open() as f:
        return yaml.safe_load(f)


def _strip_compile_prefix(state: dict) -> dict:
    """torch.compile 済みモデルで保存された '_orig_mod.' prefix を剥がす."""
    if not any(k.startswith("_orig_mod.") for k in state):
        return state
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def load_inference_model(
    ckpt_dir: Path,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """checkpoint からモデルを構築してロードし、eval モードで返す."""
    from safetensors.torch import load_file as safe_load

    model_cfg = dict(cfg["model"])
    model_cfg["gradient_checkpointing"] = False  # 推論では不要
    if model_cfg.get("arch", "arbor") == "byte_lm":
        from src.model.arbor import build_byte_lm as build_model
    else:
        from src.model.arbor import build_arbor as build_model
    model = build_model(model_cfg).to(device=device, dtype=dtype)

    state = _strip_compile_prefix(safe_load(str(ckpt_dir / "model.safetensors"), device="cpu"))
    model.load_state_dict(state, strict=True)
    model.eval()

    # 推論では重みが固定なので BitLinear を packed ternary に凍結して高速化
    from src.model.bitlinear import freeze_bitlinear_for_inference

    n_frozen = freeze_bitlinear_for_inference(model)
    if n_frozen:
        print(f"[generate] bitlinear_frozen={n_frozen} layers (packed ternary inference)")
    return model


# ---------------------------------------------------------------- sampling
def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    """最終位置の logits [vocab] から次トークンを 1 つ選ぶ。特殊 ID はマスク済み前提."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # 累積 top_p を超えた裾を捨てる (先頭 1 個は必ず残す)
        cut = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def generate_stream(
    model: torch.nn.Module,
    prompt: str | bytes,
    *,
    max_new_bytes: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    max_context: int = 2048,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int | None = None,
    use_cache: bool = True,
) -> Iterator[str]:
    """1 バイトずつ生成し、UTF-8 として確定した文字列片を逐次 yield する.

    ArborModel なら既定で 2 階層 KV cache (ArborByteGenerator) を使う。
    use_cache=False でフルフォワード方式 (検証用・遅い) に切り替え。
    """
    from src.model.arbor import ArborByteGenerator, ArborModel

    if device is None:
        device = next(model.parameters()).device
    raw = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
    if not raw:
        raw = b"\n"
    ids = [b + BYTE_OFFSET for b in raw]

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    gen: ArborByteGenerator | None = None
    if use_cache and isinstance(model, ArborModel):
        gen = ArborByteGenerator(model)
        last_logits = gen.prefill(ids)

    for _ in range(max_new_bytes):
        if gen is not None:
            logits = last_logits.float()
        else:
            window = ids[-(max_context - 1):]
            x = torch.tensor([window], dtype=torch.long, device=device)
            ctx = (
                torch.autocast(device_type=device.type, dtype=dtype)
                if use_autocast
                else torch.no_grad()
            )
            with ctx:
                logits = model(x).logits[0, -1].float()
        logits[:BYTE_OFFSET] = float("-inf")  # 特殊 ID は出さない
        logits = logits[:VOCAB_SIZE]
        # multinomial を CPU generator で引くため logits を CPU に移す
        next_id = _sample_next(logits.cpu(), temperature, top_k, top_p, generator)
        ids.append(next_id)
        if gen is not None:
            last_logits = gen.push(next_id)
        piece = decoder.decode(bytes([next_id - BYTE_OFFSET]))
        if piece:
            yield piece
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


def generate_text(model: torch.nn.Module, prompt: str | bytes, **kwargs) -> str:
    """generate_stream をまとめて文字列で返す."""
    return "".join(generate_stream(model, prompt, **kwargs))


def generate_samples(
    model: torch.nn.Module,
    prompts: list[str],
    *,
    max_new_bytes: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_context: int = 2048,
    seed: int | None = 42,
) -> list[tuple[str, str]]:
    """学習ループの checkpoint 時サンプル生成用。(prompt, completion) のリストを返す.

    呼び出し側で model.eval() / model.train() を切り替えること。
    seed 固定なので step 間で同条件の比較ができる。
    """
    out = []
    for prompt in prompts:
        text = generate_text(
            model, prompt,
            max_new_bytes=max_new_bytes, temperature=temperature,
            top_p=top_p, max_context=max_context, seed=seed,
        )
        out.append((prompt, text))
    return out


# --------------------------------------------------------------------- CLI
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="latest", help="'latest' | 'best' | 'final' | step数 | パス")
    p.add_argument("--ckpt-dir", default="./checkpoints", type=Path)
    p.add_argument("--config", default=None, type=Path,
                   help="checkpoint に config.yaml が無い場合の学習 config")
    p.add_argument("--prompt", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max-new-bytes", default=200, type=int)
    p.add_argument("--temperature", default=0.8, type=float, help="0 で greedy")
    p.add_argument("--top-k", default=0, type=int)
    p.add_argument("--top-p", default=0.95, type=float)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--no-cache", action="store_true",
                   help="KV cache を使わずフルフォワードで生成 (検証用・遅い)")
    args = p.parse_args()

    if not args.interactive and args.prompt is None:
        p.error("--prompt か --interactive のどちらかが必要")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    cfg = load_checkpoint_config(ckpt_dir, args.config)
    max_context = int(cfg.get("model", {}).get("max_position_embeddings", 2048))

    print(f"[generate] checkpoint={ckpt_dir}")
    t0 = time.perf_counter()
    model = load_inference_model(ckpt_dir, cfg, device)
    print(f"[generate] model loaded in {time.perf_counter() - t0:.1f}s device={device}")

    def run(prompt: str) -> None:
        print(f"--- prompt: {prompt!r}")
        sys.stdout.write(prompt)
        sys.stdout.flush()
        t0 = time.perf_counter()
        n = 0
        for piece in generate_stream(
            model, prompt,
            max_new_bytes=args.max_new_bytes, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            max_context=max_context, seed=args.seed,
            use_cache=not args.no_cache,
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
            n += len(piece.encode("utf-8"))
        dt = time.perf_counter() - t0
        print(f"\n--- {n} bytes in {dt:.1f}s ({n / max(dt, 1e-9):.1f} B/s)")

    if args.interactive:
        print("[generate] 対話モード。空行 or Ctrl+D で終了。")
        while True:
            try:
                prompt = input("\nprompt> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            run(prompt)
    else:
        run(args.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
