from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.infer.generate import BYTE_OFFSET, load_checkpoint_config, load_inference_model, resolve_checkpoint


def byte_ids(text: str) -> list[int]:
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def byte_label(byte: int) -> str:
    if byte in (0x0A, 0x0D, 0x09):
        return repr(bytes([byte]).decode("ascii"))
    if 0x20 <= byte <= 0x7E:
        return bytes([byte]).decode("ascii")
    return f"0x{byte:02x}"


def patch_layout(text: str, patch_size: int) -> str:
    raw = text.encode("utf-8")
    chunks = []
    for i in range(0, len(raw), patch_size):
        chunk = raw[i:i + patch_size]
        decoded = chunk.decode("utf-8", errors="replace")
        chunks.append(f"{i // patch_size}:{decoded!r}/{len(chunk)}B")
    return " | ".join(chunks)


@torch.inference_mode()
def next_logits(model: torch.nn.Module, prompt: str, device: torch.device) -> torch.Tensor:
    ids = byte_ids(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).logits[0, -1].float()
    logits[:BYTE_OFFSET] = float("-inf")
    return logits[:260].cpu()


@torch.inference_mode()
def continuation_bpb(
    model: torch.nn.Module,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    p_ids = byte_ids(prompt)
    c_ids = byte_ids(continuation)
    if not c_ids:
        return float("nan")
    input_ids = p_ids + c_ids[:-1]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).logits[0].float()
    positions = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(c_ids), device=device)
    labels = torch.tensor(c_ids, dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(logits[positions], labels, reduction="mean")
    return float(loss.cpu() / math.log(2.0))


def top_bytes(logits: torch.Tensor, k: int = 12) -> list[tuple[int, float]]:
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(int(i) - BYTE_OFFSET, float(v)) for i, v in zip(idx, vals)]


def js_divergence(a_logits: torch.Tensor, b_logits: torch.Tensor) -> float:
    p = torch.softmax(a_logits, dim=-1).clamp_min(1e-30)
    q = torch.softmax(b_logits, dim=-1).clamp_min(1e-30)
    m = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * (p.log() - m.log())) + 0.5 * torch.sum(q * (q.log() - m.log()))
    return float(js / math.log(2.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--ckpt-dir", default="checkpoints/arbor2_1b_8k", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(args.ckpt, args.ckpt_dir)
    cfg = load_checkpoint_config(ckpt)
    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    model = load_inference_model(ckpt, cfg, device)
    patch_size = int(cfg.get("model", {}).get("patch_size", 8))

    prompts = [
        "日本の四季は",
        "日本の首都は",
        "サザエさんは",
        "今日は天気が",
        "def fibonacci(n):\n",
        "Pythonでフィボナッチ数列を計算するには",
    ]
    logits_by_prompt = {}
    print("\n## next-byte top probabilities")
    for prompt in prompts:
        logits = next_logits(model, prompt, device)
        logits_by_prompt[prompt] = logits
        tops = ", ".join(f"{byte_label(b)}:{p:.3f}" for b, p in top_bytes(logits))
        entropy = float(-(torch.softmax(logits, 0) * torch.log2(torch.softmax(logits, 0).clamp_min(1e-30))).sum())
        print(f"{prompt!r} entropy_bits={entropy:.2f} top={tops}")

    print("\n## static patch layout")
    for prompt in ["日本の四季は", "日本の首都は", "サザエさんは", "今日は天気が"]:
        print(f"{prompt!r} bytes={len(prompt.encode('utf-8'))} {patch_layout(prompt, patch_size)}")

    pairs = [
        (
            "same suffix / season prefix",
            "これは日本の季節についての文章です。春夏秋冬の特徴を説明します。日本の四季は",
            "これは料理の記事です。肉や野菜の調理方法を説明します。日本の四季は",
        ),
        (
            "same suffix / capital prefix",
            "これは日本の地理についての文章です。都市と行政の話をします。日本の首都は",
            "これは野球の記事です。試合と選手の話をします。日本の首都は",
        ),
        (
            "same suffix / code prefix",
            "Pythonのプログラム例です。再帰関数を書きます。\ndef fibonacci(n):",
            "料理のレシピ記事です。材料を説明します。\ndef fibonacci(n):",
        ),
        (
            "full prompt vs short suffix",
            "これは日本の季節についての文章です。春夏秋冬の特徴を説明します。日本の四季は",
            "日本の四季は",
        ),
    ]
    print("\n## distribution shifts")
    for name, a, b in pairs:
        la = next_logits(model, a, device)
        lb = next_logits(model, b, device)
        print(f"{name}: js_bits={js_divergence(la, lb):.4f}")
        print("  A top=" + ", ".join(f"{byte_label(x)}:{p:.3f}" for x, p in top_bytes(la, 8)))
        print("  B top=" + ", ".join(f"{byte_label(x)}:{p:.3f}" for x, p in top_bytes(lb, 8)))

    scored = [
        (
            "日本の四季は",
            [
                "、春夏秋冬の四つに分けられ、それぞれに異なる気候や風景がある。",
                "あっという間においしくなった。",
                "日本のコンパクトなものである。",
                "、東京都にあるテレビ番組である。",
            ],
        ),
        (
            "これは日本の季節についての文章です。春夏秋冬の特徴を説明します。日本の四季は",
            [
                "、春夏秋冬の四つに分けられ、それぞれに異なる気候や風景がある。",
                "あっという間においしくなった。",
                "日本のコンパクトなものである。",
                "、東京都にあるテレビ番組である。",
            ],
        ),
        (
            "日本の首都は",
            [
                "東京である。",
                "、春夏秋冬の四つに分けられる。",
                "あっという間においしくなった。",
                "日本のコンパクトなものである。",
            ],
        ),
        (
            "今日は天気が",
            [
                "良いので、外を散歩した。",
                "悪く、雨が降っている。",
                "東京である。",
                "日本のコンパクトなものである。",
            ],
        ),
        (
            "def fibonacci(n):\n",
            [
                "    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
                "日本の四季は、春夏秋冬に分けられる。",
                "\tif(f_sequence())\n\t{\n\t\treturn 0;\n\t}\n",
            ],
        ),
    ]
    print("\n## continuation scores lower_is_better_bpb")
    for prompt, continuations in scored:
        print(f"prompt={prompt!r}")
        values = []
        for text in continuations:
            values.append((continuation_bpb(model, prompt, text, device), text))
        for bpb, text in sorted(values, key=lambda x: x[0]):
            print(f"  {bpb:.3f}  {text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
