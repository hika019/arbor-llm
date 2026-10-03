"""ByteLM checkpoint のエントロピー境界を可視化する (CPU で動く読み取り専用ツール).

使い方:
    python scripts/show_entropy_patches.py                          # 内蔵サンプル文
    python scripts/show_entropy_patches.py --text "任意のテキスト"
    python scripts/show_entropy_patches.py --ckpt step_0000054000 --target-avg-len 6

同じエントロピーから entropy (ByteLM の値そのまま) と entropy_char (文字単位のエントロピーが
1 つ前の文字より上がった文字の先頭で区切る) の 2 通りの境界を並べて出す。2 つは判定値の尺度が違うので、閾値はモードごとに
全サンプルの平均 patch 長が --target-avg-len になるよう二分探索する。数文だけでは偏るので、
閾値ごとの実データでの区切り方は scripts/show_entropy_threshold_stats.py で見て、ここでは
--entropy-threshold / --entropy-char-threshold で固定して見るのが正確。
表示: `|` = 文字の頭に揃った境界、`¦` = その文字の途中 (UTF-8 マルチバイトの中) に落ちた境界。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.arbor import BYTE_OFFSET, CHAR_REST_HEAD_FILE, build_byte_lm, compute_patch_starts  # noqa: E402

DEFAULT_TEXTS = [
    "政府は27日、来年度予算の概算要求を取りまとめた。一般会計の総額は過去最大の115兆円程度となる見通しで、"
    "防衛費や社会保障費の増加が主な要因だ。財務省は今後、各省庁との調整を進める。",
    "えー、それってマジ？笑 明日の会議って何時からだっけ？10時？ありがとう！助かる〜",
    "The committee approved the proposal on Tuesday, citing strong public support and a projected "
    "reduction in operating costs over the next five years.",
    "def moving_average(values, window):\n    if window <= 0:\n        raise ValueError(\"window must be positive\")\n"
    "    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]\n",
    "Let $f(x) = x^2 + 3x - 4$. Then $f'(x) = 2x + 3$, so the minimum occurs at $x = -\\frac{3}{2}$.",
]
MODES = ("entropy", "entropy_char")


def render(text: str, starts: torch.Tensor) -> tuple[str, int, int]:
    idx = {i for i in range(1, starts.numel()) if starts[i]}
    out, pos, mid_total = [], 0, 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        if pos in idx:
            out.append("|")
        mid = sum(1 for j in range(pos + 1, pos + n) if j in idx)
        mid_total += mid
        out.append(ch + "¦" * mid)
        pos += n
    return "".join(out).replace("\n", "⏎"), len(idx) + 1, mid_total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="latest", help="'latest' | 'best' | step dir 名")
    p.add_argument("--ckpt-dir", default="./checkpoints/entropy_lm", type=Path)
    p.add_argument("--entropy-threshold", default=None, type=float, help="entropy の固定閾値 (nats)")
    p.add_argument("--entropy-char-threshold", default=None, type=float,
                   help="entropy_char の固定閾値 (1 つ前の文字からの上昇幅、nats)")
    p.add_argument("--target-avg-len", default=6.0, type=float, help="閾値を合わせる平均 patch 長 (byte)")
    p.add_argument("--min-patch-len", default=1, type=int)
    p.add_argument("--max-patch-len", default=None, type=int, help="既定は上限なし")
    p.add_argument("--text", action="append", default=None, help="複数指定可")
    p.add_argument("--char-rest", action="store_true",
                   help="entropy_char の H に char_rest head (3 byte 目以降の推定) を足す")
    args = p.parse_args()

    from safetensors.torch import load_file

    ckpt = (args.ckpt_dir / args.ckpt).resolve()
    cfg = yaml.safe_load((ckpt / "config.yaml").read_text())["model"]
    cfg["char_rest_head"] = args.char_rest
    model = build_byte_lm(cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in load_file(str(ckpt / "model.safetensors")).items()}
    if args.char_rest:
        state.update({f"char_rest.{k}": v for k, v in load_file(str(ckpt / CHAR_REST_HEAD_FILE)).items()})
    model.load_state_dict(state, strict=True)
    model = model.float().eval()

    texts = args.text or DEFAULT_TEXTS
    samples = []
    with torch.no_grad():
        for text in texts:
            ids = torch.tensor([[b + BYTE_OFFSET for b in text.encode("utf-8")]])
            samples.append((text, ids, model.boundary_entropy(ids)))

    def starts_for(mode: str, thr: float, ids, ent) -> torch.Tensor:
        return compute_patch_starts(ids, mode, args.min_patch_len, args.max_patch_len,
                                    entropy_values=ent[0], rest_values=ent[1], threshold=thr)[0]

    def threshold_for(mode: str) -> float:
        fixed = args.entropy_threshold if mode == "entropy" else args.entropy_char_threshold
        if fixed is not None:
            return fixed
        total = sum(ids.numel() for _, ids, _ in samples)
        lo, hi = -20.0, 20.0
        for _ in range(30):
            mid = (lo + hi) / 2
            n = sum(int(starts_for(mode, mid, ids, ent).sum()) for _, ids, ent in samples)
            lo, hi = (mid, hi) if total / n < args.target_avg_len else (lo, mid)
        return (lo + hi) / 2

    thresholds = {mode: threshold_for(mode) for mode in MODES}
    print(f"[show] ckpt={ckpt} min/max_patch_len={args.min_patch_len}/{args.max_patch_len} "
          + " ".join(f"{m}_threshold={t:.3f}" for m, t in thresholds.items()))
    for text, ids, ent in samples:
        print()
        for mode in MODES:
            shown, n_patches, mid = render(text, starts_for(mode, thresholds[mode], ids, ent))
            print(f"[{mode:12s}] patches={n_patches} avg_len={ids.numel() / n_patches:.1f}B 文字内境界={mid}")
            print("  " + shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
