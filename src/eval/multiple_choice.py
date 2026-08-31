"""多肢選択タスクを byte 尤度で採点する.

各選択肢について continuation のバイト NLL を測り、最小のものを選ぶ。

accuracy は 1B・5択では偶然水準 (20%) に張り付いて差が見えないことがあるため、
連続量の margin も出す:
    margin = (不正解のうち最良の bpb) - (正解の bpb)
正なら「正解の方が尤もらしい」。probes.py の margin と同じ量で、accuracy が
動かなくてもモデル間の差を検出できる。

## 形状を固定する理由 (重要)

BitNet の活性量子化は per-token absmax + round() で、丸め境界をまたぐと出力が
離散的に飛ぶ。そのため cuBLAS がテンソル形状によって別カーネル (別の累積順) を
選ぶだけで、数値誤差 1e-4 が 182 層の BitLinear を通って logits 0.9 まで増幅される。
実測: 同一入力でも batch 1→2 で logits が最大 0.87 変化する (bitnet=false なら 1e-4)。

因果性自体は保たれている (末尾に何を置いても前の位置の logits は 1 bit も変わらない)
ので右 pad は安全。したがって「全 forward を同一形状 (B, W) で回す」ことで
決定性と checkpoint 間の公平性を担保する。選択肢ごと・モデルごとに形状が変わると
比較にノイズが乗る。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.eval.tasks import MCDoc, format_doc
from src.infer.generate import BYTE_OFFSET

PAD_ID = 3


@dataclass
class _Seq:
    """採点対象 1 本。``ids`` 全体を流し、[start, start+n_target) の label を採点する."""

    doc_idx: int
    choice_idx: int
    ids: list[int]
    start: int  # continuation の最初の byte に対応する label 位置
    n_target: int


def _ids(text: str) -> list[int]:
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def _build_seqs(docs: list[MCDoc], prefix: str) -> list[_Seq]:
    seqs: list[_Seq] = []
    for d_i, doc in enumerate(docs):
        for c_i, choice in enumerate(doc.choices):
            context, continuation = format_doc(doc, choice)
            ctx_ids = _ids(prefix + context)
            tgt_ids = _ids(continuation)
            if not tgt_ids:
                raise ValueError(f"empty choice in doc {d_i}")
            seqs.append(
                _Seq(
                    doc_idx=d_i,
                    choice_idx=c_i,
                    ids=ctx_ids + tgt_ids,
                    start=len(ctx_ids) - 1,
                    n_target=len(tgt_ids),
                )
            )
    return seqs


def _fixed_width(seqs: list[_Seq], patch_size: int) -> int:
    """全 seq を通す固定幅 W (= x の系列長)。patch_size の倍数に切り上げる.

    W を patch_size の倍数にしておくと model 側の内部 pad が 0 になり、
    patch 数 k = W / patch_size が全 forward で一定になる。
    """
    longest = max(len(s.ids) for s in seqs) - 1
    return math.ceil(longest / patch_size) * patch_size


@torch.inference_mode()
def _score_fixed(
    model: torch.nn.Module,
    batch: list[_Seq],
    width: int,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[float]:
    """常に (batch_size, width) の形状で流し、各 seq の continuation 合計 NLL を返す.

    batch が batch_size に満たない場合はダミー行で埋める (結果は捨てる)。
    """
    x = torch.full((batch_size, width), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, width), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        inp = s.ids[:-1]
        x[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        tgt = s.ids[s.start + 1 : s.start + 1 + s.n_target]
        labels[i, s.start : s.start + s.n_target] = torch.tensor(tgt, dtype=torch.long)
    x = x.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    with ctx:
        logits = model(x).logits
    logits = logits[:, :width].float()

    losses = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none"
    ).view(batch_size, width)
    return losses.sum(dim=1)[: len(batch)].cpu().tolist()


def _score_all(
    model: torch.nn.Module,
    seqs: list[_Seq],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    patch_size: int,
) -> dict[tuple[int, int], float]:
    width = _fixed_width(seqs, patch_size)
    out: dict[tuple[int, int], float] = {}
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        nlls = _score_fixed(model, chunk, width, batch_size, device=device, dtype=dtype)
        for s, nll in zip(chunk, nlls):
            out[(s.doc_idx, s.choice_idx)] = nll
    return out


def _metrics(docs: list[MCDoc], total_nll: dict, n_bytes: dict) -> dict[str, float]:
    n_correct = n_correct_norm = 0
    margins: list[float] = []
    margins_mean: list[float] = []
    gold_bpbs: list[float] = []
    for d_i, doc in enumerate(docs):
        sums = [total_nll[(d_i, c)] for c in range(len(doc.choices))]
        bpb = [
            total_nll[(d_i, c)] / n_bytes[(d_i, c)] / math.log(2.0)
            for c in range(len(doc.choices))
        ]
        if min(range(len(sums)), key=lambda c: sums[c]) == doc.label:
            n_correct += 1
        if min(range(len(bpb)), key=lambda c: bpb[c]) == doc.label:
            n_correct_norm += 1
        gold = bpb[doc.label]
        bad = [b for c, b in enumerate(bpb) if c != doc.label]
        # margin: 最良の不正解との差。ランダムなモデルでも負になる (4本の min と比べるため)
        margins.append(min(bad) - gold)
        # margin_mean: 不正解の平均との差。偶然水準がちょうど 0 で、正なら正解を好んでいる
        margins_mean.append(sum(bad) / len(bad) - gold)
        gold_bpbs.append(gold)

    n = len(docs)
    return {
        "n": n,
        "acc": n_correct / n,
        "acc_norm": n_correct_norm / n,
        "margin": sum(margins) / n,
        "margin_mean": sum(margins_mean) / n,
        "gold_bpb": sum(gold_bpbs) / n,
    }


@torch.inference_mode()
def evaluate_mc(
    model: torch.nn.Module,
    docs: list[MCDoc],
    prefix: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 10,
    patch_size: int = 8,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        seqs = _build_seqs(docs, prefix)
        total_nll = _score_all(model, seqs, device=device, dtype=dtype,
                               batch_size=batch_size, patch_size=patch_size)
        n_bytes = {(s.doc_idx, s.choice_idx): s.n_target for s in seqs}
        return _metrics(docs, total_nll, n_bytes)
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def selftest(
    model: torch.nn.Module,
    docs: list[MCDoc],
    prefix: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    patch_size: int,
) -> dict[str, float]:
    """harness が依存する不変性を実測する.

    - order: seq を並べ替えてバッチ構成を変えても採点値が変わらないこと
      (形状固定なら変わらないはず。これが崩れると比較が成立しない)
    - shape: batch_size を変えると採点値がどれだけ動くか
      = このモデル固有の「形状ノイズ」。checkpoint 間の差がこれ未満なら判定不能。
    """
    model.eval()
    seqs = _build_seqs(docs, prefix)
    base = _score_all(model, seqs, device=device, dtype=dtype,
                      batch_size=batch_size, patch_size=patch_size)

    shuffled = list(reversed(seqs))
    perm = _score_all(model, shuffled, device=device, dtype=dtype,
                      batch_size=batch_size, patch_size=patch_size)
    order_gap = max(abs(base[k] - perm[k]) for k in base)

    other_b = batch_size + 1
    alt = _score_all(model, seqs, device=device, dtype=dtype,
                     batch_size=other_b, patch_size=patch_size)
    shape_gap = max(abs(base[k] - alt[k]) for k in base)

    n_bytes = {(s.doc_idx, s.choice_idx): s.n_target for s in seqs}
    m_base = _metrics(docs, base, n_bytes)
    m_alt = _metrics(docs, alt, n_bytes)
    return {
        "order_gap_nats": order_gap,
        "shape_gap_nats": shape_gap,
        "margin_at_b": m_base["margin"],
        "margin_at_b_plus_1": m_alt["margin"],
        "margin_shape_noise": abs(m_base["margin"] - m_alt["margin"]),
        "acc_at_b": m_base["acc"],
        "acc_at_b_plus_1": m_alt["acc"],
    }
