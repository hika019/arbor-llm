"""答えを直接当てさせるタスク (cloze / 生成型) を teacher forcing 1 回で採点する.

多肢選択は 1B では偶然水準に張り付きやすい。こちらは「context の後に target を
greedy で出せるか」を測る: target の各 byte で argmax が正解なら greedy 生成が
target と完全一致するのと同値なので、生成ループ無しで 1 forward で判定できる。

- acc: target 全 byte の argmax が正解だった割合 (= greedy 完全一致率)
- target_bpb: target の bits/byte (連続量。acc が 0 に張り付いても差が見える)

形状は multiple_choice.py と同じ理由で全 forward を (batch_size, W) に固定する。

合成タスク (kv_recall_* / copy / add2) は seed 固定で生成するので checkpoint 間で
同一問題になる。kv_recall は答えまでの距離を変えて文脈利用 (長距離コピー) を測る。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from src.eval.multiple_choice import PAD_ID, _ids


@dataclass(frozen=True)
class ClozeDoc:
    context: str
    target: str


@dataclass(frozen=True)
class ClozeTask:
    name: str
    load: Callable[[], list[ClozeDoc]]


# ---------------------------------------------------------------- tasks
def _lambada() -> list[ClozeDoc]:
    from datasets import load_dataset

    docs = []
    for row in load_dataset("EleutherAI/lambada_openai", "default", split="test"):
        text = row["text"]
        head, last = text.rsplit(" ", 1)
        docs.append(ClozeDoc(context=head, target=" " + last))
    return docs


_FAMILY = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
           "吉田", "山田", "松本", "井上", "木村", "清水", "山崎", "森", "池田", "橋本",
           "阿部", "石川", "山下", "中島", "前田", "藤田", "小川", "後藤", "岡田", "長谷川"]


def _kv_recall(n_pairs: int, n_docs: int = 300, seed: int = 0) -> Callable[[], list[ClozeDoc]]:
    """「X さんの部屋番号は NNN 号室です。」を n_pairs 文並べ、先頭付近の 1 人を同じ言い回しで問う.

    答えは文脈中の同じフレーズの続き (induction コピー)。n_pairs で答えまでの距離を変える。
    """
    def load() -> list[ClozeDoc]:
        rng = random.Random(seed)
        docs = []
        for _ in range(n_docs):
            people = [f"{rng.choice(_FAMILY)}{i}号" for i in range(n_pairs)]
            rng.shuffle(people)
            rooms = [str(rng.randrange(101, 999)) for _ in people]
            facts = "".join(f"{p}さんの部屋番号は{r}号室です。" for p, r in zip(people, rooms))
            q = rng.randrange(max(1, n_pairs // 10))  # 先頭 10% から問う = 遠い
            docs.append(ClozeDoc(context=f"{facts}\n確認します。{people[q]}さんの部屋番号は",
                                 target=f"{rooms[q]}号室"))
        return docs
    return load


def _repeat(source: str, far: bool, n_docs: int = 300, seed: int = 3) -> Callable[[], list[ClozeDoc]]:
    """自然文の段落を一度見せ、もう一度書き始めたところで続き 32 byte を当てさせる (逐語コピー).

    far=True では間に別の段落を ~4k byte 挟む (SSD は 4k 超でコピーが消えた: 1B 20k A/B)。
    """
    def load() -> list[ClozeDoc]:
        paras = _passages(source)
        rng = random.Random(seed)
        docs = []
        for i in range(n_docs):
            p = paras[i % len(paras)]
            raw = p.encode("utf-8")
            cut = len(raw) // 2
            while cut < len(raw) and (raw[cut] & 0xC0) == 0x80:  # UTF-8 境界に合わせる
                cut += 1
            head = raw[:cut].decode("utf-8")
            tail = raw[cut:cut + 32].decode("utf-8", errors="ignore")  # 末尾の欠けた文字を落とす
            filler = ""
            if far:
                others = []
                while sum(len(o.encode()) for o in others) < 4000:
                    others.append(paras[rng.randrange(len(paras))])
                filler = "\n\n".join(others) + "\n\n"
            docs.append(ClozeDoc(context=f"{p}\n\n{filler}{head}", target=tail))
        return docs
    return load


def _passages(source: str) -> list[str]:
    """反復テスト用の自然文 (300〜1200 byte)。en = LAMBADA、ja = JCommonsenseQA の質問を連結."""
    from datasets import load_dataset

    if source == "en":
        texts = [r["text"] for r in load_dataset("EleutherAI/lambada_openai", "default", split="test")]
        return [t for t in texts if 300 <= len(t.encode()) <= 1200][:300]
    qs = [r["question"] for r in load_dataset("sbintuitions/JCommonsenseQA", split="validation")]
    return ["".join(qs[i:i + 12]) for i in range(0, len(qs) - 12, 12)][:300]


def _copy(n_docs: int = 300, length: int = 8, seed: int = 1) -> list[ClozeDoc]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    docs = []
    for _ in range(n_docs):
        s = "".join(rng.choice(alphabet) for _ in range(length))
        demo = "".join(rng.choice(alphabet) for _ in range(length))
        docs.append(ClozeDoc(context=f"Code: {demo}\nCopy: {demo}\n\nCode: {s}\nCopy: ", target=s))
    return docs


def _add(lo: int, hi: int, n_docs: int = 300, shots: int = 5, seed: int = 2) -> Callable[[], list[ClozeDoc]]:
    def load() -> list[ClozeDoc]:
        rng = random.Random(seed)

        def pair() -> tuple[int, int]:
            return rng.randrange(lo, hi), rng.randrange(lo, hi)

        docs = []
        for _ in range(n_docs):
            examples = "".join(f"{a}+{b}={a + b}\n" for a, b in (pair() for _ in range(shots)))
            a, b = pair()
            docs.append(ClozeDoc(context=f"{examples}{a}+{b}=", target=f"{a + b}\n"))
        return docs
    return load


# 易しい順。上 4 つが「文脈をそのまま使えるか」、下が知識・計算。
CLOZE_TASKS: dict[str, ClozeTask] = {
    "repeat_ja": ClozeTask("repeat_ja", _repeat("ja", far=False)),
    "repeat_en": ClozeTask("repeat_en", _repeat("en", far=False)),
    "repeat_en_far": ClozeTask("repeat_en_far", _repeat("en", far=True)),   # 間に ~4k byte
    "kv_recall_10": ClozeTask("kv_recall_10", _kv_recall(10)),       # 答えまで ~400 byte
    "kv_recall_100": ClozeTask("kv_recall_100", _kv_recall(100)),    # 答えまで ~4k byte
    "copy": ClozeTask("copy", _copy),
    "add1": ClozeTask("add1", _add(1, 10)),
    "add2": ClozeTask("add2", _add(10, 100)),
    "lambada": ClozeTask("lambada", _lambada),
}


# ---------------------------------------------------------------- scoring
@torch.inference_mode()
def evaluate_cloze(
    model: torch.nn.Module,
    docs: list[ClozeDoc],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    patch_size: int,
) -> dict[str, float]:
    seqs = []
    for d in docs:
        ctx, tgt = _ids(d.context), _ids(d.target)
        seqs.append((ctx + tgt, len(ctx) - 1, len(tgt)))
    width = math.ceil((max(len(ids) for ids, _, _ in seqs) - 1) / patch_size) * patch_size

    use_autocast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    n_correct = 0
    total_nll = 0.0
    total_bytes = 0
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        x = torch.full((batch_size, width), PAD_ID, dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for j, (ids, start, n) in enumerate(chunk):
            inp = ids[:-1]
            x[j, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            labels[j, start : start + n] = torch.tensor(ids[start + 1 : start + 1 + n], dtype=torch.long)
        x, labels = x.to(device), labels.to(device)
        ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
        with ctx:
            logits = model(x).logits
        logits = logits[:, :width].float()
        nll = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none"
        ).view(batch_size, width)
        mask = labels != -100
        hit = (logits.argmax(-1) == labels) | ~mask
        for j, (_, _, n) in enumerate(chunk):
            n_correct += int(hit[j].all())
            total_nll += float(nll[j].sum())
            total_bytes += n
    return {
        "acc": n_correct / len(docs),
        "target_bpb": total_nll / max(total_bytes, 1) / math.log(2),
        "n": len(docs),
    }
