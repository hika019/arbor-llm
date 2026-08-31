"""下流タスク (多肢選択) の定義とプロンプト整形.

bpb は「その分布の圧縮率」であって「知識が正確か」ではない。CPT で狙った
「流暢だが事実誤りを減らす」効果は val bpb には現れないため、正解/不正解を
モデル自身の尤度で選ばせる多肢選択タスクで測る。

タスクを足すときは MCTask を 1 つ書いて TASKS に登録する。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset


@dataclass(frozen=True)
class MCDoc:
    """多肢選択 1 問。``choices`` は本文のみ (プロンプト接頭辞を含めない)."""

    question: str
    choices: list[str]
    label: int


@dataclass(frozen=True)
class MCTask:
    name: str
    load: Callable[[str], list[MCDoc]]
    # few-shot 例を引く split (通常は train)。評価 split と重複させない。
    fewshot_split: str = "train"
    eval_split: str = "validation"


def _jcommonsenseqa(split: str) -> list[MCDoc]:
    ds = load_dataset("sbintuitions/JCommonsenseQA", split=split)
    docs = []
    for row in ds:
        choices = [str(row[f"choice{i}"]) for i in range(5)]
        docs.append(
            MCDoc(question=str(row["question"]), choices=choices, label=int(row["label"]))
        )
    return docs


TASKS: dict[str, MCTask] = {
    "jcommonsenseqa": MCTask(name="jcommonsenseqa", load=_jcommonsenseqa),
}


def format_doc(doc: MCDoc, choice: str) -> tuple[str, str]:
    """(context, continuation) を返す。continuation の bytes だけを採点する."""
    listed = "、".join(f"{i}.{c}" for i, c in enumerate(doc.choices))
    context = f"質問:{doc.question}\n選択肢:{listed}\n回答:"
    return context, choice


def format_fewshot_prefix(docs: list[MCDoc]) -> str:
    """few-shot 例 (正解つき) を連結した接頭辞。空リストなら空文字."""
    blocks = []
    for doc in docs:
        context, answer = format_doc(doc, doc.choices[doc.label])
        blocks.append(context + answer)
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


def sample_fewshot(docs: list[MCDoc], k: int, seed: int) -> list[MCDoc]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(docs, min(k, len(docs)))
