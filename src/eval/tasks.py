"""下流タスク (多肢選択) の定義とプロンプト整形.

bpb は「その分布の圧縮率」であって「知識が正確か」ではない。CPT で狙った
「流暢だが事実誤りを減らす」効果は val bpb には現れないため、正解/不正解を
モデル自身の尤度で選ばせる多肢選択タスクで測る。

タスクを足すときは MCTask を 1 つ書いて TASKS に登録する。
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset


@dataclass(frozen=True)
class MCDoc:
    """多肢選択 1 問。``choices`` は本文のみ (プロンプト接頭辞を含めない)."""

    question: str
    choices: list[str]
    label: int
    # 英語系タスク (lm-eval 準拠): 全選択肢共通の context。None なら JCQA 形式で整形する
    context: str | None = None
    # Winograd 系: 選択肢ごとに context が変わり continuation (choices) は共通
    contexts: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MCTask:
    name: str
    load: Callable[[str], list[MCDoc]]
    # few-shot 例を引く split (通常は train)。評価 split と重複させない。
    fewshot_split: str = "train"
    eval_split: str = "validation"
    num_fewshot: int = 0


def _jcommonsenseqa(split: str) -> list[MCDoc]:
    ds = load_dataset("sbintuitions/JCommonsenseQA", split=split)
    docs = []
    for row in ds:
        choices = [str(row[f"choice{i}"]) for i in range(5)]
        docs.append(
            MCDoc(question=str(row["question"]), choices=choices, label=int(row["label"]))
        )
    return docs


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hellaswag(split: str) -> list[MCDoc]:
    docs = []
    for row in load_dataset("Rowan/hellaswag", split=split):
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        context = _hellaswag_clean(row["activity_label"] + ": " + ctx)
        docs.append(MCDoc(
            question="", context=context, label=int(row["label"]),
            choices=[" " + _hellaswag_clean(e) for e in row["endings"]],
        ))
    return docs


def _arc(config: str) -> Callable[[str], list[MCDoc]]:
    def load(split: str) -> list[MCDoc]:
        docs = []
        for row in load_dataset("allenai/ai2_arc", config, split=split):
            labels = list(row["choices"]["label"])
            docs.append(MCDoc(
                question=row["question"], context=f"Question: {row['question']}\nAnswer:",
                choices=[" " + t for t in row["choices"]["text"]],
                label=labels.index(row["answerKey"]),
            ))
        return docs
    return load


def _openbookqa(split: str) -> list[MCDoc]:
    docs = []
    for row in load_dataset("allenai/openbookqa", "main", split=split):
        labels = list(row["choices"]["label"])
        docs.append(MCDoc(
            question=row["question_stem"], context=row["question_stem"],
            choices=[" " + t for t in row["choices"]["text"]],
            label=labels.index(row["answerKey"].strip()),
        ))
    return docs


def _sciq(split: str) -> list[MCDoc]:
    docs = []
    for row in load_dataset("allenai/sciq", split=split):
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        docs.append(MCDoc(
            question=row["question"],
            context=f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:".lstrip(),
            choices=[" " + c for c in choices], label=3,
        ))
    return docs


def _winograd(path: str, config: str, space: bool) -> Callable[[str], list[MCDoc]]:
    """空欄 "_" に選択肢を入れた前半を context、共通の後半を continuation として採点 (lm-eval 準拠)."""
    def load(split: str) -> list[MCDoc]:
        docs = []
        for row in load_dataset(path, config, split=split):
            sent = row["sentence"]
            idx = sent.index("_")
            rest = sent[idx + 1:].strip()
            cont = (" " + rest) if space else rest
            options = (row["option1"], row["option2"])
            docs.append(MCDoc(
                question=sent, contexts=tuple(sent[:idx] + o for o in options),
                choices=[cont, cont], label=int(row["answer"]) - 1,
            ))
        return docs
    return load


TASKS: dict[str, MCTask] = {
    "jcommonsenseqa": MCTask(name="jcommonsenseqa", load=_jcommonsenseqa, num_fewshot=3),
    "xwinograd_jp": MCTask(name="xwinograd_jp", load=_winograd("Muennighoff/xwinograd", "jp", space=False),
                           eval_split="test"),
    "hellaswag": MCTask(name="hellaswag", load=_hellaswag),
    "arc_easy": MCTask(name="arc_easy", load=_arc("ARC-Easy"), eval_split="test"),
    "arc_challenge": MCTask(name="arc_challenge", load=_arc("ARC-Challenge"), eval_split="test"),
    "openbookqa": MCTask(name="openbookqa", load=_openbookqa, eval_split="test"),
    "sciq": MCTask(name="sciq", load=_sciq, eval_split="test"),
    "winogrande": MCTask(name="winogrande",
                         load=_winograd("allenai/winogrande", "winogrande_xl", space=True)),
}


def format_doc(doc: MCDoc, choice_idx: int) -> tuple[str, str]:
    """(context, continuation) を返す。continuation の bytes だけを採点する."""
    if doc.contexts is not None:
        return doc.contexts[choice_idx], doc.choices[choice_idx]
    if doc.context is not None:
        return doc.context, doc.choices[choice_idx]
    listed = "、".join(f"{i}.{c}" for i, c in enumerate(doc.choices))
    context = f"質問:{doc.question}\n選択肢:{listed}\n回答:"
    return context, doc.choices[choice_idx]


def format_fewshot_prefix(docs: list[MCDoc]) -> str:
    """few-shot 例 (正解つき) を連結した接頭辞。空リストなら空文字."""
    blocks = []
    for doc in docs:
        context, answer = format_doc(doc, doc.label)
        blocks.append(context + answer)
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


def sample_fewshot(docs: list[MCDoc], k: int, seed: int) -> list[MCDoc]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(docs, min(k, len(docs)))
