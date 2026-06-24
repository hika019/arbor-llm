from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_URL_RE = re.compile(r"https?://\S+|www\.\S+|[A-Za-z0-9_.-]+\.(?:com|jp|net|org|info|biz)\b")

JA_WEB_V1_FILTER: dict[str, Any] = {
    "min_chars": 300,
    "min_ja_char_ratio": 0.60,
    "min_kana_ratio": 0.06,
    "max_url_count": 3,
    "max_url_char_ratio": 0.03,
    "max_ascii_ratio": 0.45,
    "max_digit_symbol_ratio": 0.35,
    "max_repeated_line_ratio": 0.30,
    "max_short_line_ratio": 0.55,
    "min_sentence_end_count": 3,
    "drop_if_boilerplate_hits_at_least": 3,
    "drop_if_html_js_hits_at_least": 2,
    "max_suspicious_sequence_count": 1,
    "short_line_chars": 12,
    "boilerplate_terms": [
        "JavaScript",
        "Cookie",
        "クッキー",
        "プライバシー",
        "利用規約",
        "ログイン",
        "会員登録",
        "お問い合わせ",
        "サイトマップ",
        "Copyright",
        "All rights reserved",
        "関連記事",
        "続きを読む",
        "前の記事",
        "次の記事",
        "シェア",
        "ツイート",
        "ランキング",
        "外部リンク",
        "予約",
        "カート",
        "レビューを書く",
    ],
    "html_js_terms": [
        "<div",
        "</",
        "&nbsp;",
        "function",
        "window.",
        "document.",
        "var ",
        "const ",
        "onclick",
    ],
}


@dataclass(frozen=True)
class TextFilterResult:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def resolve_text_filter_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    resolved: dict[str, Any] = {}
    preset = config.get("preset")
    if preset:
        if preset != "ja_web_v1":
            raise ValueError(f"unknown text_filter preset: {preset}")
        resolved.update(JA_WEB_V1_FILTER)
    resolved.update({k: v for k, v in config.items() if k != "preset"})
    return resolved


def evaluate_text_filter(text: str, config: dict[str, Any] | None) -> TextFilterResult:
    cfg = resolve_text_filter_config(config)
    if not cfg:
        return TextFilterResult(True, (), {})

    compact = "".join(ch for ch in text if not ch.isspace())
    total = len(compact)
    hira = kata = kanji = ascii_count = digit_symbol = sentence_end = 0
    for ch in compact:
        code = ord(ch)
        if 0x3040 <= code <= 0x309F:
            hira += 1
        elif 0x30A0 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F:
            kata += 1
        elif 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
            kanji += 1
        if code < 128:
            ascii_count += 1
        category = unicodedata.category(ch)
        if ch.isdigit() or category.startswith("P") or category.startswith("S"):
            digit_symbol += 1
        if ch in "。！？!?":
            sentence_end += 1

    urls = _URL_RE.findall(text)
    url_chars = sum(len(url) for url in urls)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_count = len(lines)
    short_limit = int(cfg.get("short_line_chars", 12))
    short_lines = sum(1 for line in lines if len(line) <= short_limit)
    repeated_lines = line_count - len(set(lines))
    lower_text = text.lower()
    boilerplate_hits = sum(1 for term in cfg.get("boilerplate_terms", []) if term.lower() in lower_text)
    html_js_hits = sum(1 for term in cfg.get("html_js_terms", []) if term.lower() in lower_text)
    suspicious_sequences = text.count("??") + text.count("？？") + text.count("�") + text.count("□")

    denom = max(total, 1)
    metrics = {
        "chars": float(total),
        "ja_char_ratio": (hira + kata + kanji) / denom,
        "kana_ratio": (hira + kata) / denom,
        "ascii_ratio": ascii_count / denom,
        "digit_symbol_ratio": digit_symbol / denom,
        "url_count": float(len(urls)),
        "url_char_ratio": url_chars / denom,
        "short_line_ratio": short_lines / max(line_count, 1),
        "repeated_line_ratio": repeated_lines / max(line_count, 1),
        "sentence_end_count": float(sentence_end),
        "boilerplate_hits": float(boilerplate_hits),
        "html_js_hits": float(html_js_hits),
        "suspicious_sequence_count": float(suspicious_sequences),
    }

    reasons: list[str] = []
    if total < int(cfg.get("min_chars", 0)):
        reasons.append("too_short")
    if metrics["ja_char_ratio"] < float(cfg.get("min_ja_char_ratio", 0.0)):
        reasons.append("low_ja_ratio")
    if metrics["kana_ratio"] < float(cfg.get("min_kana_ratio", 0.0)):
        reasons.append("low_kana_ratio")
    if metrics["url_count"] > float(cfg.get("max_url_count", float("inf"))):
        reasons.append("too_many_urls")
    if metrics["url_char_ratio"] > float(cfg.get("max_url_char_ratio", float("inf"))):
        reasons.append("too_much_url_text")
    if metrics["ascii_ratio"] > float(cfg.get("max_ascii_ratio", float("inf"))):
        reasons.append("too_much_ascii")
    if metrics["digit_symbol_ratio"] > float(cfg.get("max_digit_symbol_ratio", float("inf"))):
        reasons.append("too_much_digit_symbol")
    if metrics["repeated_line_ratio"] > float(cfg.get("max_repeated_line_ratio", float("inf"))):
        reasons.append("too_many_repeated_lines")
    if metrics["short_line_ratio"] > float(cfg.get("max_short_line_ratio", float("inf"))):
        reasons.append("too_many_short_lines")
    if metrics["sentence_end_count"] < float(cfg.get("min_sentence_end_count", 0)):
        reasons.append("too_few_sentences")
    if boilerplate_hits >= int(cfg.get("drop_if_boilerplate_hits_at_least", 10**9)):
        reasons.append("boilerplate")
    if html_js_hits >= int(cfg.get("drop_if_html_js_hits_at_least", 10**9)):
        reasons.append("html_or_js")
    if suspicious_sequences > int(cfg.get("max_suspicious_sequence_count", 10**9)):
        reasons.append("suspicious_sequences")

    return TextFilterResult(not reasons, tuple(reasons), metrics)
