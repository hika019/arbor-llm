"""Local Encoder (byte → patch).

BLT 標準の Local Encoder をそのまま FP で使用する想定。BitLinear 化は
行わない。実体は third_party/blt 取り込み後にここから import / 再エクスポート。
"""
from __future__ import annotations

# TODO: from bytelatent.local_models import LocalEncoder
