"""ArborBLT 用バイトレベル tokenizer (HuggingFace 互換, 自己完結).

token 体系は BLT に合わせる:
    0..3   : <boe> <bos> <eos> <pad> (特殊 ID)
    4..259 : 生バイト 0x00..0xFF (token = byte + 4)

テキストは UTF-8 バイト列に分解されるだけで、語彙学習は存在しない。
"""
from __future__ import annotations

from transformers import PreTrainedTokenizer

_SPECIAL = ["<boe>", "<bos>", "<eos>", "<pad>"]
_OFFSET = len(_SPECIAL)  # 4


class ArborByteTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        self._id_to_token = list(_SPECIAL) + [f"<0x{b:02X}>" for b in range(256)]
        self._token_to_id = {t: i for i, t in enumerate(self._id_to_token)}
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("model_max_length", 2048)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._token_to_id)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return [f"<0x{b:02X}>" for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id["<pad>"])

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < len(self._id_to_token):
            return self._id_to_token[index]
        return "<pad>"

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        buf = bytearray()
        for t in tokens:
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                buf.append(int(t[3:5], 16))
        return buf.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        # 語彙はコードに埋め込み済みでファイル化するものが無い
        return ()
