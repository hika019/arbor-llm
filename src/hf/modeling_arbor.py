"""Arbor v2 (バイトレベル階層 Transformer × BitNet b1.58) の HuggingFace 互換ラッパ.

このファイルはエクスポートされたモデルディレクトリに同梱され、
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` で読み込まれる。

モデル本体は同梱の `arbor_model/` パッケージ (torch のみに依存) を使って構築する。
transformers は trust_remote_code の .py をキャッシュへコピーして import するため、
`__file__` 基準だけでは同梱パッケージを見つけられないことがある。そこで config の
`_name_or_path` / `code_root` も探索候補に入れる (ローカル利用前提)。

注意: KV cache は未実装。generate() は 1 トークンごとに全系列を再フォワードする。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

_ARBOR_FIELDS = (
    "vocab_size", "patch_size", "max_bytes",
    "patching_mode", "min_patch_len", "max_patch_len",
    "entropy_threshold", "entropy_model",
    "hidden_size", "num_heads", "num_kv_heads", "intermediate_size",
    "num_hidden_layers",
    "local_hidden_size", "local_num_heads", "local_num_kv_heads",
    "local_intermediate_size", "num_local_encoder_layers", "num_local_decoder_layers",
    "rope_theta", "norm_eps", "bitnet",
)

_ARBOR_DEFAULTS = dict(
    vocab_size=260, patch_size=4, max_bytes=2048,
    patching_mode="static", min_patch_len=2, max_patch_len=16,
    entropy_threshold=1.5, entropy_model=None,
    hidden_size=2048, num_heads=16, num_kv_heads=4, intermediate_size=5632,
    num_hidden_layers=20,
    local_hidden_size=768, local_num_heads=12, local_num_kv_heads=12,
    local_intermediate_size=2048, num_local_encoder_layers=2,
    num_local_decoder_layers=4,
    rope_theta=500000.0, norm_eps=1e-5, bitnet=True,
)


class ArborConfig(PretrainedConfig):
    model_type = "arbor"

    def __init__(self, code_root: str | None = None, **kwargs):
        for name in _ARBOR_FIELDS:
            setattr(self, name, kwargs.pop(name, _ARBOR_DEFAULTS[name]))
        self.code_root = code_root
        kwargs.setdefault("use_cache", False)
        super().__init__(**kwargs)

    def to_arbor_dict(self) -> dict:
        d = {name: getattr(self, name) for name in _ARBOR_FIELDS}
        d["gradient_checkpointing"] = False
        # entropy_model の重みは model.safetensors に同梱されているので
        # 外部 checkpoint パスからは読まない
        d["entropy_model_ckpt"] = None
        return d


def _resolve_code_root(config: ArborConfig) -> Path:
    candidates: list[Path] = []
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path:
        candidates.append(Path(name_or_path))
    if config.code_root:
        candidates.append(Path(config.code_root))
    candidates.append(Path(__file__).resolve().parent)
    for c in candidates:
        if (c / "arbor_model" / "arbor.py").exists():
            return c.resolve()
    raise ImportError(
        "Arbor: 同梱コード (arbor_model/) が見つかりません。"
        "モデルディレクトリをローカルパスとして from_pretrained するか、"
        f"config.code_root を設定してください。探索先: {[str(c) for c in candidates]}"
    )


class ArborForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ArborConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config: ArborConfig):
        super().__init__(config)
        root = _resolve_code_root(config)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # 注意: transformers の check_imports が `from arbor_model import ...` を
        # pip パッケージ要求と誤認するため importlib で動的に import する
        import importlib

        arbor = importlib.import_module("arbor_model.arbor")
        self.model = arbor.build_arbor(config.to_arbor_dict())
        self.post_init()  # transformers が要求する内部属性 (tied weights 等) を設定

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        logits = self.model(input_ids).logits
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].flatten(0, 1).float()
            shift_labels = labels[:, 1:].flatten()
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # KV cache 無し: 常に全系列を渡す
        return {"input_ids": input_ids}

    def can_generate(self) -> bool:
        return True

    def _reset_rope_buffers(self) -> None:
        """RoPE の cos/sin (非永続バッファ) を再計算する.

        transformers の meta-device 初期化では state dict に無い非永続バッファが
        未初期化のまま残ることがある。ロード完了後に必ず計算し直し、dtype は
        学習時 (model.to(bf16) で揃う) と同じくパラメータに合わせる。
        """
        dtype = next(self.parameters()).dtype
        for m in self.modules():
            if hasattr(m, "cos") and hasattr(m, "reset_parameters"):
                device = m.cos.device
                m.reset_parameters()
                m.cos = m.cos.to(device=device, dtype=dtype)
                m.sin = m.sin.to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._reset_rope_buffers()
        return model
