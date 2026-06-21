"""BitLinear — BitNet b1.58 (W1.58 / A8) 公式レシピ準拠.

"The Era of 1-bit LLMs: Training Tips, Code and FAQ" (Microsoft) の実装に従う:

- 重み: per-tensor absmean scale で ternary {-1, 0, +1} に丸める (1.58 bit)。
- 活性: per-token absmax scale で int8 [-128, 127] に丸める。
- STE は detach トリックで実装する:
      x_q = x + (Q(x) - x).detach()
  これにより forward は量子化値を使い、backward は量子化を恒等写像とみなした
  勾配が「量子化後の重み・活性」で計算される (生の重み/活性で勾配を取るのは
  公式レシピと異なる)。
- bias は持たない (BitNet b1.58 2B4T 仕様)。
- 量子化前の正規化 (SubLN) は層構造側 (arbor.py) が担う。BitNet 2B4T と同じく、
  すべての BitLinear の入力は直前に RMSNorm を通る:
      q/k/v <- input_layernorm,  o <- attn_sub_norm,
      gate/up <- post_attention_layernorm,  down <- ffn_sub_norm

学習パスはカスタム autograd や Triton カーネルを使わない。純 PyTorch なので
torch.compile が全体を融合でき、CPU でもそのまま動く。学習は BF16 シャドウ重みが
master。

推論パス (任意): `freeze_for_inference()` を呼ぶと ternary 重みを 2bit/値で pack
した uint8 バッファを事前計算し、以後の eval forward は Triton カーネル
(int8 活性 × packed ternary) で計算する。重み読み出しが bf16 比 1/8 になるため
batch=1 の生成 (メモリ帯域律速) が速くなる。Triton/CUDA が無い環境では
dequantize 済み重みのキャッシュにフォールバックする (毎回の再量子化を省く)。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CUDA スタック依存
    triton = None
    tl = None


def activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-token absmax int8 量子化 (値は dequantize して返す)."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return (x * scale).round().clamp(-128, 127) / scale


def activation_quant_ste(x: torch.Tensor) -> torch.Tensor:
    """activation_quant の detach-STE 版."""
    return x + (activation_quant(x) - x).detach()


def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """per-tensor absmean ternary 量子化 (値は dequantize して返す)."""
    scale = w.abs().mean().clamp_min(eps)
    return (w / scale).round().clamp(-1, 1) * scale


def pack_ternary_weight(w_q: torch.Tensor) -> torch.Tensor:
    """int8 ternary {-1,0,1} を uint8 に 4 値/byte で pack する."""
    if w_q.dtype != torch.int8:
        raise TypeError(f"w_q must be torch.int8, got {w_q.dtype}")
    if w_q.dim() != 2:
        raise ValueError(f"w_q must be 2D, got shape={tuple(w_q.shape)}")
    n, k = w_q.shape
    k_packed = math.ceil(k / 4)
    padded = w_q.new_zeros((n, k_packed * 4))
    padded[:, :k] = w_q
    codes = (padded + 1).to(torch.uint8).view(n, k_packed, 4)
    return (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    ).contiguous()


def unpack_ternary_weight(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """pack_ternary_weight の逆変換 (検証用)."""
    codes = torch.stack(
        (
            w_packed & 0x03,
            (w_packed >> 2) & 0x03,
            (w_packed >> 4) & 0x03,
            (w_packed >> 6) & 0x03,
        ),
        dim=-1,
    )
    return (codes.reshape(w_packed.size(0), -1)[:, :k].to(torch.int8) - 1).contiguous()


if triton is not None:

    @triton.jit
    def _packed_bitlinear_kernel(
        x_ptr, w_ptr, sx_ptr, sw_ptr, y_ptr,
        m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, k_packed: tl.constexpr,
        stride_xm: tl.constexpr, stride_ym: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, k, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :],
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0,
            )
            pack_idxs = k_idxs // 4
            shifts = (k_idxs % 4) * 2
            packed = tl.load(
                w_ptr + offs_n[None, :] * k_packed + pack_idxs[:, None],
                mask=(offs_n[None, :] < n) & (k_idxs[:, None] < k),
                other=1,  # code 1 = ternary 0
            )
            codes = ((packed >> shifts[:, None]) & 3).to(tl.int32)
            w = (codes - 1).to(tl.int8)
            acc += tl.dot(x, w, out_dtype=tl.float32)

        sx = tl.load(sx_ptr + offs_m, mask=offs_m < m, other=0.0).to(tl.float32)
        sw = tl.load(sw_ptr).to(tl.float32)
        y = acc * sx[:, None] * sw
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            y,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


def _packed_linear(
    x_q: torch.Tensor,      # (M, K) int8
    inv_sx: torch.Tensor,   # (M,) float32: 行ごとの 1/activation_scale
    w_packed: torch.Tensor, # (N, ceil(K/4)) uint8
    sw: torch.Tensor,       # () float32: weight scale
    k: int,
    n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x_q.size(0)
    y = torch.empty((m, n), device=x_q.device, dtype=out_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _packed_bitlinear_kernel[grid](
        x_q.contiguous(), w_packed, inv_sx.contiguous(), sw.contiguous(), y,
        m, n, k, w_packed.size(1),
        x_q.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )
    return y


class _CachedBitLinearSTE(torch.autograd.Function):
    """キャッシュ済み量子化重みを使い、元の重みにSTE勾配を返す."""

    @staticmethod
    def forward(
        ctx,
        x_q: torch.Tensor,
        shadow_weight: torch.Tensor,
        cached_weight_q: torch.Tensor,
    ) -> torch.Tensor:
        del shadow_weight
        ctx.input_shape = tuple(x_q.shape)
        ctx.save_for_backward(x_q, cached_weight_q)
        return F.linear(x_q, cached_weight_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_q, weight_q = ctx.saved_tensors
        grad2 = grad_output.reshape(-1, weight_q.size(0))
        x2 = x_q.reshape(-1, weight_q.size(1))
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = grad2.matmul(weight_q).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            grad_w = grad2.transpose(0, 1).matmul(x2)
        return grad_x, grad_w, None


class _CachedBitLinearGroupSTE(torch.autograd.Function):
    """複数の BitLinear 射影を1回のGEMMにまとめ、各重みに勾配を返す."""

    @staticmethod
    def forward(
        ctx,
        x_q: torch.Tensor,
        cached_weight_q: torch.Tensor,
        *shadow_weights: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x_q.shape)
        ctx.out_sizes = tuple(int(weight.size(0)) for weight in shadow_weights)
        ctx.save_for_backward(x_q, cached_weight_q)
        return F.linear(x_q, cached_weight_q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_q, weight_q = ctx.saved_tensors
        grad2 = grad_output.reshape(-1, weight_q.size(0))
        x2 = x_q.reshape(-1, weight_q.size(1))
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad2.matmul(weight_q).reshape(ctx.input_shape)

        needs = ctx.needs_input_grad[2:]
        if any(needs):
            grad_all = grad2.transpose(0, 1).matmul(x2)
            raw = grad_all.split(ctx.out_sizes, dim=0)
            grad_weights = tuple(value if need else None for value, need in zip(raw, needs))
        else:
            grad_weights = tuple(None for _ in ctx.out_sizes)
        return (grad_x, None, *grad_weights)


class BitLinear(nn.Module):
    """nn.Linear (bias 無し) の drop-in 置換. W1.58 / A8 + STE."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if bias:
            raise ValueError("BitLinear is bias-free (BitNet b1.58 spec).")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # 出力側 (wo / down) は builder 側で 1/sqrt(2*n_layers) に再スケールする
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.06, b=0.06)
        self.register_parameter("bias", None)
        # 訓練時の optimizer step 間だけ使う量子化重みキャッシュ。
        self.register_buffer("_train_qweight", None, persistent=False)
        self._train_cache_enabled = False
        # 推論凍結用 (freeze_for_inference 後のみ非 None)
        self._w_packed: torch.Tensor | None = None
        self._w_scale: torch.Tensor | None = None
        self._w_dq: torch.Tensor | None = None  # CPU/Triton 無し環境のフォールバック
        self.register_load_state_dict_post_hook(self._after_load_state_dict)

    # --------------------------------------------------------- training cache
    def _after_load_state_dict(self, module: nn.Module, incompatible_keys: Any) -> None:
        del module, incompatible_keys
        if self._train_cache_enabled:
            self.refresh_training_weight_cache()
        self.unfreeze()

    @property
    def training_weight_cache_enabled(self) -> bool:
        return self._train_cache_enabled and self._train_qweight is not None

    @property
    def training_cache_bytes(self) -> int:
        cache = self._train_qweight
        return 0 if cache is None else cache.numel() * cache.element_size()

    @torch.no_grad()
    def enable_training_weight_cache(self, enabled: bool = True) -> None:
        if not enabled:
            self.disable_training_weight_cache()
            return
        if not self.weight.requires_grad:
            return
        self._train_cache_enabled = True
        if self._train_qweight is None or self._train_qweight.shape != self.weight.shape:
            self._train_qweight = torch.empty_like(
                self.weight, memory_format=torch.contiguous_format
            )
        self.refresh_training_weight_cache()

    @torch.no_grad()
    def disable_training_weight_cache(self) -> None:
        self._train_cache_enabled = False
        self._train_qweight = None

    @torch.no_grad()
    def refresh_training_weight_cache(self) -> None:
        if not self._train_cache_enabled:
            return
        if self._train_qweight is None or self._train_qweight.shape != self.weight.shape:
            self._train_qweight = torch.empty_like(
                self.weight, memory_format=torch.contiguous_format
            )
        self._train_qweight.copy_(weight_quant(self.weight.detach()))

    def forward_prequantized(self, x_q: torch.Tensor) -> torch.Tensor:
        if self.training and self.training_weight_cache_enabled:
            return _CachedBitLinearSTE.apply(x_q, self.weight, self._train_qweight)
        w = self.weight
        w_q = w + (weight_quant(w) - w).detach()
        return F.linear(x_q, w_q)

    # ------------------------------------------------------------ inference
    # 行数がこれ以下なら packed カーネルより cuBLAS GEMV の方が速い
    # (batch=1 生成はカーネル起動オーバーヘッド律速のため)
    _SMALL_M = 16

    def freeze_for_inference(self) -> None:
        """ternary 重みを事前計算して以後の eval forward を高速化する (重みは凍結前提).

        - 小バッチ (M <= _SMALL_M): dequantize 済み重みキャッシュ + cuBLAS。
          活性量子化は fake_quantize 1 カーネルに融合する
        - 大バッチ: packed ternary Triton カーネル (重み読み出し 1/8)
        """
        with torch.no_grad():
            scale = self.weight.abs().mean().clamp_min(1e-5).float()
            w_int = (self.weight.float() / scale).round().clamp(-1, 1).to(torch.int8)
            self._w_dq = (w_int.float() * scale).to(self.weight.dtype)
            if triton is not None and self.weight.is_cuda:
                self._w_packed = pack_ternary_weight(w_int)
            else:
                self._w_packed = None
            self._w_scale = scale

    def unfreeze(self) -> None:
        self._w_packed = None
        self._w_scale = None
        self._w_dq = None

    @property
    def frozen(self) -> bool:
        return self._w_scale is not None

    @staticmethod
    def _quantize_act(x2: torch.Tensor) -> torch.Tensor:
        """per-row absmax int8 量子化 (dequantize 値, 入力と同 dtype).

        注意: torch.fake_quantize_per_channel_affine は 1 カーネルに見えて
        実測 ~335us (素の F.linear の 8 倍) かかるため使わない。
        """
        s = 127.0 / x2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        return ((x2 * s).round().clamp(-128, 127)) / s

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        m = x2.size(0)
        if self._w_packed is not None and m > self._SMALL_M:
            scale = 127.0 / x2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5).float()
            x_q = (x2.float() * scale).round().clamp(-128, 127).to(torch.int8)
            y = _packed_linear(
                x_q, (1.0 / scale).reshape(-1), self._w_packed, self._w_scale,
                self.in_features, self.out_features, torch.float32,
            ).to(x.dtype)
        else:
            y = F.linear(self._quantize_act(x2), self._w_dq)
        return y.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen and not self.training:
            return self._forward_inference(x)
        return self.forward_prequantized(activation_quant_ste(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias=False, train_cache={self.training_weight_cache_enabled}, "
            f"frozen={self.frozen}"
        )


class BitLinearGroup(nn.Module):
    """既存 BitLinear Parameter を保持したまま射影をまとめて計算する補助層."""

    def __init__(self, members: Sequence[BitLinear], kind: str = "group"):
        super().__init__()
        if len(members) < 2:
            raise ValueError("BitLinearGroup requires at least two members")
        if not all(isinstance(member, BitLinear) for member in members):
            raise TypeError("all BitLinearGroup members must be BitLinear")
        in_features = members[0].in_features
        if any(member.in_features != in_features for member in members):
            raise ValueError("all BitLinearGroup members must share in_features")
        self.in_features = in_features
        self.out_splits = tuple(member.out_features for member in members)
        self.out_features = sum(self.out_splits)
        self.kind = str(kind)
        object.__setattr__(self, "_members", tuple(members))
        self.register_buffer("_train_qweight", None, persistent=False)
        self._train_cache_enabled = False

    def members(self) -> tuple[BitLinear, ...]:
        return self._members

    @property
    def matrix_count(self) -> int:
        return len(self._members)

    @property
    def cache_cost_bytes(self) -> int:
        return sum(
            member.weight.numel() * member.weight.element_size()
            for member in self.members()
        )

    @property
    def training_weight_cache_enabled(self) -> bool:
        return self._train_cache_enabled and self._train_qweight is not None

    @property
    def training_cache_bytes(self) -> int:
        cache = self._train_qweight
        return 0 if cache is None else cache.numel() * cache.element_size()

    @torch.no_grad()
    def enable_training_weight_cache(self, enabled: bool = True) -> None:
        if not enabled:
            self.disable_training_weight_cache()
            return
        members = self.members()
        weight = members[0].weight
        self._train_cache_enabled = True
        expected = (self.out_features, self.in_features)
        if self._train_qweight is None or tuple(self._train_qweight.shape) != expected:
            self._train_qweight = torch.empty(
                expected,
                device=weight.device,
                dtype=weight.dtype,
                memory_format=torch.contiguous_format,
            )
        self.refresh_training_weight_cache()

    @torch.no_grad()
    def disable_training_weight_cache(self) -> None:
        self._train_cache_enabled = False
        self._train_qweight = None

    @torch.no_grad()
    def refresh_training_weight_cache(self) -> None:
        if not self._train_cache_enabled:
            return
        if self._train_qweight is None:
            self.enable_training_weight_cache(True)
            return
        offset = 0
        for member in self.members():
            end = offset + member.out_features
            self._train_qweight[offset:end].copy_(weight_quant(member.weight.detach()))
            offset = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        members = self.members()
        x_q = activation_quant_ste(x)
        if self.training and self.training_weight_cache_enabled:
            weights = tuple(member.weight for member in members)
            return _CachedBitLinearGroupSTE.apply(x_q, self._train_qweight, *weights)
        quantized = [
            member.weight + (weight_quant(member.weight) - member.weight).detach()
            for member in members
        ]
        return F.linear(x_q, torch.cat(quantized, dim=0))


def install_arbor_projection_fusions(module: nn.Module) -> dict[str, int]:
    """Arbor の QKV / gate-up 射影を checkpoint 互換のまま融合可能にする."""
    qkv_groups = 0
    gate_up_groups = 0
    for child in list(module.modules()):
        if all(hasattr(child, name) for name in ("wq", "wk", "wv", "wo", "n_heads")):
            if all(isinstance(getattr(child, name), BitLinear) for name in ("wq", "wk", "wv")):
                if not hasattr(child, "_fast_qkv_group"):
                    child._fast_qkv_group = BitLinearGroup(
                        (child.wq, child.wk, child.wv), kind="qkv"
                    )
                qkv_groups += 1
        if all(hasattr(child, name) for name in ("gate", "up", "down", "ffn_sub_norm")):
            if isinstance(child.gate, BitLinear) and isinstance(child.up, BitLinear):
                if not hasattr(child, "_fast_gate_up_group"):
                    child._fast_gate_up_group = BitLinearGroup(
                        (child.gate, child.up), kind="gate_up"
                    )
                gate_up_groups += 1
    return {"qkv_groups": qkv_groups, "gate_up_groups": gate_up_groups}


def _parse_cache_mode(
    value: bool | str | None,
    *,
    grad_accum_steps: int,
    has_cuda: bool,
) -> str:
    if value is None:
        value = "auto"
    if isinstance(value, bool):
        return "full" if value else "off"
    normalized = str(value).lower().replace("-", "_")
    if normalized == "auto":
        return "full" if has_cuda and grad_accum_steps > 1 else "off"
    if normalized in {"1", "true", "yes", "on", "full", "all"}:
        return "full"
    if normalized in {"fused", "groups", "projection_groups"}:
        return "fused"
    if normalized in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    raise ValueError(f"unknown bitnet_weight_cache setting: {value!r}")


@torch.no_grad()
def configure_bitlinear_training_cache(
    module: nn.Module,
    *,
    enabled: bool | str | None = "auto",
    grad_accum_steps: int = 1,
    max_cache_gib: float | None = 1.25,
    min_numel: int = 0,
) -> dict[str, int | float | bool | str]:
    """torch.compile 前に射影融合を取り付け、訓練用量子化重みcacheを選ぶ."""
    fusion_info = install_arbor_projection_fusions(module)
    all_modules = list(module.modules())
    groups = [child for child in all_modules if isinstance(child, BitLinearGroup)]
    linears = [
        child
        for child in all_modules
        if isinstance(child, BitLinear)
        and child.weight.requires_grad
        and child.weight.numel() >= int(min_numel)
    ]
    mode = _parse_cache_mode(
        enabled,
        grad_accum_steps=max(1, int(grad_accum_steps)),
        has_cuda=any(layer.weight.is_cuda for layer in linears),
    )
    for group in groups:
        group.disable_training_weight_cache()
    for layer in linears:
        layer.disable_training_weight_cache()

    budget = None if max_cache_gib is None else max(0, int(float(max_cache_gib) * 2**30))
    used = 0
    selected_groups: list[BitLinearGroup] = []
    selected_linears: list[BitLinear] = []
    covered: set[int] = set()

    def fits(cost: int) -> bool:
        return budget is None or used + cost <= budget

    if mode in {"fused", "full"}:
        groups.sort(
            key=lambda group: (
                (group.matrix_count - 1) / max(group.cache_cost_bytes, 1),
                group.cache_cost_bytes,
            ),
            reverse=True,
        )
        for group in groups:
            if any(member.weight.numel() < int(min_numel) for member in group.members()):
                continue
            cost = group.cache_cost_bytes
            if fits(cost):
                group.enable_training_weight_cache(True)
                selected_groups.append(group)
                used += cost
                covered.update(id(member) for member in group.members())

    if mode == "full":
        for layer in sorted(linears, key=lambda item: item.weight.numel(), reverse=True):
            if id(layer) in covered:
                continue
            cost = layer.weight.numel() * layer.weight.element_size()
            if fits(cost):
                layer.enable_training_weight_cache(True)
                selected_linears.append(layer)
                used += cost

    cached_matrices = sum(group.matrix_count for group in selected_groups) + len(
        selected_linears
    )
    return {
        "enabled": bool(selected_groups or selected_linears),
        "mode": mode,
        "eligible_layers": len(linears),
        "cached_layers": cached_matrices,
        "fused_groups": len(selected_groups),
        "standalone_layers": len(selected_linears),
        "cache_bytes": used,
        "cache_gib": used / 2**30,
        "grad_accum_steps": int(grad_accum_steps),
        **fusion_info,
    }


@torch.no_grad()
def refresh_bitlinear_training_cache(module: nn.Module) -> int:
    """optimizer.step 後に有効な量子化重みcacheを更新する."""
    all_modules = list(module.modules())
    covered: set[int] = set()
    count = 0
    for child in all_modules:
        if isinstance(child, BitLinearGroup) and child.training_weight_cache_enabled:
            child.refresh_training_weight_cache()
            covered.update(id(member) for member in child.members())
            count += child.matrix_count
    for child in all_modules:
        if (
            isinstance(child, BitLinear)
            and id(child) not in covered
            and child.training_weight_cache_enabled
        ):
            child.refresh_training_weight_cache()
            count += 1
    return count


def freeze_bitlinear_for_inference(module: nn.Module) -> int:
    """module 以下の全 BitLinear を推論凍結する。凍結した層数を返す."""
    count = 0
    for child in module.modules():
        if isinstance(child, BitLinear):
            child.freeze_for_inference()
            count += 1
    return count
