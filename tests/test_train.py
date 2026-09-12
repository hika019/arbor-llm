from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from src.train.train import resolve_precision
from src.train.train import resolve_autocast
from src.train.train import resolve_entropy_lm_reference
from src.train.train import adapt_config_for_device
from src.train.train import pick_device
from src.train.train import byte_kind_loss_stats
from src.train.train import build_validation_model
from src.train.train import evaluate_validation
from src.train.train import CudaBatchPrefetcher
from src.train.train import ThreadedBatchPrefetcher
from src.train.train import rebase_scheduler_lr
from src.train.train import should_restore_dataloader_state
from src.train.train import prepare_cudagraph_gradient_buffers
from src.train.train import uses_cudagraph_compile
from src.train.train import parse_args
from src.train.train import _run_tuning_preflight_preserving_state


def test_parse_args_accepts_positive_benchmark_steps(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config), "--benchmark-steps", "120"],
    )

    args = parse_args()

    assert args.config == config
    assert args.benchmark_steps == 120
    assert not args.dry_run


@pytest.mark.parametrize("steps", ["0", "-1"])
def test_parse_args_rejects_nonpositive_benchmark_steps(monkeypatch, tmp_path, steps):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(tmp_path / "config.yaml"), "--benchmark-steps", steps],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_rejects_benchmark_steps_with_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(tmp_path / "config.yaml"),
            "--benchmark-steps",
            "1",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_resolve_precision_accepts_supported_modes():
    assert resolve_precision("bf16") == (torch.bfloat16, True)
    assert resolve_precision("fp16") == (torch.float16, True)
    assert resolve_precision("fp32") == (torch.float32, False)


def test_resolve_precision_rejects_unknown_mode():
    with pytest.raises(ValueError, match="speed.precision"):
        resolve_precision("int4")


def test_resolve_precision_bf8_is_explicit_error_not_silent():
    with pytest.raises(ValueError, match="bf8"):
        resolve_precision("bf8")


def test_resolve_autocast_requires_real_bool():
    assert resolve_autocast({}, True) is True
    assert resolve_autocast({"autocast": False}, True) is False
    with pytest.raises(TypeError, match="speed.autocast"):
        resolve_autocast({"autocast": "false"}, True)


def test_mps_adaptation_preserves_model_optimizer_and_effective_batch():
    cfg = {
        "model": {
            "bitnet": True,
            "patching_mode": "static",
            "hidden_size": 2048,
            "gradient_checkpointing": False,
        },
        "optim": {
            "optimizer": "adamw",
            "state_precision": "int8",
            "lr": 1e-3,
        },
        "speed": {"micro_batch_size": 2, "grad_accum_steps": 32},
        "validation": {"micro_batch_size": 2},
    }

    resolved = adapt_config_for_device(cfg, torch.device("mps"))

    assert resolved["model"]["bitnet"] is True
    assert resolved["model"]["patching_mode"] == "static"
    assert resolved["model"]["hidden_size"] == 2048
    assert resolved["model"]["gradient_checkpointing"] is True
    assert resolved["optim"] == cfg["optim"]
    assert resolved["speed"]["micro_batch_size"] == 1
    assert resolved["speed"]["grad_accum_steps"] == 64
    assert resolved["validation"]["micro_batch_size"] == 1
    assert cfg["model"]["gradient_checkpointing"] is False


def test_cuda_adaptation_does_not_change_config():
    cfg = {
        "model": {"gradient_checkpointing": False},
        "optim": {"optimizer": "adamw", "state_precision": "fp32"},
        "speed": {"micro_batch_size": 2, "grad_accum_steps": 32},
    }
    assert adapt_config_for_device(cfg, torch.device("cuda")) == cfg


def test_attention_auto_is_rejected_instead_of_falling_back():
    cfg = {
        "model": {"global_attn_impl": "auto"},
        "speed": {"torch_compile": True},
    }
    with pytest.raises(ValueError, match="auto/fallback"):
        adapt_config_for_device(cfg, torch.device("cuda"))


def test_flex_without_compile_is_error():
    cfg = {
        "model": {"global_attn_impl": "flex"},
        "speed": {"torch_compile": False},
    }
    with pytest.raises(ValueError, match="torch_compile=true"):
        adapt_config_for_device(cfg, torch.device("cuda"))


def test_flex_on_non_cuda_is_error():
    cfg = {
        "model": {"global_attn_impl": "flex"},
        "speed": {"torch_compile": True},
    }
    with pytest.raises(ValueError, match="CUDA専用"):
        adapt_config_for_device(cfg, torch.device("cpu"))


def test_fp8_on_non_cuda_is_error():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_fp8": "bwd"},
    }
    with pytest.raises(ValueError, match="CUDA専用"):
        adapt_config_for_device(cfg, torch.device("cpu"))


def test_unknown_int8_backend_is_error():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_int8_backend": "cutlass"},
    }
    with pytest.raises(ValueError, match="bitlinear_int8_backend"):
        adapt_config_for_device(cfg, torch.device("cuda"))


def test_unknown_ternary_backend_is_error():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_backend": "multiply_free"},
    }
    with pytest.raises(ValueError, match="bitlinear_ternary_backend"):
        adapt_config_for_device(cfg, torch.device("cuda"))


def test_kmajor_ternary_backend_alias_is_normalized():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_backend": "kmajor"},
    }
    resolved = adapt_config_for_device(cfg, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_backend"] == "kmajor_current"


def test_decode_v2_ternary_backend_alias_is_normalized():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_backend": "decode_v2"},
    }
    resolved = adapt_config_for_device(cfg, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_backend"] == "kmajor_single_dot"


def test_unknown_ternary_tuning_mode_is_error():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_tuning": "guess"},
    }
    with pytest.raises(ValueError, match="bitlinear_ternary_tuning"):
        adapt_config_for_device(cfg, torch.device("cuda"))


def test_fixed_ternary_tuning_requires_and_validates_tile():
    missing = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_tuning": "fixed"},
    }
    with pytest.raises(ValueError, match="requires.*fixed_tile"):
        adapt_config_for_device(missing, torch.device("cuda"))

    invalid = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_tuning": "fixed",
            "bitlinear_ternary_fixed_tile": "64,64",
        },
    }
    with pytest.raises(ValueError, match="invalid.*fixed_tile"):
        adapt_config_for_device(invalid, torch.device("cuda"))

    valid = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_tuning": "fixed",
            "bitlinear_ternary_fixed_tile": "64,64,32,4,3",
        },
    }
    resolved = adapt_config_for_device(valid, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_tuning"] == "fixed"


def test_ternary_execution_path_is_validated_against_tuning_mode():
    invalid = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_execution": "graph_break"},
    }
    with pytest.raises(ValueError, match="bitlinear_ternary_execution"):
        adapt_config_for_device(invalid, torch.device("cuda"))

    raw_auto = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_execution": "raw",
            "bitlinear_ternary_tuning": "auto",
        },
    }
    with pytest.raises(ValueError, match="execution=raw.*tuning=fixed"):
        adapt_config_for_device(raw_auto, torch.device("cuda"))

    raw_fixed = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_execution": "raw",
            "bitlinear_ternary_tuning": "fixed",
            "bitlinear_ternary_fixed_tile": "64,64,32,4,3",
        },
    }
    resolved = adapt_config_for_device(raw_fixed, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_execution"] == "raw"

    legacy_raw = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_execution": "legacy_raw",
            "bitlinear_ternary_tuning": "off",
        },
    }
    resolved = adapt_config_for_device(legacy_raw, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_execution"] == "legacy_raw"

    legacy_raw["speed"]["bitlinear_ternary_tuning"] = False
    resolved = adapt_config_for_device(legacy_raw, torch.device("cuda"))
    assert resolved["speed"]["bitlinear_ternary_tuning"] == "off"

    raw_plan_without_cache = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_execution": "raw_plan",
            "bitlinear_ternary_tuning_cache": False,
        },
    }
    with pytest.raises(ValueError, match="raw_plan requires cache"):
        adapt_config_for_device(raw_plan_without_cache, torch.device("cuda"))

    raw_plan_without_auto = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_execution": "raw_plan",
            "bitlinear_ternary_tuning": "off",
        },
    }
    with pytest.raises(ValueError, match="raw_plan requires.*tuning=auto"):
        adapt_config_for_device(raw_plan_without_auto, torch.device("cuda"))

    wrong_legacy_backend = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_backend": "kmajor_single_dot",
            "bitlinear_ternary_execution": "legacy_custom_op",
        },
    }
    with pytest.raises(ValueError, match="legacy_custom_op requires"):
        adapt_config_for_device(wrong_legacy_backend, torch.device("cuda"))

    wrong_legacy_tuning = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "bitlinear_ternary_backend": "dot_current",
            "bitlinear_ternary_execution": "legacy_raw",
            "bitlinear_ternary_tuning": "fixed",
            "bitlinear_ternary_fixed_tile": "64,64,32,4,3",
        },
    }
    with pytest.raises(ValueError, match="legacy_raw requires.*tuning=off"):
        adapt_config_for_device(wrong_legacy_tuning, torch.device("cuda"))


def test_unknown_ternary_wgrad_backend_is_error():
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {"bitlinear_ternary_wgrad_backend": "bf16"},
    }
    with pytest.raises(ValueError, match="bitlinear_ternary_wgrad_backend"):
        adapt_config_for_device(cfg, torch.device("cuda"))


@pytest.mark.parametrize("compile_mode", ["reduce-overhead", "max-autotune"])
def test_cuda_graph_compile_modes_allow_gradient_accumulation(compile_mode):
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "torch_compile": True,
            "compile_mode": compile_mode,
            "grad_accum_steps": 16,
        },
    }
    assert adapt_config_for_device(cfg, torch.device("cuda")) == cfg
    assert uses_cudagraph_compile(cfg["speed"])


@pytest.mark.parametrize("compile_mode", ["default", "max-autotune-no-cudagraphs"])
def test_non_cudagraph_compile_modes_allow_gradient_accumulation(compile_mode):
    cfg = {
        "model": {"global_attn_impl": "sdpa"},
        "speed": {
            "torch_compile": True,
            "compile_mode": compile_mode,
            "grad_accum_steps": 16,
        },
    }
    assert adapt_config_for_device(cfg, torch.device("cuda")) == cfg
    assert not uses_cudagraph_compile(cfg["speed"])


def test_cudagraph_gradient_buffers_are_persistent_and_reused():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.Linear(3, 2, bias=False),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    count, num_bytes = prepare_cudagraph_gradient_buffers(model, optimizer)
    parameters = list(model.parameters())
    pointers = [parameter.grad.data_ptr() for parameter in parameters]

    assert count == len(parameters)
    assert num_bytes == sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in parameters)

    for parameter in parameters:
        parameter.grad.fill_(1)
    optimizer.zero_grad(set_to_none=False)

    assert [parameter.grad.data_ptr() for parameter in parameters] == pointers
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in parameters)
    assert prepare_cudagraph_gradient_buffers(model, optimizer) == (0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cudagraph_gradient_accumulation_reuses_external_buffers_cuda():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16, bias=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    prepare_cudagraph_gradient_buffers(model, optimizer)
    grad_pointers = [parameter.grad.data_ptr() for parameter in model.parameters()]
    compiled = torch.compile(model, mode="reduce-overhead")

    for _ in range(3):
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            compiled(torch.randn(8, 16, device="cuda")).square().mean().backward()
        torch.cuda.synchronize()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        assert [parameter.grad.data_ptr() for parameter in model.parameters()] == grad_pointers


def test_validation_model_is_eager_and_separate_from_training_wrapper_by_default():
    base = torch.nn.Linear(4, 4)
    train_wrapper = object()

    val_model = build_validation_model(
        base,
        {"enabled": True, "torch_compile": False},
        torch.device("cuda"),
    )

    assert val_model is base
    assert val_model is not train_wrapper


def test_validation_model_uses_independent_compile_mode(monkeypatch):
    base = torch.nn.Linear(4, 4)
    calls = []
    wrapper = torch.nn.Sequential(base)

    def fake_compile(model, *, mode):
        calls.append((model, mode))
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    val_model = build_validation_model(
        base,
        {"enabled": True, "torch_compile": True, "compile_mode": "default"},
        torch.device("cuda"),
    )

    assert val_model is wrapper
    assert calls == [(base, "default")]


def test_validation_model_rejects_unknown_compile_mode():
    with pytest.raises(ValueError, match="validation.compile_mode"):
        build_validation_model(
            torch.nn.Linear(4, 4),
            {"enabled": True, "torch_compile": True, "compile_mode": "turbo"},
            torch.device("cuda"),
        )


def test_validation_exception_restores_shared_base_model_training_state():
    class FailingModel(torch.nn.Module):
        def forward(self, input_ids):
            raise RuntimeError("validation failure")

    model = FailingModel().train()
    loaders = {
        "unit": [
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.ones(1, 4, dtype=torch.long),
            }
        ]
    }

    with pytest.raises(RuntimeError, match="validation failure"):
        evaluate_validation(
            model,
            loaders,
            torch.device("cpu"),
            torch.float32,
            False,
            1,
        )

    assert model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_train_eager_validation_compiled_train_cuda():
    """train wrapper→eager validation→同じtrain wrapper復帰のCUDA回帰テスト."""
    from src.model.arbor import ArborConfig, ArborModel
    from src.model.bitlinear import (
        configure_bitlinear_training_cache,
        install_arbor_projection_fusions,
        refresh_bitlinear_training_cache,
        set_bitlinear_fp8_mode,
        set_bitlinear_int8_backend,
    )

    cfg = ArborConfig.from_dict(
        {
            "vocab_size": 260,
            "patch_size": 4,
            "patch_pooling": "mean",
            "max_bytes": 64,
            "hidden_size": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "local_hidden_size": 16,
            "local_num_heads": 2,
            "local_num_kv_heads": 2,
            "local_intermediate_size": 32,
            "num_local_encoder_layers": 1,
            "num_local_decoder_layers": 1,
        }
    )
    base = ArborModel(cfg).to(device="cuda", dtype=torch.bfloat16).train()
    install_arbor_projection_fusions(base)
    set_bitlinear_int8_backend("auto")
    set_bitlinear_fp8_mode(base, "int8")
    configure_bitlinear_training_cache(
        base,
        enabled="full",
        grad_accum_steps=1,
        max_cache_gib=0.1,
        min_numel=0,
    )
    train_model = torch.compile(base, mode="default")
    val_model = build_validation_model(
        base,
        {"enabled": True, "torch_compile": False},
        torch.device("cuda"),
    )
    assert val_model is base
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-4)
    x = torch.randint(4, 260, (1, 64), device="cuda")

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        logits = train_model(x).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        refresh_bitlinear_training_cache(base)
        return loss.detach()

    before = train_step()
    val_batch = {"input_ids": x.cpu(), "labels": x.cpu()}
    results = evaluate_validation(
        val_model,
        {"unit": [val_batch]},
        torch.device("cuda"),
        torch.bfloat16,
        True,
        1,
    )
    after = train_step()

    assert torch.isfinite(before)
    assert torch.isfinite(after)
    assert torch.isfinite(torch.tensor(results["mean_bpb"]))
    assert base.training


def test_entropy_model_config_is_loaded_from_single_reference(tmp_path):
    entropy_path = tmp_path / "entropy_lm.yaml"
    entropy_path.write_text(
        """
model:
  arch: byte_lm
  vocab_size: 260
  hidden_size: 128
checkpoint:
  dir: ./checkpoints/entropy_lm
"""
    )
    arbor_path = tmp_path / "arbor.yaml"
    arbor_path.write_text("")
    cfg = {
        "entropy_lm_config": "entropy_lm.yaml",
        "model": {"patching_mode": "entropy", "bitnet": True},
    }

    resolved = resolve_entropy_lm_reference(cfg, arbor_path)

    assert resolved["model"]["bitnet"] is True
    assert resolved["model"]["entropy_model"] == {
        "vocab_size": 260,
        "hidden_size": 128,
    }
    assert (
        resolved["model"]["entropy_model_ckpt"]
        == "checkpoints/entropy_lm/latest"
    )
    assert "entropy_model" not in cfg["model"]


def test_entropy_inline_model_and_reference_cannot_be_double_managed(tmp_path):
    cfg = {
        "entropy_lm_config": "entropy_lm.yaml",
        "model": {
            "patching_mode": "entropy",
            "entropy_model": {"hidden_size": 128},
        },
    }
    with pytest.raises(ValueError, match="二重管理は禁止"):
        resolve_entropy_lm_reference(cfg, tmp_path / "arbor.yaml")


def test_pick_device_does_not_fallback_from_explicit_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    with pytest.raises(RuntimeError, match="暗黙フォールバック"):
        pick_device("mps")


def test_pick_device_binds_torchrun_process_to_local_rank(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    bound: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", bound.append)

    assert pick_device("cuda") == torch.device("cuda", 1)
    assert bound == [1]


def test_pick_device_rejects_incomplete_torchrun_environment(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="LOCAL_RANK"):
        pick_device("cuda")


def test_tuning_preflight_restores_rng_buffers_grads_and_mode():
    class PreflightModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, input_ids):
            self.counter.add_(1)
            logits = self.linear(input_ids.float().unsqueeze(-1))
            return SimpleNamespace(logits=logits)

    model = PreflightModel().eval()
    model.linear.weight.grad = torch.full_like(model.linear.weight, 7)
    rng_before = torch.get_rng_state().clone()

    metrics = _run_tuning_preflight_preserving_state(
        model, torch.device("cpu"), micro_batch=2, context=3, vocab=10
    )

    assert metrics["peak_allocated_bytes"] == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is False
    assert model.counter.item() == 0
    assert torch.equal(model.linear.weight.grad, torch.full_like(model.linear.weight, 7))
    assert model.linear.bias.grad is None


def test_should_restore_dataloader_state_when_data_config_matches():
    data_cfg = {
        "sources": [{"path": "dataset-a", "weight": 1.0}],
        "context_length": 128,
        "micro_batch_size": 2,
    }

    assert should_restore_dataloader_state(data_cfg, dict(data_cfg))


def test_should_not_restore_dataloader_state_when_data_config_changed():
    saved_data_cfg = {
        "sources": [{"path": "dataset-a", "weight": 1.0}],
        "context_length": 128,
        "micro_batch_size": 2,
    }
    current_data_cfg = {
        "sources": [
            {"path": "dataset-a", "weight": 0.5},
            {"path": "dataset-b", "weight": 0.5},
        ],
        "context_length": 128,
        "micro_batch_size": 2,
    }

    assert not should_restore_dataloader_state(saved_data_cfg, current_data_cfg)


def test_byte_kind_loss_stats_groups_utf8_classes():
    labels = torch.tensor([[ord("A") + 4, 0xE3 + 4, 0x81 + 4, 0xF8 + 4, -100]])
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0])

    stats = byte_kind_loss_stats(losses, labels)

    assert stats["ascii_count"] == 1.0
    assert stats["ascii_loss_sum"] == 1.0
    assert stats["utf8_lead_count"] == 1.0
    assert stats["utf8_lead_loss_sum"] == 2.0
    assert stats["utf8_cont_count"] == 1.0
    assert stats["utf8_cont_loss_sum"] == 3.0
    assert stats["other_count"] == 1.0
    assert stats["other_loss_sum"] == 4.0


def test_rebase_scheduler_lr_preserves_step_but_changes_base_lr():
    p = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.SGD([p], lr=8.0e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5)

    opt.step()
    sched.step()
    old_sched_state = sched.state_dict()
    old_opt_state = opt.state_dict()

    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=2.0e-4)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda step: 0.5)
    opt2.load_state_dict(old_opt_state)
    sched2.load_state_dict(old_sched_state)

    assert sched2.base_lrs == [8.0e-4]
    lrs = rebase_scheduler_lr(opt2, sched2, 2.0e-4)

    assert sched2.last_epoch == old_sched_state["last_epoch"]
    assert sched2.base_lrs == [2.0e-4]
    assert lrs == [1.0e-4]
    assert opt2.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert sched2.get_last_lr() == [1.0e-4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_batch_prefetcher_moves_tensor_batches_to_device():
    batches = iter([
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        },
        {
            "input_ids": torch.full((2, 4), 2, dtype=torch.long),
            "labels": torch.full((2, 4), 3, dtype=torch.long),
        }
    ])
    prefetcher = CudaBatchPrefetcher(batches, torch.device("cuda"))

    batch = next(prefetcher)
    assert batch["input_ids"].device.type == "cuda"
    assert batch["labels"].device.type == "cuda"
    state = prefetcher.state_dict()
    assert state["input_ids"].device.type == "cpu"
    resumed = CudaBatchPrefetcher(iter([]), torch.device("cuda"), initial_batch=state)
    resumed_batch = next(resumed)
    assert resumed_batch["input_ids"].device.type == "cuda"
    assert resumed_batch["input_ids"].cpu().tolist() == [[2, 2, 2, 2], [2, 2, 2, 2]]
    with pytest.raises(StopIteration):
        next(resumed)


class _FakeStatefulLoader:
    """順序付き batch 供給 + state_dict/load_state_dict を持つ streaming loader の代役."""

    def __init__(self, n: int = 100):
        self.n = n
        self.pos = 0

    def __iter__(self):
        while self.pos < self.n:
            item = {"input_ids": torch.tensor([self.pos])}
            self.pos += 1
            yield item

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, state):
        self.pos = state["pos"]


def _drain(iterator, k):
    return [int(next(iterator)["input_ids"][0]) for _ in range(k)]


def test_threaded_batch_prefetcher_preserves_order_and_exhausts():
    loader = _FakeStatefulLoader(n=10)
    pf = ThreadedBatchPrefetcher(loader, depth=3)
    assert _drain(pf, 10) == list(range(10))
    with pytest.raises(StopIteration):
        next(pf)
    pf.close()


def test_threaded_batch_prefetcher_exact_resume_roundtrip():
    loader = _FakeStatefulLoader(n=50)
    pf = ThreadedBatchPrefetcher(loader, depth=4)
    consumed = _drain(pf, 7)
    # checkpoint: loader state と未消費 batch を同一瞬間のペアで取得
    state, pending = pf.state_dict()
    pf.close()
    assert consumed == list(range(7))
    # resume: 新しい loader に state を戻し、pending を先に replay する
    loader2 = _FakeStatefulLoader(n=50)
    loader2.load_state_dict(state)
    pf2 = ThreadedBatchPrefetcher(loader2, depth=4, initial_batches=pending)
    rest = _drain(pf2, 43)
    pf2.close()
    # 欠落も重複もなく元の続き
    assert consumed + rest == list(range(50))


def test_threaded_batch_prefetcher_replays_initial_batches_first():
    loader = _FakeStatefulLoader(n=5)
    loader.pos = 2  # state 上は 0,1 消費済み
    pending = [{"input_ids": torch.tensor([0])}, {"input_ids": torch.tensor([1])}]
    pf = ThreadedBatchPrefetcher(loader, depth=2, initial_batches=pending)
    assert _drain(pf, 5) == [0, 1, 2, 3, 4]
    pf.close()


def test_threaded_batch_prefetcher_propagates_source_exception():
    class _Boom:
        def __iter__(self):
            yield {"input_ids": torch.tensor([0])}
            raise RuntimeError("source failed")

    pf = ThreadedBatchPrefetcher(_Boom(), depth=2)
    assert _drain(pf, 1) == [0]
    with pytest.raises(RuntimeError, match="source failed"):
        next(pf)
    pf.close()


def test_threaded_batch_prefetcher_close_unblocks_full_buffer():
    loader = _FakeStatefulLoader(n=1000)
    pf = ThreadedBatchPrefetcher(loader, depth=2)
    _drain(pf, 3)
    pf.close()  # worker が buffer 満杯待ちでも deadlock しない
    assert not pf._thread.is_alive()
