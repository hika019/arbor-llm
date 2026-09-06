"""Static Arbor checkpoint bottleneck diagnostics.

Measures, on the same held-out validation batches:

- full-model BPB
- module ablations: skip local encoder / zero global context / skip local decoder
- representation RMS through the hierarchy
- linear reconstruction probes from pooled patch representation and global-input patch vector

The checkpoint's own config.yaml is always used, so this remains valid even if
configs/arbor.yaml has changed after the run was trained.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file as safe_load

from src.data.byte_dataset import build_byte_dataloader
from src.model.arbor import ArborModel, build_arbor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--max-batches", type=int, default=2)
    p.add_argument("--probe-steps", type=int, default=100)
    p.add_argument("--probe-lr", type=float, default=3e-3)
    p.add_argument("--probe-max-patches", type=int, default=4096)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default=None,
        help="default: saved speed.precision on CUDA, fp32 on CPU",
    )
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    for name in ("config.yaml", "model.safetensors"):
        if not (path / name).exists():
            raise FileNotFoundError(f"{path / name} is missing")
    return path


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def load_checkpoint(
    checkpoint: Path,
    device: torch.device,
    precision_override: str | None,
) -> tuple[ArborModel, dict, torch.dtype]:
    cfg = yaml.safe_load((checkpoint / "config.yaml").read_text()) or {}
    model_cfg = dict(cfg.get("model", {}))

    if model_cfg.get("arch", "arbor") != "arbor":
        raise ValueError("diagnose_arbor supports model.arch=arbor only")
    if model_cfg.get("patching_mode", "static") != "static":
        raise ValueError("diagnose_arbor currently requires static patching")

    saved_precision = str(cfg.get("speed", {}).get("precision", "bf16"))
    precision_name = precision_override or (
        saved_precision if device.type == "cuda" else "fp32"
    )
    dtype = resolve_dtype(precision_name)

    model = build_arbor(model_cfg).to(device=device, dtype=dtype)
    weights = safe_load(
        str(checkpoint / "model.safetensors"),
        device=str(device),
    )
    if any(k.startswith("_orig_mod.") for k in weights):
        weights = {
            k.removeprefix("_orig_mod."): v
            for k, v in weights.items()
        }
    model.load_state_dict(weights, strict=True)
    model.cfg.gradient_checkpointing = False
    model.eval()
    return model, cfg, dtype


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_validation_cfgs(
    cfg: dict,
    domains: list[str] | None,
) -> dict[str, dict]:
    validation_cfg = cfg.get("validation", {})
    configured = validation_cfg.get("domains", {})
    if not configured:
        raise ValueError("checkpoint config has no validation.domains")

    selected = list(configured) if not domains else domains
    unknown = [name for name in selected if name not in configured]
    if unknown:
        raise ValueError(
            f"unknown validation domains: {unknown}; configured={list(configured)}"
        )

    data_cfg = dict(cfg["data"])
    speed_micro = cfg.get("speed", {}).get("micro_batch_size")
    if speed_micro is not None:
        data_cfg["micro_batch_size"] = int(speed_micro)
    else:
        data_cfg.setdefault("micro_batch_size", 4)
    data_cfg.setdefault("seed", cfg.get("seed", 42))

    model_cfg = cfg.get("model", {})
    if str(model_cfg.get("patching_mode", "static")) == "static":
        data_cfg.setdefault(
            "patch_align",
            int(model_cfg.get("patch_size", 1)),
        )

    val_micro = int(
        validation_cfg.get(
            "micro_batch_size",
            data_cfg["micro_batch_size"],
        )
    )

    result: dict[str, dict] = {}
    for name in selected:
        val = dict(configured[name])
        val.setdefault("context_length", data_cfg["context_length"])

        base_packing = data_cfg.get("packing", "concat")
        if base_packing == "sft":
            base_packing = "document"
        val.setdefault("packing", base_packing)

        val.setdefault("byte_offset", data_cfg.get("byte_offset", 4))
        val.setdefault("eos_token_id", data_cfg.get("eos_token_id", 2))
        val.setdefault("pad_token_id", data_cfg.get("pad_token_id", 3))
        val.setdefault("shuffle_buffer", 0)
        val.setdefault("num_workers", 0)
        val.setdefault("pin_memory", False)
        val.setdefault("micro_batch_size", val_micro)
        val.setdefault("seed", cfg.get("seed", 42) + 10_000)

        if "patch_align" in data_cfg:
            val.setdefault("patch_align", data_cfg["patch_align"])

        result[name] = val

    return result


def fetch_validation_batches(
    cfg: dict,
    domains: list[str] | None,
    max_batches: int,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    if max_batches <= 0:
        raise ValueError("--max-batches must be > 0")

    result: dict[str, list[dict[str, torch.Tensor]]] = {}
    for name, val_cfg in build_validation_cfgs(cfg, domains).items():
        print(f"[diag] loading validation domain={name} ...", flush=True)
        loader = build_byte_dataloader(val_cfg, split="validation")
        it = iter(loader)

        batches: list[dict[str, torch.Tensor]] = []
        for _ in range(max_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batches.append(
                {
                    key: (
                        value.detach().cpu()
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in batch.items()
                }
            )

        if hasattr(loader, "shutdown_workers"):
            loader.shutdown_workers()

        if not batches:
            raise RuntimeError(
                f"no validation batches produced for domain={name}"
            )
        result[name] = batches
        print(
            f"[diag] domain={name} batches={len(batches)}",
            flush=True,
        )

    return result


@torch.no_grad()
def static_forward_parts(
    model: ArborModel,
    input_ids: torch.Tensor,
    *,
    skip_encoder: bool = False,
    zero_global: bool = False,
    skip_decoder: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce static Arbor forward while exposing ablation points."""
    b, t = input_ids.shape
    p = model.cfg.patch_size
    pad = (p - t % p) % p
    ids = F.pad(input_ids, (0, pad), value=3) if pad else input_ids
    k = ids.size(1) // p

    byte_emb = model.byte_emb(ids)

    enc = byte_emb.view(b * k, p, -1)
    if not skip_encoder:
        for layer in model.encoder_layers:
            enc = layer(enc)

    enc_patch = enc.view(b, k, p, -1)

    if model.cfg.patch_pooling == "legacy":
        pooled = enc_patch.flatten(2)
    elif model.cfg.patch_pooling == "mean":
        pooled = enc_patch.mean(dim=2)
    elif model.cfg.patch_pooling == "max":
        pooled = enc_patch.amax(dim=2)
    else:
        raise RuntimeError(
            f"unsupported patch_pooling={model.cfg.patch_pooling!r}"
        )

    patches = model.patch_proj(pooled)
    patch_doc = model._byte_doc_ids(ids)[:, ::p]

    if zero_global:
        global_context = byte_emb.new_zeros(
            (b, k, byte_emb.size(-1))
        )
    else:
        global_context = model._run_global(
            patches,
            model._global_mask(patch_doc),
            patch_doc,
        )

    decoder_input = (
        byte_emb.view(b, k, p, -1)
        + global_context.unsqueeze(2)
    )

    dec = decoder_input.view(b * k, p, -1)
    if not skip_decoder:
        for layer in model.decoder_layers:
            dec = layer(dec)

    logits = model.head(model.head_norm(dec)).view(
        b,
        k * p,
        -1,
    )
    if pad:
        logits = logits[:, :t]

    return logits, {
        "byte_emb": byte_emb,
        "encoder_states": enc_patch,
        "pooled": pooled,
        "patches": patches,
        "global_context": global_context,
        "decoder_input": decoder_input,
    }


def masked_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int]:
    valid = labels != -100
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0

    loss = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss), count


@torch.no_grad()
def score_variant(
    model: ArborModel,
    batches_by_domain: dict[str, list[dict[str, torch.Tensor]]],
    device: torch.device,
    dtype: torch.dtype,
    variant: str,
) -> dict[str, float]:
    kwargs = {
        "full": {},
        "skip_encoder": {"skip_encoder": True},
        "zero_global": {"zero_global": True},
        "skip_decoder": {"skip_decoder": True},
    }[variant]

    result: dict[str, float] = {}
    total_nll = 0.0
    total_labels = 0

    for domain, batches in batches_by_domain.items():
        domain_nll = 0.0
        domain_labels = 0

        for batch in batches:
            ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            with autocast_context(device, dtype):
                if variant == "full":
                    logits = model(ids).logits
                else:
                    logits, _ = static_forward_parts(
                        model,
                        ids,
                        **kwargs,
                    )

            nll, count = masked_nll(logits, labels)
            domain_nll += nll
            domain_labels += count

        result[domain] = (
            domain_nll / domain_labels / math.log(2.0)
        )
        total_nll += domain_nll
        total_labels += domain_labels

    result["mean"] = total_nll / total_labels / math.log(2.0)
    return result


@torch.no_grad()
def representation_stats(
    model: ArborModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    ids = batch["input_ids"].to(device, non_blocking=True)

    with autocast_context(device, dtype):
        _, parts = static_forward_parts(model, ids)

    stats = {
        f"{name}_rms": float(
            tensor.float().pow(2).mean().sqrt().cpu()
        )
        for name, tensor in parts.items()
    }
    stats["global_to_byte_rms_ratio"] = (
        stats["global_context_rms"]
        / max(stats["byte_emb_rms"], 1e-12)
    )
    return stats


@torch.no_grad()
def collect_probe_data(
    model: ArborModel,
    batches_by_domain: dict[str, list[dict[str, torch.Tensor]]],
    device: torch.device,
    dtype: torch.dtype,
    max_patches: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collect clean byte-only patches for linear reconstruction probes."""
    p = model.cfg.patch_size

    pooled_chunks: list[torch.Tensor] = []
    patch_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    for batches in batches_by_domain.values():
        for batch in batches:
            ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            if ids.size(1) % p != 0:
                continue

            with autocast_context(device, dtype):
                _, parts = static_forward_parts(model, ids)

            target = ids.reshape(-1, p)
            clean = (target >= 4).all(dim=1)
            if not bool(clean.any()):
                continue

            pooled = parts["pooled"].reshape(
                -1,
                parts["pooled"].size(-1),
            )[clean]
            patches = parts["patches"].reshape(
                -1,
                parts["patches"].size(-1),
            )[clean]

            pooled_chunks.append(pooled.float().cpu())
            patch_chunks.append(patches.float().cpu())
            target_chunks.append(
                (target[clean] - 4).long().cpu()
            )

    if not target_chunks:
        raise RuntimeError(
            "no clean byte-only patches available for probe"
        )

    targets = torch.cat(target_chunks, dim=0)
    sources = {
        "pooled": torch.cat(pooled_chunks, dim=0),
        "patches": torch.cat(patch_chunks, dim=0),
    }

    n = min(len(targets), max_patches)
    targets = targets[:n]
    sources = {
        name: tensor[:n]
        for name, tensor in sources.items()
    }
    return sources, targets


def fit_linear_probe(
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, object]:
    """Train a linear decoder on 80%, report held-out byte recovery."""
    torch.manual_seed(seed)

    n, dim = x_cpu.shape
    positions = y_cpu.shape[1]
    if n < 128:
        raise RuntimeError(
            f"too few patches for reconstruction probe: {n}"
        )

    perm = torch.randperm(n)
    split = max(1, min(n - 1, int(n * 0.8)))
    train_idx = perm[:split]
    test_idx = perm[split:]

    x_train = x_cpu[train_idx].to(device)
    y_train = y_cpu[train_idx].to(device)
    x_test = x_cpu[test_idx].to(device)
    y_test = y_cpu[test_idx].to(device)

    probe = torch.nn.Linear(
        dim,
        positions * 256,
        bias=True,
        device=device,
        dtype=torch.float32,
    )
    torch.nn.init.normal_(probe.weight, std=0.01)
    torch.nn.init.zeros_(probe.bias)

    opt = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    batch_size = min(1024, len(x_train))

    for _ in range(steps):
        idx = torch.randint(
            0,
            len(x_train),
            (batch_size,),
            device=device,
        )
        logits = probe(x_train[idx]).view(
            batch_size,
            positions,
            256,
        )
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            y_train[idx].flatten(),
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = probe(x_test).view(
            len(x_test),
            positions,
            256,
        )
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            y_test.flatten(),
        )
        pred = logits.argmax(dim=-1)

        position_accuracy = (
            (pred == y_test)
            .float()
            .mean(dim=0)
            .cpu()
        )
        byte_accuracy = float(
            (pred == y_test).float().mean().cpu()
        )
        exact_patch_accuracy = float(
            (pred == y_test)
            .all(dim=1)
            .float()
            .mean()
            .cpu()
        )

    del probe, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "bpb": float(loss.cpu()) / math.log(2.0),
        "byte_accuracy": byte_accuracy,
        "exact_patch_accuracy": exact_patch_accuracy,
        "position_accuracy": [
            float(v)
            for v in position_accuracy
        ],
    }


def nearest_train_metrics(checkpoint: Path) -> dict | None:
    metrics_path = checkpoint.parent / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    try:
        target_step = int(
            checkpoint.name.removeprefix("step_")
        )
    except ValueError:
        target_step = 10**30

    best = None
    for line in metrics_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "ema" not in row:
            continue

        step = int(row.get("step", -1))
        if (
            step <= target_step
            and (
                best is None
                or step >= int(best["step"])
            )
        ):
            best = row

    return best


def main() -> None:
    args = parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)

    model, cfg, dtype = load_checkpoint(
        checkpoint,
        device,
        args.precision,
    )

    print(
        "[diag] "
        f"checkpoint={checkpoint} "
        f"device={device} dtype={dtype} "
        f"patch={model.cfg.patch_size} "
        f"pooling={model.cfg.patch_pooling} "
        f"enc={len(model.encoder_layers)} "
        f"global={len(model.global_layers)} "
        f"dec={len(model.decoder_layers)}",
        flush=True,
    )

    train_row = nearest_train_metrics(checkpoint)
    if train_row:
        print(
            "[diag] nearest_train "
            f"step={train_row['step']} "
            f"loss={train_row.get('loss')} "
            f"ema={train_row.get('ema')} "
            f"lr={train_row.get('lr')}",
            flush=True,
        )

    batches = fetch_validation_batches(
        cfg,
        args.domains,
        args.max_batches,
    )

    variants = (
        "full",
        "skip_encoder",
        "zero_global",
        "skip_decoder",
    )
    ablations: dict[str, dict[str, float]] = {}

    for variant in variants:
        print(
            f"[diag] scoring {variant} ...",
            flush=True,
        )
        score = score_variant(
            model,
            batches,
            device,
            dtype,
            variant,
        )
        ablations[variant] = score
        print(
            f"[diag] {variant} "
            + " ".join(
                f"{k}_bpb={v:.4f}"
                for k, v in score.items()
            ),
            flush=True,
        )

    full = ablations["full"]["mean"]
    deltas = {
        "encoder_delta_bpb": (
            ablations["skip_encoder"]["mean"] - full
        ),
        "global_delta_bpb": (
            ablations["zero_global"]["mean"] - full
        ),
        "decoder_delta_bpb": (
            ablations["skip_decoder"]["mean"] - full
        ),
    }

    print(
        "[diag] ablation_delta "
        + " ".join(
            f"{k}={v:+.4f}"
            for k, v in deltas.items()
        ),
        flush=True,
    )

    first_batch = next(iter(batches.values()))[0]
    stats = representation_stats(
        model,
        first_batch,
        device,
        dtype,
    )

    print(
        "[diag] reps "
        + " ".join(
            f"{k}={v:.6f}"
            for k, v in stats.items()
        ),
        flush=True,
    )

    probe_sources, probe_targets = collect_probe_data(
        model,
        batches,
        device,
        dtype,
        args.probe_max_patches,
    )

    probes: dict[str, dict[str, object]] = {}
    for i, (name, source) in enumerate(
        probe_sources.items()
    ):
        print(
            "[diag] fitting reconstruction probe "
            f"source={name} "
            f"patches={len(source)} "
            f"dim={source.shape[1]} ...",
            flush=True,
        )

        probe = fit_linear_probe(
            source,
            probe_targets,
            device,
            args.probe_steps,
            args.probe_lr,
            seed=int(cfg.get("seed", 42)) + i,
        )
        probes[name] = probe

        pos = ",".join(
            f"{v:.3f}"
            for v in probe["position_accuracy"]
        )
        print(
            f"[probe] source={name} "
            f"bpb={probe['bpb']:.4f} "
            f"byte_acc={probe['byte_accuracy']:.4f} "
            f"exact_patch_acc="
            f"{probe['exact_patch_accuracy']:.4f} "
            f"pos_acc=[{pos}]",
            flush=True,
        )

    result = {
        "checkpoint": str(checkpoint),
        "model": {
            "patch_size": model.cfg.patch_size,
            "patch_pooling": model.cfg.patch_pooling,
            "encoder_layers": len(
                model.encoder_layers
            ),
            "global_layers": len(
                model.global_layers
            ),
            "decoder_layers": len(
                model.decoder_layers
            ),
            "local_hidden_size": (
                model.cfg.local_hidden_size
            ),
            "global_hidden_size": (
                model.cfg.hidden_size
            ),
        },
        "nearest_train_metrics": train_row,
        "validation_bpb": ablations,
        "ablation_delta_bpb": deltas,
        "representation_stats": stats,
        "reconstruction_probes": probes,
    }

    json_out = (
        args.json_out
        or checkpoint / "diagnostics.json"
    )
    json_out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"[diag] wrote {json_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
