"""Arbor MLX の学習スループット実測 (Apple Silicon GPU).

PyTorch/MPS の本走実測 (context 2048, effective batch 64 seq, 約128s/opt-step)
と比較するための合成データ bench。MLX は遅延評価なので各 iter で mx.eval して
壁時計を測る。
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_port.arbor_mlx import build_arbor_mlx

HONSOU = dict(
    vocab_size=260, max_bytes=2048, patch_size=8,
    hidden_size=2048, num_heads=16, num_kv_heads=4, intermediate_size=5632,
    num_hidden_layers=20,
    local_hidden_size=768, local_num_heads=12, local_num_kv_heads=12,
    local_intermediate_size=2048, num_local_encoder_layers=2, num_local_decoder_layers=4,
    rope_theta=500000.0, rope_theta_global=10000.0, rope_theta_local=500000.0,
    bitnet=True, activation_precision="int8",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1, help="1 opt-step あたりの micro 数")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=6, help="計測する opt-step 数")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--fwd-only", action="store_true")
    args = ap.parse_args()

    dtype = getattr(mx, args.dtype)
    cfg = dict(HONSOU, max_bytes=max(args.seq, HONSOU["patch_size"]))
    model = build_arbor_mlx(cfg)
    if dtype != mx.float32:
        model.set_dtype(dtype)
        mx.eval(model.parameters())

    opt = optim.AdamW(learning_rate=2e-4, betas=[0.9, 0.95], weight_decay=0.1)

    def loss_fn(model, x):
        logits = model(x)
        return nn.losses.cross_entropy(
            logits[:, :-1].reshape(-1, 260).astype(mx.float32), x[:, 1:].reshape(-1)
        ).mean()

    lvg = nn.value_and_grad(model, loss_fn)

    def rand_batch():
        return mx.random.randint(4, 260, (args.micro_batch, args.seq))

    def opt_step():
        if args.fwd_only:
            total = 0.0
            for _ in range(args.grad_accum):
                total = total + loss_fn(model, rand_batch())
            mx.eval(total)
            return total
        # grad accumulation
        acc = None
        loss_sum = None
        for _ in range(args.grad_accum):
            loss, grads = lvg(model, rand_batch())
            if acc is None:
                acc = grads
                loss_sum = loss
            else:
                acc = _tree_add(acc, grads)
                loss_sum = loss_sum + loss
            # accumulator を毎回 eval して micro のグラフを解放する
            # (しないと 16 個分の grad グラフが残りメモリを食い潰す)
            mx.eval(acc, loss_sum)
        if args.grad_accum > 1:
            acc = _tree_scale(acc, 1.0 / args.grad_accum)
        opt.update(model, acc)
        mx.eval(model.parameters(), opt.state)
        return loss_sum

    bytes_per_step = args.micro_batch * args.seq * args.grad_accum

    print(f"[bench] seq={args.seq} micro_batch={args.micro_batch} grad_accum={args.grad_accum} "
          f"dtype={args.dtype} fwd_only={args.fwd_only} bytes/opt-step={bytes_per_step}")
    for i in range(args.warmup):
        opt_step()
    mx.eval(model.parameters())

    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        loss = opt_step()
        mx.eval(loss)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  iter {i}: {dt*1000:.0f} ms  loss={float(loss)/max(args.grad_accum,1):.4f}")

    times.sort()
    med = times[len(times) // 2]
    avg = sum(times) / len(times)
    print(f"[bench] per opt-step: median={med*1000:.0f} ms avg={avg*1000:.0f} ms "
          f"({bytes_per_step/med:.0f} bytes/s)")
    print(f"[bench] => 1000 opt-steps ~= {med*1000/3600:.2f} h (median-based)")


def _tree_add(a, b):
    import mlx.utils as u
    return u.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, s):
    import mlx.utils as u
    return u.tree_map(lambda x: x * s, a)


if __name__ == "__main__":
    main()
