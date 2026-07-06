from __future__ import annotations

import pytest
import torch

from src.train.train import resolve_precision
from src.train.train import byte_kind_loss_stats
from src.train.train import CudaBatchPrefetcher
from src.train.train import ThreadedBatchPrefetcher
from src.train.train import rebase_scheduler_lr
from src.train.train import should_restore_dataloader_state


def test_resolve_precision_accepts_supported_modes():
    assert resolve_precision("bf16") == (torch.bfloat16, True)
    assert resolve_precision("fp16") == (torch.float16, True)
    assert resolve_precision("fp32") == (torch.float32, False)


def test_resolve_precision_rejects_unknown_mode():
    with pytest.raises(ValueError, match="speed.precision"):
        resolve_precision("int8")


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
