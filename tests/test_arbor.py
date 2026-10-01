"""Arbor v2 モデルの形状・因果性・勾配のテスト (CPU, 全 patching モード)."""
from __future__ import annotations

import pytest
import torch

from src.model.arbor import (
    ArborByteGenerator,
    ArborConfig,
    ArborModel,
    ByteLM,
    build_arbor,
    compute_patch_starts,
)

TINY = dict(
    vocab_size=260, patch_size=4, max_bytes=64,
    hidden_size=64, num_heads=4, num_kv_heads=2, intermediate_size=128,
    num_hidden_layers=2,
    local_hidden_size=32, local_num_heads=2, local_num_kv_heads=2,
    local_intermediate_size=64,
    num_local_encoder_layers=1, num_local_decoder_layers=2,
    cross_attn_k=2, cross_attn_heads=2, hash_ngram_vocab=97,
    rope_theta=10000.0,
)
ALL_MODES = ["static", "utf8", "space", "entropy", "entropy_char"]
TINY_ENTROPY_LM = dict(hidden_size=32, num_heads=2, num_kv_heads=2,
                       intermediate_size=64, num_hidden_layers=1)


def tiny_cfg(mode: str) -> dict:
    cfg = dict(TINY, patching_mode=mode)
    if mode in ("entropy", "entropy_char"):
        cfg["entropy_model"] = TINY_ENTROPY_LM
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_arbor(TINY).eval()


def test_forward_shape(model):
    x = torch.randint(4, 260, (2, 32))
    out = model(x)
    assert out.logits.shape == (2, 32, 260)


def test_forward_handles_partial_patch(model):
    # T が patch_size の倍数でなくても T 分の logits を返す
    x = torch.randint(4, 260, (1, 10))
    out = model(x)
    assert out.logits.shape == (1, 10, 260)


def test_unknown_global_attention_impl_is_error():
    cfg = ArborConfig.from_dict(dict(TINY, global_attn_impl="auto"))
    with pytest.raises(ValueError, match="暗黙フォールバックは禁止"):
        ArborModel(cfg)


def test_removed_config_keys_are_errors():
    for key, value in (("patch_pooling", "concat"), ("num_byte_layers", 2), ("byte_attn_window", 8)):
        with pytest.raises(ValueError, match="廃止"):
            ArborConfig.from_dict(dict(TINY, **{key: value}))


def test_patching_modes_share_the_same_weights():
    """区切り方だけが違い、patch 以降の重みは同じ形 (区切りを差し替えて推論できる)."""
    ref = ArborModel(ArborConfig.from_dict(tiny_cfg("static"))).state_dict()
    for mode in ALL_MODES[1:]:
        cfg = dict(tiny_cfg(mode), min_patch_len=1, max_patch_len=TINY["patch_size"])
        state = ArborModel(ArborConfig.from_dict(cfg)).state_dict()
        body = {k: v.shape for k, v in state.items() if not k.startswith("entropy_model.")}
        assert body == {k: v.shape for k, v in ref.items()}, mode


def test_patch_projection_only_when_k_times_local_differs_from_global():
    assert ArborModel(ArborConfig.from_dict(TINY)).patch_proj is None  # 2 * 32 = 64
    m = ArborModel(ArborConfig.from_dict(dict(TINY, cross_attn_k=3)))
    assert m.patch_proj is not None and m.patch_proj.in_features == 3 * TINY["local_hidden_size"]
    x = torch.randint(4, 260, (1, 16))
    assert m(x).logits.shape == (1, 16, 260)


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("pos", [4, 7, 13])  # patch 境界 (4) と patch 内部
def test_causality(mode, pos):
    """位置 pos のバイトを変えても、位置 < pos の logits は変わらないこと.

    動的モードでは境界判定自体もバイトに依存するため、その経路の因果性も
    まとめて検証される (空白バイトを混ぜて境界が動く入力にする)。
    """
    torch.manual_seed(1)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg(mode))).eval()
    a = torch.randint(4, 260, (1, 32))
    a[0, ::5] = 0x20 + 4  # 空白を混ぜて space 境界を発生させる
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4  # 必ず違うバイトに
    with torch.inference_mode():
        la = m(a).logits
        lb = m(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5), (
        f"mode={mode}: position {pos} の変更が過去 (<{pos}) の logits に漏れている"
    )
    # 当該位置以降には影響していること (degenerate でないことの確認)
    assert not torch.allclose(la[:, pos:], lb[:, pos:], atol=1e-5)


def test_document_attention_isolation(monkeypatch):
    """packing=document で連結した文書間に global attention が漏れないこと.

    doc1 のバイトを 1 つ変えても doc2 の logits が不変であることを確認する
    (#2: 文書境界 block-diagonal マスク)。マスクを無効化すると漏れることも
    合わせて確認し、テスト自体が leak を検出できることを担保する。
    """
    torch.manual_seed(3)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg("static"))).eval()
    # doc1 = [0..7] (index 7 が EOS), doc2 = [8..15]。patch_size=4 なので
    # 文書境界が patch 境界 (index 8) に揃い、straddle patch は生じない。
    a = torch.randint(4, 260, (1, 16))
    a[0, 7] = 2  # EOS
    b = a.clone()
    b[0, 2] = (a[0, 2] - 4 + 1) % 256 + 4  # doc1 内の 1 バイトだけ別バイトに
    with torch.inference_mode():
        la = m(a).logits
        lb = m(b).logits
    assert torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5), (
        "doc1 の変更が doc2 の logits に漏れている (global attention leak)"
    )
    # doc1 側は当然変化する (degenerate でないことの確認)
    assert not torch.allclose(la[:, 2:8], lb[:, 2:8], atol=1e-5)

    # マスクを無効化すると doc2 へ漏れる = テストが leak を検出できている
    monkeypatch.setattr(ArborModel, "_global_doc_mask", lambda self, patch_doc: None)
    with torch.inference_mode():
        la_leak = m(a).logits
        lb_leak = m(b).logits
    assert not torch.allclose(la_leak[:, 8:], lb_leak[:, 8:], atol=1e-5), (
        "マスク無効化でも doc2 が不変。テストが leak を検出できていない"
    )


def test_document_isolation_with_padding_to_patch_boundary():
    """短いdocをPADでpatch境界へ揃えた場合も、前文書が次文書へ漏れないこと.

    patch_size=4, docA=[0,1,EOS], PAD=[3], docB starts at index 4。
    packing側のpatch_align=4が生成する形を直接モデルへ通す regression test。
    """
    torch.manual_seed(7)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg("static"))).eval()
    a = torch.randint(4, 260, (1, 12))
    a[0, 2] = 2  # doc A EOS
    a[0, 3] = 3  # patch boundary までの alignment PAD
    a[0, 9] = 2  # doc B EOS
    b = a.clone()
    b[0, 0] = (a[0, 0] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la = m(a).logits
        lb = m(b).logits
    assert not torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)
    assert torch.allclose(la[:, 4:], lb[:, 4:], atol=1e-5), (
        "patch境界へalignしたdoc Aの変更がdoc Bへ漏れている"
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_forward_shape_and_grads(mode):
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg(mode)))
    x = torch.randint(4, 260, (2, 30))  # patch_size の倍数でなくてもよい
    out = m(x)
    assert out.logits.shape == (2, 30, 260)
    loss = torch.nn.functional.cross_entropy(
        out.logits.flatten(0, 1), torch.randint(4, 260, (60,))
    )
    loss.backward()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    assert not missing, f"勾配が届いていない: {missing[:5]}"
    bad = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"非有限の勾配: {bad[:5]}"
    if mode in ("entropy", "entropy_char"):
        # 凍結 ByteLM は学習されない
        assert all(not p.requires_grad for p in m.entropy_model.parameters())


def test_bytelm_attention_window_matches_full_when_large():
    torch.manual_seed(0)
    base_cfg = dict(TINY_ENTROPY_LM, vocab_size=260, max_bytes=32)
    full = ByteLM(base_cfg).eval()
    windowed = ByteLM({**base_cfg, "attention_window": 32}).eval()
    windowed.load_state_dict(full.state_dict())
    x = torch.randint(4, 260, (2, 24))
    with torch.inference_mode():
        full_logits = full(x).logits
        window_logits = windowed(x).logits
    assert torch.allclose(window_logits, full_logits, atol=1e-5)


@pytest.mark.parametrize("window", [32, 200])
def test_bytelm_window_path_matches_dense(window, monkeypatch):
    """T が chunk の倍数のときの ByteLM の窓経路が密マスク経路と一致すること."""
    import src.model.arbor as arbor_mod

    torch.manual_seed(3)
    t = 2 * arbor_mod._WINDOW_CHUNK
    m = ByteLM(dict(TINY_ENTROPY_LM, vocab_size=260, max_bytes=t, attention_window=window)).eval()
    x = torch.randint(4, 260, (2, t))
    wm = m._attention_mask(x)
    assert isinstance(wm, arbor_mod.WindowMask) and wm.causal
    assert wm.mask.shape[-1] == arbor_mod._WINDOW_CHUNK + window
    with torch.inference_mode():
        win = m(x).logits
        monkeypatch.setattr(arbor_mod, "_WINDOW_CHUNK", 10**9)      # t >= c を破り密経路へ
        dense = m(x).logits
    assert torch.allclose(win, dense, atol=1e-5), f"max diff={(win - dense).abs().max().item():.2e}"


def test_bytelm_attention_window_forward_shape():
    torch.manual_seed(0)
    m = ByteLM(dict(TINY_ENTROPY_LM, vocab_size=260, max_bytes=64, attention_window=8)).eval()
    x = torch.randint(4, 260, (2, 32))
    out = m(x)
    assert out.logits.shape == (2, 32, 260)


def test_space_boundaries():
    # "ab cd" -> 空白の直後 (c の位置) で新 patch
    ids = torch.tensor([[ord("a"), ord("b"), 0x20, ord("c"), ord("d")]]) + 4
    starts = compute_patch_starts(ids, "space", min_len=1, max_len=16)
    assert starts.tolist() == [[True, False, False, True, False]]


def test_utf8_boundaries_follow_codepoint_starts():
    ids = torch.tensor([list("あいbう".encode("utf-8"))]) + 4
    starts = compute_patch_starts(ids, "utf8", min_len=1, max_len=16)
    assert starts[0].nonzero().flatten().tolist() == [0, 3, 6, 7]


def test_utf8_boundaries_respect_min_len():
    ids = torch.tensor([list("あいbう".encode("utf-8"))]) + 4
    starts = compute_patch_starts(ids, "utf8", min_len=4, max_len=16)
    assert starts[0].nonzero().flatten().tolist() == [0, 6]


def test_boundary_min_max_enforcement():
    # 毎バイト空白 (= 毎位置が境界候補) でも min_len 未満では切らない
    ids = torch.full((1, 12), 0x20 + 4)
    starts = compute_patch_starts(ids, "space", min_len=3, max_len=16)
    assert starts.long().sum() == 4  # 12 / 3
    # 境界候補ゼロでも max_len で強制的に切る
    ids = torch.full((1, 12), ord("a") + 4)
    starts = compute_patch_starts(ids, "space", min_len=2, max_len=4)
    assert starts[0].nonzero().flatten().tolist() == [0, 4, 8]


def _space_raw(ids: torch.Tensor) -> torch.Tensor:
    """実装と同じ空白系バイト (space/tab/LF/CR) で境界候補を作る."""
    raw = torch.zeros_like(ids, dtype=torch.bool)
    for sb in (0x20, 0x09, 0x0A, 0x0D):
        raw[:, 1:] |= (ids[:, :-1] - 4) == sb
    return raw


def _reference_patch_starts(raw: torch.Tensor, min_len: int, max_len: int) -> torch.Tensor:
    """旧実装 (バイト毎の逐次ループ)。ジャンプ版の等価性検証用リファレンス."""
    b, t = raw.shape
    starts = torch.zeros(b, t, dtype=torch.bool)
    run = torch.zeros(b, dtype=torch.long)
    for i in range(t):
        s = (run >= max_len) | (raw[:, i] & (run >= min_len)) if i > 0 \
            else torch.ones(b, dtype=torch.bool)
        starts[:, i] = s
        run = torch.where(s, torch.ones_like(run), run + 1)
    return starts


@pytest.mark.parametrize("min_len,max_len", [(1, 16), (2, 16), (3, 4), (2, 2), (4, 8)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_patch_starts_matches_sequential_reference(min_len, max_len, seed):
    """ジャンプ版 compute_patch_starts が旧逐次実装と完全一致すること."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, 260, (3, 97), generator=g)
    ids[torch.rand(ids.shape, generator=g) < 0.15] = 0x20 + 4  # 空白を散らす
    raw = _space_raw(ids)
    got = compute_patch_starts(ids, "space", min_len=min_len, max_len=max_len)
    want = _reference_patch_starts(raw, min_len, max_len)
    assert torch.equal(got, want)


def test_patch_starts_edge_cases():
    # 全バイト空白 / 候補ゼロ / T=1
    all_space = torch.full((1, 10), 0x20 + 4)
    no_space = torch.full((1, 10), ord("a") + 4)
    for ids in (all_space, no_space, torch.full((1, 1), ord("a") + 4)):
        raw = _space_raw(ids)
        got = compute_patch_starts(ids, "space", min_len=2, max_len=5)
        want = _reference_patch_starts(raw, 2, 5)
        assert torch.equal(got, want)


def test_entropy_model_runs_without_grad_during_boundary_scoring(monkeypatch):
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg("entropy")))
    grad_states = []
    orig_hidden = m.entropy_model._hidden

    def wrapped_hidden(input_ids):
        grad_states.append(torch.is_grad_enabled())
        return orig_hidden(input_ids)

    monkeypatch.setattr(m.entropy_model, "_hidden", wrapped_hidden)
    x = torch.randint(4, 260, (2, 30))
    out = m(x)
    assert out.logits.requires_grad
    assert grad_states and not any(grad_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_patch_starts_cuda_matches_cpu_reference():
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        pytest.skip("CUDA toolkit is not available")

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(4, 260, (2, 113), generator=g)
    ids[torch.rand(ids.shape, generator=g) < 0.2] = 0x20 + 4
    cpu = compute_patch_starts(ids, "space", min_len=3, max_len=16)
    cuda = compute_patch_starts(ids.cuda(), "space", min_len=3, max_len=16).cpu()
    assert torch.equal(cuda, cpu)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_generator_matches_full_forward(mode):
    """KV cache 逐次生成器がフルフォワードと同じ logits を返すこと (全モード)."""
    torch.manual_seed(3)
    # bitnet=False: BitNet の per-token int8 活性量子化は 1e-7 の数値差でも丸め境界を跨ぐと
    # 出力が ~1e-3 跳ねるため、境界・KV cache のロジック一致の検証には使えない
    m = ArborModel(ArborConfig.from_dict(dict(tiny_cfg(mode), bitnet=False))).eval()
    ids = torch.randint(4, 260, (26,))
    ids[::5] = 0x20 + 4  # 空白を混ぜて動的境界を発生させる
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-4), (
                f"mode={mode}: 位置 {i} で逐次生成とフルフォワードの logits が不一致 "
                f"(max diff={float((inc - full).abs().max()):.2e})"
            )


def test_generator_context_rebuild():
    """max_bytes 到達時に内部で window を作り直しても落ちないこと."""
    torch.manual_seed(4)
    m = ArborModel(ArborConfig.from_dict(dict(TINY, max_bytes=16))).eval()
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(40):  # max_bytes=16 を 2 回以上超える
            logits = gen.push(4 + (i * 7) % 256)
    assert torch.isfinite(logits).all()
    assert len(gen.byte_ids) <= 16


def test_pad_patches_get_no_nan_gradients():
    """byte の無い pad patch (予算に対して patch が少ない) があっても勾配は有限."""
    torch.manual_seed(7)
    m = ArborModel(ArborConfig.from_dict(dict(tiny_cfg("space"), min_patch_len=1, max_patch_len=8)))
    assert m.max_patches > 30
    x = torch.randint(4, 260, (2, 30))
    x[:, ::5] = 0x20 + 4
    m(x).logits.float().square().mean().backward()
    bad = [n for n, par in m.named_parameters() if par.grad is not None and not torch.isfinite(par.grad).all()]
    assert not bad, bad


def test_byte_lm_forward_and_entropy():
    torch.manual_seed(0)
    lm = ByteLM(dict(TINY_ENTROPY_LM, max_bytes=64))
    x = torch.randint(4, 260, (2, 16))
    assert lm(x).logits.shape == (2, 16, 260)
    ent = lm.next_byte_entropy(x)
    assert ent.shape == (2, 16)
    assert torch.isfinite(ent).all() and (ent >= 0).all()


def test_partial_patch_does_not_leak(model):
    """途中で切った入力 (最後の patch が端数) でも、それ以前の位置の logits は同じ."""
    torch.manual_seed(2)
    x = torch.randint(4, 260, (1, 32))
    with torch.inference_mode():
        full = model(x).logits
        trunc = model(x[:, :10]).logits
    assert torch.allclose(full[:, :9], trunc[:, :9], atol=1e-5)


def test_gradients_reach_all_parameters():
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(TINY))
    x = torch.randint(4, 260, (2, 16))
    out = m(x)
    loss = torch.nn.functional.cross_entropy(
        out.logits.flatten(0, 1), torch.randint(4, 260, (32,))
    )
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"勾配が届いていないパラメータ: {missing[:5]}"
    bad = [n for n, p in m.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"非有限の勾配: {bad[:5]}"


def test_global_bos_is_not_zero_initialized():
    """ゼロ初期化の BOS は全層で厳密ゼロ行のまま伝播し、RMSNorm backward の
    1/sqrt(eps) 増幅が複利になって勾配が overflow する (実際に起きた事故)."""
    m = ArborModel(ArborConfig.from_dict(TINY))
    assert m.global_bos.abs().max() > 0


def test_bitnet_flag_swaps_linears():
    from src.model.bitlinear import BitLinear

    bit = ArborModel(ArborConfig.from_dict(TINY))
    fp = ArborModel(ArborConfig.from_dict({**TINY, "bitnet": False}))
    assert sum(1 for m in bit.modules() if isinstance(m, BitLinear)) > 0
    assert sum(1 for m in fp.modules() if isinstance(m, BitLinear)) == 0


def test_param_count_reporting(model):
    counts = model.num_parameters()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["global"] > 0 and counts["local_decoder"] > 0


def test_rope_theta_per_level_and_fallback():
    """rope_theta_global/local が階層別に効き、未指定なら rope_theta に落ちること (#1)."""
    # 階層別指定: global と local で theta が分かれる
    cfg = dict(TINY, rope_theta=500000.0, rope_theta_global=10000.0, rope_theta_local=123456.0)
    m = ArborModel(ArborConfig.from_dict(cfg))
    assert m.global_layers[0].attn.rope.theta == 10000.0
    assert m.encoder_layers[0].attn.rope.theta == 123456.0
    assert m.decoder_layers[0].attn.rope.theta == 123456.0

    # 後方互換: global/local 未指定なら両方 rope_theta を使う
    m2 = ArborModel(ArborConfig.from_dict(dict(TINY, rope_theta=777.0)))
    assert m2.global_layers[0].attn.rope.theta == 777.0
    assert m2.encoder_layers[0].attn.rope.theta == 777.0


def test_global_attn_flex_matches_sdpa():
    """global_attn_impl=flex が sdpa と同一 logits を返すこと (#CUDA speed path).

    flex_attention (BlockMask + native GQA) は密マスク SDPA と同じ文書境界規則を
    表す。torch に flex_attention が無い環境では skip。
    """
    pytest.importorskip("torch.nn.attention.flex_attention")
    import warnings

    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(tiny_cfg("static"))).eval()
    x = torch.randint(4, 260, (2, 32))
    x[0, 15] = 2  # EOS -> 複数文書
    x[1, 7] = 2
    x[1, 20] = 2
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.cfg.global_attn_impl = "sdpa"
        la = m(x).logits
        m.cfg.global_attn_impl = "flex"
        lb = m(x).logits
    assert torch.allclose(la, lb, atol=1e-4), (
        f"flex と sdpa の logits 不一致 (max diff={(la - lb).abs().max().item():.2e})"
    )


# ---------------------------------------------------------------- 境界: 文書先頭の強制区切り + 予算ガード
def _check_patch_rules(starts, raw, force, min_len, max_len, budget, horizon):
    """_patch_starts_reference の出力が規則を満たすこと (独立な検査)."""
    for r in range(starts.size(0)):
        pos = starts[r].nonzero().flatten().tolist()
        assert pos[0] == 0
        lens = [b - a for a, b in zip(pos, pos[1:] + [starts.size(1)])]
        assert max(lens) <= max_len
        if budget:
            assert len(pos) <= budget
        for c, p in enumerate(pos[1:], start=1):  # c = p より前に開いていた patch 数
            if lens[c - 1] == max_len:
                continue  # max_len 到達の強制境界
            assert force[r, p] or (raw[r, p] and lens[c - 1] >= min_len)
            if budget:
                assert c + -(-max(horizon - p, 0) // max_len) <= budget


@pytest.mark.parametrize("budget", [0, 9, 12, 20])
@pytest.mark.parametrize("seed", [0, 1])
def test_patch_starts_force_and_budget_rules(budget, seed):
    from src.model.arbor import _patch_starts_reference

    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(3, 64, generator=g) < 0.5
    force = torch.rand(3, 64, generator=g) < 0.05
    starts = _patch_starts_reference(raw, force, 2, 8, budget, 64)
    _check_patch_rules(starts, raw, force, 2, 8, budget, 64)
    if budget == 0:
        assert all(starts[force].tolist())  # 予算無しなら文書先頭は必ず境界


def test_patch_budget_binds_and_never_overflows():
    """全位置が候補でも patch 数は budget ちょうどに収まる (上限超えのエラーが起きない)."""
    from src.model.arbor import _patch_starts_reference

    raw = torch.ones(2, 64, dtype=torch.bool)
    force = torch.zeros_like(raw)
    starts = _patch_starts_reference(raw, force, 1, 16, 10, 64)
    assert starts.sum(1).tolist() == [10, 10]
    free = _patch_starts_reference(raw, force, 1, 16, 0, 64)
    assert free.sum(1).tolist() == [64, 64]


def test_budget_rejects_infeasible_max_patches():
    with pytest.raises(ValueError, match="max_patches"):
        ArborModel(ArborConfig.from_dict(dict(tiny_cfg("space"), max_patches=2, max_patch_len=16)))


def test_document_start_forces_patch_boundary():
    eos = 2
    ids = torch.full((1, 20), ord("a") + 4)
    ids[0, 6] = eos
    starts = compute_patch_starts(ids, "space", min_len=4, max_len=16, eos_token_id=eos)
    assert starts[0].nonzero().flatten().tolist() == [0, 7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("budget", [0, 16, 30])
def test_patch_starts_cuda_matches_reference_with_force_and_budget(budget):
    from src.model.arbor import _patch_starts_reference
    from src.model.patch_starts_cuda import patch_starts_cuda

    g = torch.Generator().manual_seed(7)
    raw = torch.rand(4, 173, generator=g) < 0.4
    force = torch.rand(4, 173, generator=g) < 0.03
    want = _patch_starts_reference(raw, force, 3, 12, budget, 180)
    got = patch_starts_cuda(raw.cuda(), force.cuda(), 3, 12, budget, 180).cpu()
    assert torch.equal(got, want)


def _tight_dynamic_cfg(mode):
    # max_bytes=64, max_patch_len=8 → 予算の下限 8 (entropy_char は単位 5 で 13)。下限の少し上に置き、
    # 候補の大半を予算で削らせる
    return dict(tiny_cfg(mode), min_patch_len=1, max_patch_len=8, patch_size=8,
                max_patches=14 if mode == "entropy_char" else 10, entropy_threshold=0.0)


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("pos", [5, 20, 40])
def test_causality_with_budget_and_documents(mode, pos):
    """予算ガードが効き、文書境界がある状態でも未来のバイトが過去の logits に漏れない."""
    torch.manual_seed(11)
    m = ArborModel(ArborConfig.from_dict(_tight_dynamic_cfg(mode))).eval()
    a = torch.randint(4, 260, (1, 60))
    a[0, ::3] = 0x20 + 4
    a[0, 30] = 2  # EOS: 位置 31 から次の文書
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_generator_matches_full_forward_with_budget(mode):
    """逐次生成器の境界判定 (予算ガードが効く状態) がフルフォワードと一致すること.

    EOS は入れない: 生成器は文書分離を実装しておらず、プロンプト中の EOS ではフルフォワードと一致しない。
    """
    torch.manual_seed(12)
    m = ArborModel(ArborConfig.from_dict(dict(_tight_dynamic_cfg(mode), bitnet=False))).eval()
    ids = torch.randint(4, 260, (50,))
    ids[ids == 2] = 5
    ids[::3] = 0x20 + 4
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-4), f"mode={mode} pos={i}"


# ---------------------------------------------------------------- local の byte 層 (文書内 causal、窓)
def _local_cfg(mode, window=None, **over):
    cfg = dict(tiny_cfg(mode), local_attn_window=window)
    if mode != "static":
        cfg.update(min_patch_len=1, max_patch_len=8)
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("window", [None, 6])
@pytest.mark.parametrize("pos", [4, 13, 29])
def test_local_window_causality(mode, window, pos):
    torch.manual_seed(1)
    m = ArborModel(ArborConfig.from_dict(_local_cfg(mode, window))).eval()
    a = torch.randint(4, 260, (1, 40))
    a[0, ::5] = 0x20 + 4
    a[0, 20] = 2
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5)
    assert not torch.allclose(la[:, pos:], lb[:, pos:], atol=1e-5)


def test_local_window_limits_byte_level_reach():
    """global を切ると、byte 同士は local の窓 (と hash n-gram) の届く範囲でしか影響しない.

    窓 None なら遠い byte も local の byte 層で直接効く。窓 4 では encoder 1 層 + decoder 2 層で
    3 * 3 byte、hash n-gram で 7 byte 先までしか届かない。
    """
    torch.manual_seed(2)
    x = torch.randint(4, 260, (1, 40))
    y = x.clone()
    y[0, 1] = (x[0, 1] - 4 + 1) % 256 + 4
    reach = 1 + 3 * 3 + 7
    for window, expect_change in ((None, True), (4, False)):
        m = ArborModel(ArborConfig.from_dict(dict(TINY, local_attn_window=window))).eval()
        with torch.no_grad():
            m.global_to_local.weight.zero_()
            la, lb = m(x).logits, m(y).logits
        changed = not torch.allclose(la[:, reach + 1:], lb[:, reach + 1:], atol=1e-6)
        assert changed == expect_change, f"window={window}"


def test_local_layers_document_isolation():
    torch.manual_seed(3)
    m = ArborModel(ArborConfig.from_dict(_local_cfg("static"))).eval()
    a = torch.randint(4, 260, (1, 16))
    a[0, 7] = 2
    b = a.clone()
    b[0, 6] = (a[0, 6] - 4 + 1) % 256 + 4  # hash n-gram も文書をまたがないこと
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5), "doc1 の変更が doc2 に漏れている"


@pytest.mark.parametrize("mode", ["static", "space", "entropy", "entropy_char"])
@pytest.mark.parametrize("window", [None, 5])
def test_generator_matches_full_forward_with_local_window(mode, window):
    torch.manual_seed(4)
    m = ArborModel(ArborConfig.from_dict(_local_cfg(mode, window, bitnet=False))).eval()
    ids = torch.randint(4, 260, (30,))
    ids[ids == 2] = 5
    ids[::4] = 0x20 + 4
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-5), f"mode={mode} window={window} pos={i}"


def test_local_bitnet_switch():
    from src.model.bitlinear import BitLinear

    m = ArborModel(ArborConfig.from_dict(dict(TINY, local_bitnet=False)))
    for name in ("encoder_layers", "decoder_layers", "encoder_cross_attn", "decoder_cross_attn"):
        assert not any(isinstance(x, BitLinear) for x in getattr(m, name).modules()), name
    assert any(isinstance(x, BitLinear) for x in m.global_layers.modules())


def test_cross_attention_is_bitlinear_and_not_qkv_fused():
    """cross-attention の射影は 3 値。q と kv は入力が違うので qkv 融合の対象にしない."""
    from src.model.bitlinear import BitLinear, install_arbor_projection_fusions

    m = ArborModel(ArborConfig.from_dict(TINY))
    for xattn in (*m.encoder_cross_attn, *m.decoder_cross_attn):
        assert all(isinstance(getattr(xattn, n), BitLinear) for n in ("wq", "wk", "wv", "wo"))
    info = install_arbor_projection_fusions(m)
    n_self_attn = len(m.encoder_layers) + len(m.global_layers) + len(m.decoder_layers)
    assert info["qkv_groups"] == n_self_attn
    assert not any(hasattr(x, "_fast_qkv_group") for x in (*m.encoder_cross_attn, *m.decoder_cross_attn))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("window", [None, 40])
def test_local_flex_mask_matches_dense_under_compile(window):
    """compile 下の flex BlockMask 経路が eager の密 mask 経路と一致すること (fp32)."""
    torch.manual_seed(6)
    cfg = dict(TINY, max_bytes=256, local_attn_window=window, bitnet=False)
    m = ArborModel(ArborConfig.from_dict(cfg)).cuda().eval()
    x = torch.randint(4, 260, (2, 256), device="cuda")
    x[0, 100] = 2
    x[1, 37] = 2
    with torch.no_grad():
        eager = m(x).logits
        compiled = torch.compile(m)(x).logits
    assert (eager - compiled).abs().max().item() < 1e-4


# ---------------------------------------------------------------- hash n-gram 埋め込み
def _blt_byte_group_hash(x: torch.Tensor, group_size: int, max_hash: int) -> torch.Tensor:
    """facebookresearch/blt の byte_group_hash_function (hash 関数 0 番) の写し."""
    prime = torch.tensor(1000000007, dtype=torch.int64)
    powers = torch.stack([prime**i for i in range(group_size)])
    x = torch.cat([torch.zeros(x.shape[0], group_size - 1, dtype=torch.int64), x], dim=1)
    return torch.sum(x.unfold(1, group_size, 1) * powers, dim=-1) % max_hash


def test_hash_ngram_ids_match_blt_reference():
    from src.model.arbor import HashNgramEmbedding

    emb = HashNgramEmbedding((3, 5, 8), vocab=50002, dim=4)
    ids = torch.randint(4, 260, (2, 40))
    seen = []
    for table in emb.tables:
        table.register_forward_hook(lambda mod, args, out: seen.append(args[0]))
    emb(ids, torch.zeros_like(ids))
    for n, got in zip((3, 5, 8), seen):
        assert torch.equal(got, _blt_byte_group_hash(ids, n, 50002)), n


def test_hash_ngram_resets_at_document_start():
    from src.model.arbor import HashNgramEmbedding

    torch.manual_seed(0)
    emb = HashNgramEmbedding((3, 4), vocab=101, dim=8)
    ids = torch.randint(4, 260, (1, 12))
    doc = torch.zeros_like(ids)
    doc[0, 6:] = 1
    other = ids.clone()
    other[0, 5] = (ids[0, 5] - 4 + 1) % 256 + 4  # 前の文書の最後の byte
    a, b = emb(ids, doc), emb(other, doc)
    assert torch.equal(a[0, 6:], b[0, 6:])
    assert not torch.equal(a[0, 5], b[0, 5])


def test_bytelm_is_llama_style_by_default():
    m = ByteLM(dict(TINY_ENTROPY_LM, max_bytes=32))
    blk = m.layers[0]
    assert blk.attn.attn_sub_norm is None and blk.ffn.ffn_sub_norm is None
    assert blk.ffn.activation == "swiglu"
    old = ByteLM(dict(TINY_ENTROPY_LM, max_bytes=32, sub_norm=True, ffn_activation="relu2"))
    assert old.layers[0].ffn.ffn_sub_norm is not None and old.layers[0].ffn.activation == "relu2"


def test_arbor_blocks_keep_bitnet_subln_relu2():
    m = ArborModel(ArborConfig.from_dict(TINY))
    for blk in (m.global_layers[0], m.decoder_layers[0]):
        assert blk.attn.attn_sub_norm is not None and blk.ffn.activation == "relu2"


def test_bytelm_residual_stream_uses_autocast_dtype_with_fp32_params():
    m = ByteLM(dict(TINY_ENTROPY_LM, max_bytes=32))
    seen = []
    m.layers[0].register_forward_hook(lambda mod, args, out: seen.append(args[0].dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        m(torch.randint(4, 260, (1, 16)))
    assert next(m.parameters()).dtype is torch.float32
    assert seen == [torch.bfloat16]



# ---------------------------------------------------------------- entropy_char (文字先頭だけで区切る)
def _ja_ids(text: str) -> torch.Tensor:
    return torch.tensor([[b + 4 for b in text.encode()]])


def test_entropy_char_never_splits_inside_a_character():
    """2 byte 目 (継続 byte) の予測が難しくても、区切りは必ず文字の先頭に来る."""
    ids = _ja_ids("政府は来年度予算の概算要求を取りまとめた。一般会計の総額は過去最大となる。")
    g = torch.Generator().manual_seed(0)
    ent = torch.rand(ids.shape, generator=g) * 4
    cur = ids - 4
    cont = (cur & 0xC0) == 0x80
    ent[cont] += 3.0  # 継続 byte の予測を難しくする (実モデルの傾向)
    for mode in ("entropy", "entropy_char"):
        st = compute_patch_starts(ids, mode, 1, 16, entropy_values=ent, threshold=3.0)[0]
        mid = int((st & cont[0]).sum())
        if mode == "entropy":
            assert mid > 0, "対照: entropy は文字の途中で区切る"
        else:
            assert mid == 0


def test_entropy_char_cuts_where_char_entropy_rises():
    text = "来年度予算のabc、数字123と😀を含む。"
    ids = _ja_ids(text)
    g = torch.Generator().manual_seed(1)
    ent = torch.rand(ids.shape, generator=g) * 5
    e = ent[0].tolist()
    expected, prev_h, pos = [0], None, 0
    for ch in text:
        n = len(ch.encode())
        h = (e[pos - 1] if pos > 0 else 0.0) + (e[pos] if n > 1 else 0.0)
        if prev_h is not None and h - prev_h > 1.0:
            expected.append(pos)
        prev_h, pos = h, pos + n
    st = compute_patch_starts(ids, "entropy_char", 1, 1000, entropy_values=ent, threshold=1.0)[0]
    assert st.nonzero().flatten().tolist() == expected
    assert len(expected) > 3


def test_entropy_char_soft_boundary_before_max_len_and_budget():
    from src.model.arbor import _patch_starts_reference

    ids = _ja_ids("あ" * 60)
    ent = torch.zeros(ids.shape)  # 閾値を超えない → 最長付近の文字先頭でだけ区切る
    cur = ids - 4
    cont = (cur & 0xC0) == 0x80
    st = compute_patch_starts(ids, "entropy_char", 1, 16, entropy_values=ent, threshold=10.0,
                              budget=15, horizon=180)[0]
    pos = st.nonzero().flatten().tolist()
    lens = [b - a for a, b in zip(pos, pos[1:] + [ids.size(1)])]
    assert not bool((st & cont[0]).any())
    assert max(lens) <= 16 and all(ln >= 13 for ln in lens[:-1])
    assert len(pos) <= 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_entropy_char_cuda_matches_reference():
    g = torch.Generator().manual_seed(3)
    text = "日本語のテキストとEnglish words、数字123と記号!?を混ぜた文章です。" * 4
    ids = _ja_ids(text)
    ent = torch.rand(ids.shape, generator=g) * 5
    for budget in (0, 40):
        for threshold in (0.0, 1.5):
            kw = dict(entropy_values=ent, threshold=threshold, eos_token_id=2, budget=budget,
                      horizon=ids.size(1))
            cpu = compute_patch_starts(ids, "entropy_char", 1, 12, **kw)
            gpu = compute_patch_starts(ids.cuda(), "entropy_char", 1, 12,
                                       **dict(kw, entropy_values=ent.cuda())).cpu()
            assert torch.equal(cpu, gpu)


def test_generator_matches_full_forward_entropy_char_japanese():
    torch.manual_seed(13)
    cfg = dict(tiny_cfg("entropy_char"), min_patch_len=1, max_patch_len=8, max_patches=40,
               entropy_threshold=1.0, bitnet=False, max_bytes=128)
    m = ArborModel(ArborConfig.from_dict(cfg)).eval()
    ids = _ja_ids("今日は天気が良いので散歩に行きました。明日も晴れるといいな。")[0][:90]
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-4), f"pos={i}"


# ---------------------------------------------------------------- entropy_char_rest (3 byte 目以降の推定)
def _rest_cfg(**over):
    cfg = dict(_tight_dynamic_cfg("entropy_char"), entropy_char_rest=True)
    cfg.update(over)
    return cfg


def _randomize_rest_head(m):
    for layer in (m.entropy_model.char_rest[0], m.entropy_model.char_rest[-1]):
        torch.nn.init.normal_(layer.weight, std=1.0)
        torch.nn.init.constant_(layer.bias, 0.5)


def _utf8_heavy_ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, 260, (n,), generator=g)
    ids[::4] = 0xE3 + 4  # 3 byte 文字の 1 byte 目を多めに混ぜる
    ids[1::9] = 0xF0 + 4
    ids[ids == 2] = 5
    return ids


def test_char_rest_head_changes_boundaries_only_at_3_4_byte_leads():
    ids = torch.tensor([[0x41, 0xE3, 0x81, 0xAF, 0x42, 0xC3, 0xA9, 0xF0, 0x9F, 0x98, 0x80]]) + 4
    ent = torch.zeros(ids.shape)
    rest = torch.full(ids.shape, 5.0)
    base = compute_patch_starts(ids, "entropy_char", 1, 32, entropy_values=ent, threshold=1.0)
    with_rest = compute_patch_starts(ids, "entropy_char", 1, 32, entropy_values=ent,
                                     rest_values=rest, threshold=1.0)
    assert not base[0, 1:].any()
    # rest は 3 byte (E3) と 4 byte (F0) の 1 byte 目だけに足され、2 byte (C3)・ASCII には効かない
    assert with_rest[0].nonzero().flatten().tolist() == [0, 1, 7]


def test_char_rest_requires_entropy_char_mode():
    with pytest.raises(ValueError, match="entropy_char_rest"):
        ArborModel(ArborConfig.from_dict(dict(tiny_cfg("entropy"), entropy_char_rest=True)))


@pytest.mark.parametrize("pos", [5, 20, 40])
def test_char_rest_causality(pos):
    torch.manual_seed(13)
    m = ArborModel(ArborConfig.from_dict(_rest_cfg())).eval()
    _randomize_rest_head(m)
    a = _utf8_heavy_ids(60, 3).unsqueeze(0)
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5)


def test_char_rest_generator_matches_full_forward():
    torch.manual_seed(14)
    m = ArborModel(ArborConfig.from_dict(_rest_cfg(bitnet=False))).eval()
    _randomize_rest_head(m)
    ids = _utf8_heavy_ids(50, 4)
    with torch.inference_mode():
        _, rest = m.entropy_model.boundary_entropy(ids.unsqueeze(0))
    assert rest is not None and rest.std() > 0.1  # head が実際に区切りへ効く状態で比べる
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-4), f"pos={i}"


def test_char_rest_generator_runs_in_bf16_without_autocast():
    torch.manual_seed(15)
    m = ArborModel(ArborConfig.from_dict(_rest_cfg(bitnet=False))).eval().to(torch.bfloat16)
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for b in _utf8_heavy_ids(12, 5).tolist():
            gen.push(b)
    assert gen.prev_rest != 0.0
