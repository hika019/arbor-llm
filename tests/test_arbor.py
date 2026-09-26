"""Arbor v2 モデルの形状・因果性・勾配のテスト (CPU, 3 patching モード)."""
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
    num_local_encoder_layers=1, num_local_decoder_layers=1,
    rope_theta=10000.0,
)
TINY_ENTROPY_LM = dict(hidden_size=32, num_heads=2, num_kv_heads=2,
                       intermediate_size=64, num_hidden_layers=1)


def tiny_cfg(mode: str) -> dict:
    cfg = dict(TINY, patching_mode=mode)
    if mode != "static":
        cfg["patch_pooling"] = "max"  # 動的モードは concat 不可 (既定 concat は static 用)
    if mode == "entropy":
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
    # T が patch_size の倍数でなくても内部 pad で処理し、T 分の logits を返す
    x = torch.randint(4, 260, (1, 10))
    out = model(x)
    assert out.logits.shape == (1, 10, 260)


def test_unknown_global_attention_impl_is_error():
    cfg = ArborConfig.from_dict(dict(TINY, global_attn_impl="auto"))
    with pytest.raises(ValueError, match="暗黙フォールバックは禁止"):
        ArborModel(cfg)


def test_unknown_patch_pooling_is_error():
    cfg = ArborConfig.from_dict(dict(TINY, patch_pooling="attention"))
    with pytest.raises(ValueError, match="patch_pooling"):
        ArborModel(cfg)


def test_concat_patch_pooling_is_rejected_in_dynamic_modes():
    """concat は patch 長固定 (static) 専用。動的モードで黙って max に化けないこと."""
    for mode in ("utf8", "space", "entropy"):
        cfg = ArborConfig.from_dict(dict(tiny_cfg(mode), patch_pooling="concat"))
        with pytest.raises(ValueError, match="concat"):
            ArborModel(cfg)
    m = ArborModel(ArborConfig.from_dict(dict(tiny_cfg("static"), patch_pooling="concat")))
    assert m.patch_proj.in_features == TINY["patch_size"] * TINY["local_hidden_size"]


@pytest.mark.parametrize("pooling", ["mean", "max"])
def test_fixed_dim_patch_pooling_is_patch_size_independent(pooling):
    cfg4 = ArborConfig.from_dict(dict(TINY, patch_size=4, patch_pooling=pooling))
    cfg8 = ArborConfig.from_dict(dict(TINY, patch_size=8, patch_pooling=pooling))
    m4 = ArborModel(cfg4).eval()
    m8 = ArborModel(cfg8).eval()
    assert m4.patch_proj.in_features == TINY["local_hidden_size"]
    assert m8.patch_proj.in_features == TINY["local_hidden_size"]
    x = torch.randint(4, 260, (1, 32))
    with torch.inference_mode():
        assert m4(x).logits.shape == (1, 32, 260)
        assert m8(x).logits.shape == (1, 32, 260)


def test_dynamic_mean_patch_pooling_forward_and_grad():
    cfg = ArborConfig.from_dict(
        dict(tiny_cfg("space"), patch_pooling="mean")
    )
    m = ArborModel(cfg)
    x = torch.randint(4, 260, (2, 30))
    x[:, ::5] = 0x20 + 4
    loss = m(x).logits.float().square().mean()
    loss.backward()
    assert m.patch_proj.weight.grad is not None
    assert torch.isfinite(m.patch_proj.weight.grad).all()


@pytest.mark.parametrize("mode", ["static", "utf8", "space", "entropy"])
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
    m = ArborModel(
        ArborConfig.from_dict(dict(tiny_cfg("static"), patch_pooling="mean"))
    ).eval()
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


@pytest.mark.parametrize("mode", ["utf8", "space", "entropy"])
def test_window_path_matches_dense(mode, monkeypatch):
    """T が chunk の倍数のときの窓 attention 経路が密マスク経路と一致すること.

    既存テストは T < _WINDOW_CHUNK で密経路しか通らないため、T=256 で
    窓経路を踏み、_WINDOW_CHUNK を巨大化して得た密経路の logits と比較する。
    """
    import src.model.arbor as arbor_mod

    torch.manual_seed(2)
    t = 2 * arbor_mod._WINDOW_CHUNK
    m = ArborModel(ArborConfig.from_dict(dict(tiny_cfg(mode), max_bytes=t))).eval()
    x = torch.randint(4, 260, (2, t))
    x[0, ::5] = 0x20 + 4  # space 境界を発生させる
    with torch.inference_mode():
        win = m(x).logits
        monkeypatch.setattr(arbor_mod, "_WINDOW_CHUNK", 10**9)  # t >= c を破り密経路へ
        dense = m(x).logits
    assert torch.allclose(win, dense, atol=1e-5), (
        f"mode={mode}: 窓経路と密マスク経路の logits が不一致 "
        f"(max diff={(win - dense).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("mode", ["utf8", "space", "entropy"])
def test_dynamic_forward_shape_and_grads(mode):
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
    if mode == "entropy":
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
    """T が chunk の倍数のときの ByteLM の窓経路 (causal、未来側 kv を持たない) が密マスク経路と一致する."""
    import src.model.arbor as arbor_mod

    torch.manual_seed(3)
    t = 2 * arbor_mod._WINDOW_CHUNK
    m = ByteLM(dict(TINY_ENTROPY_LM, vocab_size=260, max_bytes=t, attention_window=window)).eval()
    x = torch.randint(4, 260, (2, t))
    wm = m._attention_mask(x)
    assert isinstance(wm, arbor_mod.WindowMask) and wm.causal
    assert wm.mask.shape[-1] == arbor_mod._WINDOW_CHUNK + window   # 未来側 w 列を持たない
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
    orig_forward = m.entropy_model.forward

    def wrapped_forward(input_ids):
        grad_states.append(torch.is_grad_enabled())
        return orig_forward(input_ids)

    monkeypatch.setattr(m.entropy_model, "forward", wrapped_forward)
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


@pytest.mark.parametrize("mode", ["static", "utf8", "space", "entropy"])
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


def test_generator_matches_full_forward_with_mean_pooling():
    torch.manual_seed(5)
    cfg = ArborConfig.from_dict(
        dict(TINY, patch_size=8, patch_pooling="mean", max_bytes=32)
    )
    m = ArborModel(cfg).eval()
    ids = torch.randint(4, 260, (24,))
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i, byte_id in enumerate(ids):
            inc = gen.push(int(byte_id))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-4)


def test_byte_lm_forward_and_entropy():
    torch.manual_seed(0)
    lm = ByteLM(dict(TINY_ENTROPY_LM, max_bytes=64))
    x = torch.randint(4, 260, (2, 16))
    assert lm(x).logits.shape == (2, 16, 260)
    ent = lm.next_byte_entropy(x)
    assert ent.shape == (2, 16)
    assert torch.isfinite(ent).all() and (ent >= 0).all()


def test_partial_patch_padding_does_not_leak(model):
    """端数 patch の内部 pad が、それ以前の位置の logits に影響しないこと."""
    torch.manual_seed(2)
    x = torch.randint(4, 260, (1, 32))
    with torch.inference_mode():
        full = model(x).logits
        trunc = model(x[:, :10]).logits  # 内部で 12 まで pad される
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("compiled", [False, True])
def test_short_seq_attention_uses_efficient_sdpa(monkeypatch, compiled):
    """static patching の local attention (T=16) が mem-efficient SDPA を使い、flash と
    同じ値を返すこと (#CUDA speed path)。

    flash は 128 行 tile を 16 行にしか使えず 2.7 倍遅い (nsys 実測 2026-09-14)。
    数値は厳密 attention 同士なので bf16 の加算順の差だけ。
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    import src.model.arbor as arbor_mod
    from src.model.arbor import Attention, RotaryEmbedding

    torch.manual_seed(0)
    dim, heads, t = 64, 2, 16
    rope = RotaryEmbedding(dim // heads, max_pos=t, theta=10000.0).cuda()
    attn = Attention(dim, heads, heads, rope, bitnet=False, norm_eps=1e-5, causal=True).cuda().to(torch.bfloat16)
    x = torch.randn(8, t, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    fn = torch.compile(attn) if compiled else attn
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn(x)
        out.float().sum().backward()
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    assert any("fmha_cutlass" in n for n in names), names
    assert not any("flash_fwd" in n for n in names), names
    grad_eff = x.grad.clone()
    x.grad = None

    # 参照: しきい値を 0 にして既定の選択 (flash) を通す
    monkeypatch.setattr(arbor_mod, "_SHORT_SEQ_EFFICIENT_SDPA_MAX", 0)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = attn(x)
        ref.float().sum().backward()
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(grad_eff.float(), x.grad.float(), atol=2e-2, rtol=2e-2)


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
    # max_bytes=64, max_patch_len=8 → 予算の下限 8。max_patches=10 で候補の大半を予算で削らせる
    return dict(tiny_cfg(mode), min_patch_len=1, max_patch_len=8, max_patches=10,
                entropy_threshold=0.0)


@pytest.mark.parametrize("mode", ["utf8", "space", "entropy"])
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


@pytest.mark.parametrize("mode", ["utf8", "space", "entropy"])
def test_generator_matches_full_forward_with_budget(mode):
    """逐次生成器の境界判定 (予算ガードが効く状態) がフルフォワードと一致すること.

    EOS は入れない: 生成器は global の文書分離 (新文書は BOS しか見ない) を実装しておらず
    (static も同じ既存の制約)、プロンプト中の EOS ではフルフォワードと一致しない。
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


# ---------------------------------------------------------------- byte 層 (全文脈 causal の byte 単位 attention)
def _byte_cfg(mode, window=None, n=2, **over):
    cfg = dict(tiny_cfg(mode), num_byte_layers=n, byte_attn_window=window)
    if mode != "static":
        cfg.update(min_patch_len=1, max_patch_len=8)
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("mode", ["static", "space", "entropy"])
@pytest.mark.parametrize("window", [None, 5])
def test_byte_layers_forward_and_grads(mode, window):
    torch.manual_seed(0)
    m = ArborModel(ArborConfig.from_dict(_byte_cfg(mode, window)))
    x = torch.randint(4, 260, (2, 30))
    x[0, 10] = 2  # 文書境界
    out = m(x)
    assert out.logits.shape == (2, 30, 260)
    out.logits.float().square().mean().backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and n.startswith("byte_layers") and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("mode", ["static", "utf8", "space", "entropy"])
@pytest.mark.parametrize("window", [None, 6])
@pytest.mark.parametrize("pos", [4, 13, 29])
def test_byte_layers_causality(mode, window, pos):
    torch.manual_seed(1)
    m = ArborModel(ArborConfig.from_dict(_byte_cfg(mode, window))).eval()
    a = torch.randint(4, 260, (1, 40))
    a[0, ::5] = 0x20 + 4
    a[0, 20] = 2
    b = a.clone()
    b[0, pos] = (a[0, pos] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, :pos], lb[:, :pos], atol=1e-5)
    assert not torch.allclose(la[:, pos:], lb[:, pos:], atol=1e-5)


def test_byte_layers_see_far_past_bytes_directly():
    """byte 層があると、同じ patch に属さない遠い byte が patch 内 decoder を経ずに効く.

    global を切った (出力 0) モデルでも、byte 層経由で過去 patch の byte が後ろの logits を変える。
    byte 層無しなら global を切ると patch を跨ぐ情報は完全に途切れる (対照)。
    """
    torch.manual_seed(2)
    x = torch.randint(4, 260, (1, 32))
    y = x.clone()
    y[0, 1] = (x[0, 1] - 4 + 1) % 256 + 4  # patch 0 の byte
    for n_byte, expect_change in ((1, True), (0, False)):
        m = ArborModel(ArborConfig.from_dict(dict(TINY, num_byte_layers=n_byte))).eval()
        with torch.no_grad():
            m.global_to_local.weight.zero_()  # global の寄与を消す
            la, lb = m(x).logits, m(y).logits
        changed = not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-6)
        assert changed == expect_change, f"num_byte_layers={n_byte}"


def test_byte_layers_document_isolation():
    torch.manual_seed(3)
    m = ArborModel(ArborConfig.from_dict(_byte_cfg("static"))).eval()
    a = torch.randint(4, 260, (1, 16))
    a[0, 7] = 2
    b = a.clone()
    b[0, 2] = (a[0, 2] - 4 + 1) % 256 + 4
    with torch.inference_mode():
        la, lb = m(a).logits, m(b).logits
    assert torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5), "doc1 の変更が byte 層経由で doc2 に漏れている"


@pytest.mark.parametrize("mode", ["static", "space", "entropy"])
@pytest.mark.parametrize("window", [None, 5])
def test_generator_matches_full_forward_with_byte_layers(mode, window):
    torch.manual_seed(4)
    m = ArborModel(ArborConfig.from_dict(_byte_cfg(mode, window, bitnet=False))).eval()
    ids = torch.randint(4, 260, (30,))
    ids[ids == 2] = 5
    ids[::4] = 0x20 + 4
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(len(ids)):
            inc = gen.push(int(ids[i]))
            full = m(ids[: i + 1].unsqueeze(0)).logits[0, -1]
            assert torch.allclose(inc, full, atol=2e-5), f"mode={mode} window={window} pos={i}"


def test_generator_rebuild_with_byte_layers():
    torch.manual_seed(5)
    m = ArborModel(ArborConfig.from_dict(dict(TINY, max_bytes=16, num_byte_layers=1))).eval()
    gen = ArborByteGenerator(m)
    with torch.inference_mode():
        for i in range(40):
            logits = gen.push(4 + (i * 7) % 256)
    assert torch.isfinite(logits).all() and len(gen.byte_ids) <= 16


def test_local_bitnet_switch():
    from src.model.bitlinear import BitLinear

    m = ArborModel(ArborConfig.from_dict(dict(TINY, num_byte_layers=1, local_bitnet=False)))
    for name in ("encoder_layers", "byte_layers", "decoder_layers"):
        assert not any(isinstance(x, BitLinear) for x in getattr(m, name).modules()), name
    assert any(isinstance(x, BitLinear) for x in m.global_layers.modules())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("window", [None, 40])
def test_byte_layers_flex_mask_matches_dense_under_compile(window):
    """compile 下の flex BlockMask 経路が eager の密 mask 経路と一致すること (fp32)."""
    torch.manual_seed(6)
    cfg = dict(TINY, max_bytes=256, num_byte_layers=2, byte_attn_window=window, bitnet=False)
    m = ArborModel(ArborConfig.from_dict(cfg)).cuda().eval()
    x = torch.randint(4, 260, (2, 256), device="cuda")
    x[0, 100] = 2
    x[1, 37] = 2
    with torch.no_grad():
        eager = m(x).logits
        compiled = torch.compile(m)(x).logits
    assert (eager - compiled).abs().max().item() < 1e-4
