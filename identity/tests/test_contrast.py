"""The contrast search: pair filters, splits, t-statistics, the sign-flip null, replication and the mean tap. No real model."""
import json

import numpy as np
import pytest

from identity import contrast as C


def planted(n=40, layers=5, width=30, seed=0):
    """Three registers' differences [n, L, W] of pure noise, plus a neuron (3, 7) that register "a" raises and a neuron
    (3, 9) that every register raises (the "this is not plain text" mode)."""
    rng = np.random.default_rng(seed)
    D = {r: rng.normal(size=(n, layers, width)).astype(np.float32) for r in "abc"}
    D["a"][:, 3, 7] += 1.5
    for r in D:
        D[r][:, 3, 9] += 1.5
    return D


def test_split_is_stable_by_topic_and_uses_all_three_splits():
    assert C.split_of("a tide table") == C.split_of("a tide table")
    seen = {C.split_of(t) for t in C.TOPICS}
    assert seen == {"train", "val", "test"}
    assert sum(C.split_of(t) == "train" for t in C.TOPICS) > len(C.TOPICS) / 2


def test_reject_reason_for_plain_passages_and_rewrites():
    ok = "Fold the sheet in half. Crease it well. Then fold the top corners down to the middle line of the page."
    assert C.reject_reason(ok) is None
    assert C.reject_reason("Here is a short passage about tea.") == "meta talk"
    assert C.reject_reason("Too short.") == "length"
    assert C.reject_reason(ok, plain=ok) == "barely changed"
    assert C.reject_reason(ok.upper(), plain=ok) == "barely changed"
    assert C.reject_reason(ok + " Just like a hug.", plain=ok) == "barely changed"       # (a clause tacked on the end)
    assert C.reject_reason("Fold it.", plain=ok) == "length"
    assert C.reject_reason("Sure! Here's the rewrite: " + ok, plain=ok) == "meta talk"
    rewrite = "Fold ye sheet in twain, good sir, and crease it well, then bring the top corners down to the midmost line."
    assert C.reject_reason(rewrite, plain=ok) is None


def test_complete_keeps_only_pairs_with_every_register():
    full = {"regs": {r: "x" for r in C.REGISTERS}}
    assert C.complete([full, {"regs": {"baby": "x"}}]) == [full]


def test_tstat_is_the_paired_t_and_a_constant_entry_gets_zero():
    rng = np.random.default_rng(1)
    d = rng.normal(0.5, 1.0, size=(25, 4))
    ref = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(25))
    assert np.allclose(C.tstat(d), ref)
    d[:, 2] = 0.0
    assert C.tstat(d)[2] == 0.0


def test_planted_register_neuron_leads_against_the_other_registers_and_the_shared_one_drops_out():
    D = planted()
    rows = np.arange(40)
    plain = C.t_map(D, "a", rows, "plain")
    others = C.t_map(D, "a", rows, "others")
    assert {top[:2] for top in C.top_k(plain, 2, 0)} == {(3, 7), (3, 9)}            # (both lead against plain)
    assert C.top_k(others, 1, 0)[0] == (3, 7)                                        # (only the specific one against the rest)
    assert abs(others[3, 9]) < 4 < others[3, 7]
    assert abs(C.t_map(D, "b", rows, "others")[3, 9]) < 4


def test_null_threshold_sits_above_noise_and_below_the_planted_neuron():
    D = planted()
    rows = np.arange(40)
    t = C.t_map(D, "a", rows, "others")
    thr = C.null_threshold(D, "a", rows, "others", min_layer=0, flips=200)
    assert 3 < thr < t[3, 7]
    assert int((np.abs(t) > thr).sum()) == 1


def test_the_null_ignores_layers_below_min_layer():
    D = planted()
    rows = np.arange(40)
    low = C.null_threshold(D, "a", rows, "others", min_layer=4, flips=100)
    assert low < C.null_threshold(D, "a", rows, "others", min_layer=0, flips=100) + 1e-9
    assert low < 6


def test_replication_counts_the_neurons_that_hold_on_other_topics():
    D = planted(n=60)
    train, held = np.arange(40), np.arange(40, 60)
    t_tr, t_he = C.t_map(D, "a", train, "others"), C.t_map(D, "a", held, "others")
    thr = C.null_threshold(D, "a", train, "others", min_layer=0)
    assert C.replication(t_tr, t_he, thr, min_layer=0) == (1, 1)
    flipped = -t_he                                                                    # (the held-out sign disagrees)
    assert C.replication(t_tr, flipped, thr, min_layer=0) == (1, 0)
    assert C.replication(t_tr, t_he, thr, min_layer=4) == (0, 0)                      # (the planted neuron is in layer 3)


def test_flat_ranks_and_top_k_agree():
    a = np.zeros((4, 5))
    a[2, 3], a[3, 1], a[0, 0] = -9.0, 5.0, 100.0
    assert C.top_k(a, 2, min_layer=1) == [(2, 3), (3, 1)]
    r = C.flat_ranks(a, min_layer=1)
    assert r[2, 3] == 1 and r[3, 1] == 2 and r[0, 0] == 0


def test_load_diffs_keeps_only_the_pairs_that_were_measured(tmp_path):
    n = 6
    for r in ("a", "b"):
        np.save(tmp_path / f"{r}.npy", np.ones((n, 2, 3), np.float16))
    (tmp_path / "meta.json").write_text(json.dumps({"n": n, "registers": ["a", "b"], "done": 4,
                                                    "splits": ["train"] * n, "topics": list("uvwxyz")}))
    D, meta = C.load_diffs(str(tmp_path))
    assert D["a"].shape == (4, 2, 3) and meta["topics"] == list("uvwx") and len(meta["splits"]) == 4


def test_mean_activations_skip_the_first_position_and_leave_the_tap_clean():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from identity import neurons

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.taps = [neurons.MLPTap(MLP()), neurons.MLPTap(MLP())]

        def __call__(self, ids):
            x = self.emb(ids)
            for t in self.taps:
                x = x + t(x)
            return x

    model = Model()
    ids = [3, 1, 4, 1, 5]
    got = neurons.mean_activations(model, model.taps, ids)
    assert got.shape == (2, 6)
    x = model.emb(mx.array([ids]))
    m = model.taps[0].mlp
    h = nn.silu(m.gate_proj(x)) * m.up_proj(x)
    assert np.allclose(got[0], np.array(h[0, 1:].mean(axis=0)), atol=1e-5)
    assert all(not t.record_mean and t.mean is None for t in model.taps)
    with pytest.raises(AssertionError):
        neurons.mean_activations(model, model.taps, [3])
