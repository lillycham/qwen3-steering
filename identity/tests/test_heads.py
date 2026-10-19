"""The head mask, the induction score's bookkeeping and the helpers. No model needed. Runs in `nix develop .#mlx`."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from identity import heads as H


def identity_proj(n):
    lin = nn.Linear(n, n, bias=False)
    lin.weight = mx.eye(n)
    return lin


def tap(n_heads=3, head_dim=2):
    return H.HeadTap(identity_proj(n_heads * head_dim), n_heads, head_dim)


def test_keep_vector_zeroes_only_the_chosen_heads_slices():
    v = H.keep_vector(3, 2, {1})
    assert v.tolist() == [1, 1, 0, 0, 1, 1]
    assert H.keep_vector(3, 2, set()).tolist() == [1] * 6
    with pytest.raises(ValueError):
        H.keep_vector(3, 2, {3})


def test_an_ablated_head_becomes_zero_or_its_fill_and_the_others_pass_through():
    t = tap()
    x = mx.arange(12, dtype=mx.float32).reshape(1, 2, 6) + 1
    assert np.array_equal(np.array(t(x)), np.array(x))                      # (nothing ablated: untouched)
    t.keep = mx.array(H.keep_vector(3, 2, {1}))
    out = np.array(t(x))
    assert np.all(out[..., 2:4] == 0) and np.array_equal(out[..., :2], np.array(x)[..., :2])
    t.fill = mx.array([9.0] * 6)
    assert np.all(np.array(t(x))[..., 2:4] == 9.0) and np.all(np.array(t(x))[..., 4:] == np.array(x)[..., 4:])


def test_recording_keeps_the_mean_output_over_positions_after_the_first():
    t = tap(1, 2)
    t.record = True
    t(mx.array([[[100.0, 100.0], [1.0, 3.0], [3.0, 5.0]]]))
    t(mx.array([[[100.0, 100.0], [2.0, 4.0]]]))
    assert np.allclose(np.array(t.total / t.count), [2.0, 4.0]) and t.count == 3


def fake_model(layers=2, n_heads=3, head_dim=2):
    def layer():
        attn = SimpleNamespace(o_proj=identity_proj(n_heads * head_dim), n_heads=n_heads,
                               q_norm=SimpleNamespace(weight=mx.ones(head_dim)))
        return SimpleNamespace(self_attn=attn)
    return SimpleNamespace(model=SimpleNamespace(layers=[layer() for _ in range(layers)]))


def test_install_wraps_each_o_proj_once_and_set_heads_clears_what_it_does_not_name():
    model = fake_model()
    taps = H.install(model)
    assert H.install(model)[0] is taps[0] and (taps[0].n_heads, taps[0].head_dim) == (3, 2)
    H.set_heads(taps, {(0, 1), (1, 2)}, np.full((2, 6), 7.0, np.float32))
    assert taps[0].keep.tolist() == [1, 1, 0, 0, 1, 1] and taps[1].keep.tolist() == [1, 1, 1, 1, 0, 0]
    assert taps[0].fill is not None
    H.set_heads(taps, {(0, 0)})
    assert taps[1].keep is None and taps[0].fill is None                      # (zero ablation: no fill)
    H.set_heads(taps, None)
    assert all(t.keep is None for t in taps)
    with pytest.raises(ValueError):
        H.set_heads(taps, {(5, 0)})


def test_repeated_batch_shows_each_random_sequence_twice_within_the_range():
    batch = H.repeated_batch(np.random.default_rng(0), 3, 8, lo=10, hi=20)
    assert len(batch) == 3 and all(len(s) == 16 and s[:8] == s[8:] and 10 <= min(s) and max(s) < 20 for s in batch)
    assert batch == H.repeated_batch(np.random.default_rng(0), 3, 8, lo=10, hi=20)


def test_copy_losses_measure_the_second_copy_from_the_positions_where_it_can_be_predicted():
    L, V = 6, 11
    ids = np.array(H.repeated_batch(np.random.default_rng(1), 2, L, lo=0, hi=V))
    logits = np.zeros((2, 2 * L, V), np.float32)
    for b in range(2):
        for j in range(L, 2 * L - 1):                                        # a perfect copier from position L on
            logits[b, j, ids[b, j + 1]] = 30.0
    first, second = H.copy_losses(mx.array(logits), ids, L)
    assert second == pytest.approx(0.0, abs=1e-3)                             # (position L-1 must not be counted)
    assert first == pytest.approx(np.log(V), abs=1e-3)


def test_top_heads_and_layer_totals():
    e = np.zeros((3, 4), np.float32)
    e[2, 1], e[0, 3], e[1, 0] = 0.9, 0.5, 0.01
    assert H.top_heads(e, 2) == [(2, 1), (0, 3)]
    assert H.layer_totals(e, 0.02).tolist() == pytest.approx([0.5, 0.0, 0.9])
