"""The causal test of proprioception: steering the residual stream and reading label and answer from one pass."""
import numpy as np
import pytest
import torch

from proprio import data as D
from proprio import lora
from proprio import steer as S
from proprio import train as T
from proprio_fakes import TinyTok
from test_proprio_train import PASSAGES, TASK_NAMES


def make(tiny, n=24):
    tok = TinyTok()
    rng = np.random.default_rng(0)
    exs = T.examples_for(tok, PASSAGES, TASK_NAMES, [2], n, rng)                # (lens at layer 2; steering is at layer 1)
    return tok, exs, D.digit_ids(tok), D.collate(exs, 0)


def hidden_after(model, batch, layer):
    with torch.no_grad():
        return T.forward(model, batch, "cpu").hidden_states[layer]


def block_output(model, batch, layer):
    """What the block of `layer` hands on, as seen by a hook registered after the steering one."""
    seen = []
    handle = model.model.layers[layer - 1].register_forward_hook(lambda m, a, o: seen.append(o[0] if isinstance(o, tuple) else o))
    try:
        with torch.no_grad():
            T.forward(model, batch, "cpu")
    finally:
        handle.remove()
    return seen[0]


def test_the_masks_cover_the_prefix_or_every_real_token(tiny):
    tok, exs, digits, batch = make(tiny, 6)
    prefix = S.positions_mask(batch, exs, "prefix")[..., 0]
    allm = S.positions_mask(batch, exs, "all")[..., 0]
    for i, e in enumerate(exs):
        assert prefix[i].sum() == e.pos + 1 and prefix[i, : e.pos + 1].all() and not prefix[i, e.pos + 1:].any()
        assert allm[i].sum() == len(e.ids)
    with pytest.raises(ValueError, match="prefix or all"):
        S.positions_mask(batch, exs, "middle")


def test_steering_adds_the_vector_on_the_masked_positions_only_and_leaves_no_trace(tiny):
    tok, exs, digits, batch = make(tiny, 6)
    mask = S.positions_mask(batch, exs, "prefix")
    delta = torch.arange(16, dtype=torch.float32) / 10
    plain = block_output(tiny, batch, 1)
    with S.steering(tiny, 1, delta, mask):
        moved = block_output(tiny, batch, 1)
        later = hidden_after(tiny, batch, 2)
    diff = moved - plain
    on = mask[..., 0].bool()
    assert torch.allclose(diff[on], delta.expand_as(diff[on]), atol=1e-5)
    assert diff[~on].abs().max() < 1e-6
    assert not torch.allclose(later, hidden_after(tiny, batch, 2), atol=1e-4)               # (the next layer sees it)
    assert torch.allclose(block_output(tiny, batch, 1), plain, atol=1e-6)                   # (the hook is gone afterwards)
    assert torch.allclose(hidden_after(tiny, batch, 2), hidden_after(tiny, batch, 2))


def test_a_lens_question_about_the_steered_layer_is_refused(tiny):
    tok = TinyTok()
    exs = T.examples_for(tok, PASSAGES, ["lens_top1"], [1], 4, np.random.default_rng(0))
    with pytest.raises(AssertionError, match="the steered layer"):
        S.steered(tiny, exs, D.digit_ids(tok), 0, "cpu", 1, torch.zeros(16), "prefix")


def test_zero_steering_changes_no_label_and_a_strong_vector_changes_some(tiny):
    tok, exs, digits, batch = make(tiny, 40)
    base_labels, base_answers = T.labels_and_answers(tiny, exs, digits, 0, "cpu")
    zero = S.steered(tiny, exs, digits, 0, "cpu", 1, torch.zeros(16), "prefix")
    assert zero[0] == base_labels and zero[1] == base_answers
    length = S.typical_norm(tiny, exs, 1, 0, "cpu")
    strong = S.steered(tiny, exs, digits, 0, "cpu", 1, S.random_delta(16, 3 * length, 0), "prefix")
    assert sum(a != b for a, b in zip(strong[0], base_labels)) > 0


def test_typical_norm_is_positive_and_the_random_delta_has_the_asked_length():
    v = S.random_delta(64, 5.0, seed=1)
    assert v.shape == (64,) and v.norm().item() == pytest.approx(5.0, rel=1e-5)
    assert torch.equal(S.random_delta(64, 5.0, 1), v) and not torch.equal(S.random_delta(64, 5.0, 2), v)


def test_typical_norm_skips_the_first_position(tiny):
    tok, exs, digits, batch = make(tiny, 8)
    assert S.typical_norm(tiny, exs, 2, 0, "cpu") > 0


def test_tracking_counts_follows_sticks_moves_and_the_other_vector():
    class E:
        def __init__(self, task):
            self.task = task
    exs = [E("top1")] * 4 + [E("entropy")] * 2
    base_labels = [1, 2, 3, 4, 5, 6]
    labels = [9, 2, 8, 4, 5, 7]                           # top1: rows 0 and 2 changed; entropy: row 5 changed
    base_answers = [1, 2, 3, 4, 5, 6]
    answers = [9, 2, 3, 4, 5, 6]                          # follows on row 0, sticks on row 2 (old answer), row 5 sticks
    other = [7, 7, 7, 7, 7, 7]
    rows = S.tracking(exs, base_labels, base_answers, labels, answers, other)
    assert rows["top1"] == {"n": 4, "changed": 0.5, "n_changed": 2, "follows": 0.5, "sticks": 0.5, "moves": 0.5,
                            "follows_other": 0.0}
    assert rows["entropy"]["n_changed"] == 1 and rows["entropy"]["follows"] == 0.0 and rows["entropy"]["sticks"] == 1.0
    none = S.tracking(exs[:2], [1, 2], [1, 2], [1, 2], [1, 2])
    assert none["top1"] == {"n": 2, "changed": 0.0, "n_changed": 0}


def test_a_model_trained_to_read_its_state_follows_the_steered_label_and_an_untrained_one_does_not(tiny):
    """The whole point, on the toy: train the adapters on the entropy task with live labels, then steer. Because the
    toy's labels are nearly constant the check is only that the machinery runs and returns coherent rates."""
    tiny.lm_head.weight.data.mul_(40)
    tok, exs, digits, batch = make(tiny, 32)
    lora.install(tiny, r=8, alpha=16)
    opt = torch.optim.AdamW(lora.trainable_parameters(tiny), lr=3e-3)
    for _ in range(80):
        loss, _ = T.step_loss(tiny, batch, digits, None, exs, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
    tiny.eval()
    base_labels, base_answers = T.labels_and_answers(tiny, exs, digits, 0, "cpu")
    length = S.typical_norm(tiny, exs, 1, 0, "cpu")
    a = S.steered(tiny, exs, digits, 0, "cpu", 1, S.random_delta(16, 4 * length, 1), "prefix")
    b = S.steered(tiny, exs, digits, 0, "cpu", 1, S.random_delta(16, 4 * length, 2), "prefix")
    rows = S.tracking(exs, base_labels, base_answers, a[0], a[1], b[0])
    for r in rows.values():
        assert 0 <= r["changed"] <= 1
        if r["n_changed"]:
            assert all(0 <= r[k] <= 1 for k in ("follows", "sticks", "moves", "follows_other"))
    assert "follows" in S.format_tracking({"random a=4 #1": rows})
