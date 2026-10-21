"""Live labels of proprioception questions, on the tiny random Qwen3 of conftest."""
import math

import pytest
import torch

from proprio import labels as P


def run(model, ids):
    with torch.no_grad():
        return model(ids, output_hidden_states=True)


def test_lens_after_the_last_layer_is_the_real_output_and_earlier_layers_differ(tiny, ids):
    out = run(tiny, ids)
    n = len(out.hidden_states) - 1
    assert n == 3
    last = P.lens_row(tiny, out.hidden_states, n, row=1, pos=4)
    assert torch.allclose(last, out.logits[1, 4], atol=1e-5)
    early = P.lens_row(tiny, out.hidden_states, 1, row=1, pos=4)
    assert early.shape == last.shape and not torch.allclose(early, last, atol=1e-3)
    with pytest.raises(AssertionError, match="outside 1..3"):
        P.lens_row(tiny, out.hidden_states, 4, 0, 0)


def test_entropy_of_a_uniform_and_of_a_certain_distribution():
    uniform = torch.log_softmax(torch.zeros(64), -1)
    assert P.entropy_bits(uniform).item() == pytest.approx(6.0, abs=1e-5)
    certain = torch.log_softmax(torch.tensor([50.0] + [0.0] * 63), -1)
    assert P.entropy_bits(certain).item() < 1e-6


def test_buckets_follow_the_edges():
    v = torch.tensor([0.0, 0.25, 0.26, 1.0, 1.01, 5.9, 6.0, 6.01, 9.0])
    assert P.bucket(v).tolist() == [0, 0, 1, 2, 3, 6, 6, 7, 7]
    assert P.N_BUCKETS == 8


def test_label_of_gives_the_kth_token_and_the_entropy_bucket():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    assert P.label_of(P.TASKS["top1"], logits) == 1
    assert P.label_of(P.TASKS["top3"], logits) == 2                        # (5, 4, then 3)
    sharp = torch.tensor([30.0, 0.0, 0.0, 0.0])
    flat = torch.zeros(1024)                                              # (10 bits)
    assert P.label_of(P.TASKS["entropy"], sharp) == 0 and P.label_of(P.TASKS["entropy"], flat) == 7


def test_the_alternative_label_is_the_strongest_prediction_that_is_not_the_token_that_followed():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])              # ranking: token 1, then 3, then 2
    alt = P.TASKS["alt"]
    assert P.label_of(alt, logits, nxt=7) == 1                      # (the real next token is elsewhere: the top one)
    assert P.label_of(alt, logits, nxt=1) == 3                      # (the top one is the real next token: the second)
    assert P.label_of(alt, logits, nxt=3) == 1
    assert P.TASKS["alt"].kind == "token" and P.TASKS["lens_alt"].source == "lens"
    assert P.TASKS["top1"].leaky and not P.TASKS["top3"].leaky and not alt.leaky and not P.TASKS["entropy"].leaky


def test_read_labels_passes_the_real_next_token_to_the_alternative_tasks(tiny, ids):
    out = run(tiny, ids)
    top = int(out.logits[0, 3].argmax())
    same = P.read_labels(tiny, out, [P.Spec("alt", 0, 3, 0, nxt=top)])[0]
    other = P.read_labels(tiny, out, [P.Spec("alt", 0, 3, 0, nxt=(top + 1) % 97)])[0]
    assert same == int(out.logits[0, 3].topk(2).indices[1]) and other == top


def test_task_kinds_and_targets():
    assert P.TASKS["top3"].kind == "token" and P.TASKS["lens_entropy"].kind == "bucket"
    digits = tuple(range(100, 108))
    assert P.target_id(P.TASKS["top1"], 42, digits) == 42 and P.target_id(P.TASKS["entropy"], 3, digits) == 103


def test_a_label_does_not_depend_on_what_comes_after_the_quoted_token(tiny, ids):
    """The label at position p must be the same from the whole sequence (question and answer included) as from the
    prefix that ends at p: that is what lets it be read from the same forward pass as the loss."""
    full = run(tiny, ids)
    p = 5
    prefix = run(tiny, ids[:, : p + 1])
    for name in P.TASKS:
        for layer in (1, 2, 3):
            if P.TASKS[name].source == "logits" and layer != 1:
                continue
            for row in (0, 1):
                s = P.Spec(name, row, p, layer)
                assert P.read_labels(tiny, full, [s]) == P.read_labels(tiny, prefix, [P.Spec(name, row, p, layer)])


def test_read_labels_matches_a_by_hand_computation(tiny, ids):
    out = run(tiny, ids)
    specs = [P.Spec("top1", 0, 3), P.Spec("top3", 1, 7), P.Spec("entropy", 0, 2), P.Spec("lens_top1", 1, 6, 2),
             P.Spec("lens_entropy", 0, 4, 1)]
    got = P.read_labels(tiny, out, specs)
    lg = out.logits
    assert got[0] == int(lg[0, 3].argmax()) and got[1] == int(lg[1, 7].topk(3).indices[2])
    assert got[2] == int(P.bucket(P.entropy_bits(torch.log_softmax(lg[0, 2].float(), -1))))
    early = tiny.lm_head(tiny.model.norm(out.hidden_states[2][1, 6]))
    assert got[3] == int(early.argmax())
    assert all(isinstance(x, int) for x in got)
