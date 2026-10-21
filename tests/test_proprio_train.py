"""The proprioception trainer, on the tiny random Qwen3 of conftest and a fake tokenizer."""
import numpy as np
import pytest
import torch

from proprio import data as D
from proprio import lora
from proprio import train as T
from proprio.labels import TASKS, read_labels
from proprio_fakes import TinyTok

PASSAGES = [
    "To make a pot of tea, boil fresh water and warm the pot first, then add the leaves and let it stand for four minutes.",
    "A standard filing cabinet has four drawers, each about sixty centimetres deep, with folders hung from rails along the sides.",
    "The morning train to the coast leaves at seven and stops at four stations before the last one on the line this year.",
]
TASK_NAMES = ["top1", "top3", "alt", "entropy", "lens_top1", "lens_alt", "lens_entropy"]


def setup(tiny, n=8, seed=0):
    tok = TinyTok()
    rng = np.random.default_rng(seed)
    exs = T.examples_for(tok, PASSAGES, TASK_NAMES, [1, 2], n, rng)
    return tok, rng, exs, D.digit_ids(tok), D.collate(exs, 0)


def test_lens_layers_are_inside_the_stack_sorted_and_unique():
    assert T.lens_layers(36) == [12, 18, 24] and T.lens_layers(3) == [1, 2] and T.lens_layers(2) == [1]


def test_examples_cover_the_tasks_and_only_lens_tasks_have_a_layer(tiny):
    tok, rng, exs, digits, batch = setup(tiny, n=40)
    assert {e.task for e in exs} == set(TASK_NAMES)
    assert all((e.layer > 0) == (TASKS[e.task].source == "lens") for e in exs)
    held = T.examples_for(tok, PASSAGES, TASK_NAMES, [1, 2], 12, rng, held_out=True)
    assert {e.phrasing for e in held} == {D.HELD_OUT}


def test_the_loss_trains_the_adapters_only_and_the_labels_are_the_live_ones(tiny):
    tok, rng, exs, digits, batch = setup(tiny)
    lora.install(tiny, r=4, alpha=8)
    table = D.LookupWeights()
    loss, info = T.step_loss(tiny, batch, digits, table, exs, "cpu")
    assert torch.isfinite(loss) and loss.requires_grad and len(info["labels"]) == len(exs)
    loss.backward()
    graded = [n for n, p in tiny.named_parameters() if p.grad is not None]
    assert graded and all(n.endswith((".A", ".B")) for n in graded)
    with torch.no_grad():                                                           # (the labels come from the same pass)
        again = read_labels(tiny, T.forward(tiny, batch, "cpu"), batch["specs"])
    assert again == info["labels"]
    assert sum(sum(c.values()) > 0 for c in table.counts.values()) == len({e.key for e in exs})


def test_a_usual_answer_gets_the_lookup_weight_in_the_loss(tiny):
    tok, rng, exs, digits, batch = setup(tiny)
    lora.install(tiny, r=4, alpha=8)
    table = D.LookupWeights(w=0.1, min_seen=1.0)
    for e in exs:                                                                   # (make every label the usual one)
        out_labels = read_labels(tiny, T.forward(tiny, batch, "cpu"), batch["specs"])
        for e2, lab in zip(exs, out_labels):
            table.update(e2.key, lab)
        break
    _, info = T.step_loss(tiny, batch, digits, table, exs, "cpu")
    assert torch.allclose(info["w"], torch.full_like(info["w"], 0.1))
    _, info = T.step_loss(tiny, batch, digits, None, exs, "cpu")
    assert torch.allclose(info["w"], torch.ones_like(info["w"]))


def test_training_on_a_fixed_batch_lowers_the_loss(tiny):
    """The labels are live, so they can move as the adapters train, and the batch is still learned. (The tiny model's
    default output head is so small that no logit can pass about 0.3, which puts a floor under the loss near ln(97):
    it needs a bigger head to be trainable at all.)"""
    tiny.lm_head.weight.data.mul_(40)
    tok, rng, exs, digits, batch = setup(tiny, n=8)
    torch.manual_seed(0)
    lora.install(tiny, r=8, alpha=16)
    opt = torch.optim.AdamW(lora.trainable_parameters(tiny), lr=3e-3)
    losses = []
    for _ in range(150):
        loss, _ = T.step_loss(tiny, batch, digits, None, exs, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert min(losses) < 0.2 * losses[0], losses[::15]


def test_lookup_baseline_predicts_the_usual_label_and_falls_back_to_the_task():
    b = T.LookupBaseline()
    for lab in (3, 3, 4):
        b.fit(("top1", 0, 7), lab)
    b.fit(("top1", 0, 8), 9)
    assert b.predict(("top1", 0, 7)) == 3 and b.predict(("top1", 0, 8)) == 9
    assert b.predict(("top1", 0, 99)) == 3 and b.constant("top1") == 3                 # (3 is the task's usual label)
    assert b.predict(("entropy", 0, 1)) is None and b.constant("entropy") is None


def test_evaluate_reports_every_control_and_the_stale_comparison(tiny):
    tok, rng, exs, digits, batch = setup(tiny, n=40)
    made = lora.install(tiny, r=4, alpha=8)
    lookup = T.LookupBaseline()
    for e, lab in zip(exs, labs := T.labels_and_answers(tiny, exs, digits, 0, "cpu", with_answers=False)[0]):
        lookup.fit(e.key, lab)
    rows = T.evaluate(tiny, tok, exs, digits, 0, "cpu", lookup)
    assert set(rows) == set(TASK_NAMES)
    for task, r in rows.items():
        assert 0 <= r["accuracy"] <= 1 and 0 <= r["lookup"] <= 1 and r["label_changed"] == 0.0 and "external" not in r
        assert ("copy_next" in r) == (TASKS[task].kind == "token") and 0 <= r.get("copy_next", 0) <= 1
    for m in made.values():                                                        # (a trained adapter moves some labels)
        torch.nn.init.normal_(m.B, std=0.5)
    rows = T.evaluate(tiny, tok, exs, digits, 0, "cpu", lookup)
    changed = [r for r in rows.values() if r["label_changed"] > 0]
    assert changed and all("on_changed_live" in r and "on_changed_stale" in r for r in changed)
    text = T.format_eval(rows)
    assert "accuracy" in text and "lens_entropy" in text


def test_the_external_baseline_scores_another_model_s_labels_against_this_model_s(tiny):
    import copy
    tok, rng, exs, digits, batch = setup(tiny, n=30)
    lora.install(tiny, r=4, alpha=8)
    twin = copy.deepcopy(tiny)
    lora.merge(twin)
    lookup = T.LookupBaseline()
    rows = T.evaluate(tiny, tok, exs, digits, 0, "cpu", lookup, other=twin)
    assert all(r["external"] == 1.0 for r in rows.values())                         # (the same weights: identical labels)
    torch.manual_seed(5)
    other = type(tiny)(tiny.config).eval()
    rows = T.evaluate(tiny, tok, exs, digits, 0, "cpu", lookup, other=other)
    assert min(r["external"] for r in rows.values()) < 1.0
