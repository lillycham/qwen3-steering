"""The concept search's word lists, judging and candidate choice. Runs in `nix develop .#mlx`."""
import json
from pathlib import Path

import numpy as np
import pytest

from identity import concept as C


class FakeTok:
    """encode() gives one token for a word in the vocabulary, two for anything else."""
    vocab = {" rabbit": 11, " bunny": 12, " cat": 21}

    def encode(self, w, add_special_tokens=False):
        return [self.vocab[w]] if w in self.vocab else [900, 901]


def test_single_token_ids_keeps_only_single_tokens():
    ids, kept = C.single_token_ids(FakeTok(), (" rabbit", " bunnies", " bunny"))
    assert ids == [11, 12] and kept == [" rabbit", " bunny"]


def test_the_rabbit_judge_matches_rabbit_words_and_not_lookalikes():
    r = C.RABBIT
    for text in ("A bunny hopped by.", "I love rabbits!", "The March Hare", "a cottontail", "Bunnies are soft."):
        assert r.hit.search(text), text
    for text in ("I like carrots.", "A bunch of flowers.", "It is a rabbity thing.", "My name is Qwen."):
        assert not r.hit.search(text), text
    assert r.other.search("My favourite is the cat.") and r.other.search("Foxes and owls.")
    assert not r.other.search("A category of things.")
    assert C.rate(["a rabbit", "no", "bunny!", "dog"], r.hit) == 0.5 and C.rate([], r.hit) == 0.0


def test_candidate_union_signs_dedupes_and_keeps_the_largest_entry():
    topic = np.zeros((3, 5), np.float32)
    self_ = np.zeros((3, 5), np.float32)
    topic[0, 1], topic[1, 2] = 4.0, -3.0          # add neuron (0,1); subtract neuron (1,2)
    self_[0, 1] = -1.0                           # (0,1) again, with a smaller opposite entry: the larger one wins
    self_[2, 4] = 2.0
    out = C.candidate_union({"topic": topic, "self": self_}, per_set=1, aligned=[(2, 0, -0.3), (0, 1, -0.9)])
    by = {(l, n): s for l, n, s in out}
    assert by[(0, 1)] == 1.0 and by[(1, 2)] == -1.0 and by[(2, 4)] == 1.0
    assert by[(2, 0)] == -1.0                     # (from alignment: the sign of the cosine)
    assert len(out) == len(by)                    # (no duplicates)


def test_rabbit_spec_is_consistent():
    r = C.RABBIT
    held = set(r.topic_held) | set(r.self_held)
    used = {q for q, _ in r.topic_prompts + r.self_prompts}
    assert not held & used                       # (held-out questions are not the ones the gradient saw)
    assert not set(r.targets) & (set(r.topic_contrast) | set(r.self_contrast))
    assert all(prefix.strip() for _, prefix in r.topic_prompts + r.self_prompts)


def test_stimuli_file_has_rabbit_first_and_balanced_sets():
    data = json.loads((Path(__file__).parent.parent / "rabbit_stimuli.json").read_text())
    assert list(data)[0] == "rabbit"             # (identity.screen.candidates uses the first direction)
    assert set(data) == {"rabbit", "other animals", "neutral"} and all(len(v) == 30 for v in data.values())
    assert all(C.RABBIT.hit.search(s) for s in data["rabbit"])
    assert not any(C.RABBIT.hit.search(s) for s in data["other animals"] + data["neutral"])


def test_word_lists_are_mostly_single_tokens_in_the_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    _, targets = C.single_token_ids(tok, C.RABBIT.targets)
    _, topic = C.single_token_ids(tok, C.RABBIT.topic_contrast)
    _, own = C.single_token_ids(tok, C.RABBIT.self_contrast)
    assert " rabbit" in targets and len(targets) >= 3, targets
    assert len(topic) >= 6 and len(own) >= 4, (topic, own)
    ids = tok.encode(" rabbit", add_special_tokens=False)
    assert len(ids) == 1


def test_candidate_union_skips_early_layers_after_cutting_the_lists():
    d = np.zeros((4, 5), np.float32)
    d[0, 1], d[3, 2], d[2, 3] = 9.0, 1.0, -0.5       # the biggest entry is in layer 0, below the floor
    out = C.candidate_union({"topic": d}, per_set=1, aligned=[(1, 0, 0.2), (3, 4, 0.2)], min_layer=2)
    units = {(l, n) for l, n, _ in out}
    assert (0, 1) not in units and (1, 0) not in units
    assert {(3, 2), (2, 3), (3, 4)} <= units


def test_best_coherent_picks_the_strongest_push_that_keeps_the_model_coherent():
    trials = [{"k": 2, "topic": 1.0, "self": 0.1, "kl": 0.2}, {"k": 8, "topic": 3.0, "self": 0.5, "kl": 0.9},
              {"k": 32, "topic": 9.0, "self": 4.0, "kl": 5.0}]
    assert C.best_coherent(trials, "topic", 1.0)["k"] == 8              # (the k=32 push breaks the model)
    assert C.best_coherent(trials, "self", 1.0)["k"] == 8
    assert C.best_coherent(trials, "topic", 0.1) is None
    assert C.kl_np(np.log(np.array([0.5, 0.5])), np.log(np.array([0.5, 0.5]))) == pytest.approx(0)


def test_report_ranks_coherent_neurons_and_ignores_broken_ones(tmp_path, capsys):
    def row(kind, layer, neuron, rises, kls, aligned=False):
        return {"kind": kind, "layer": layer, "neuron": neuron, "aligned": aligned, "scale": 0.5,
                "trials": [{"k": k, "steer": 0.5 * k, "topic": r, "self": r / 2, "kl": kl}
                           for k, r, kl in zip((2, 8, 32), rises, kls)]}
    data = {"concept": "rabbit", "baseline": {"topic": -9.0, "self": -20.0}, "ks": [2, 8, 32], "max_kl": 1.0,
            "rows": [row("candidate", 20, 7, (0.5, 2.0, 6.0), (0.1, 0.4, 0.8)),
                     row("candidate", 3, 9, (1.0, 9.0, 20.0), (0.5, 3.0, 9.0)),        # (a breaker: huge rise, huge KL)
                     row("random", 11, 2, (0.0, 0.1, 0.0), (0.1, 0.2, 0.3))]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    C.report(str(p))
    out = capsys.readouterr().out
    first = [l for l in out.splitlines() if l.startswith("  L")]
    assert first[0].startswith("  L20.N7") and "6.00" in first[0] and "32x" in first[0]
    assert "random neurons: mean +0.10" in out


def test_peak_activations_skip_the_first_position():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from identity import neurons

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj = nn.Linear(3, 5, bias=False), nn.Linear(3, 5, bias=False)
            self.down_proj = nn.Linear(5, 3, bias=False)

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeMLP()

    tap = neurons.MLPTap(FakeMLP())
    x = mx.array(np.random.default_rng(0).normal(size=(1, 4, 3)).astype(np.float32))
    h = nn.silu(tap.mlp.gate_proj(x)) * tap.mlp.up_proj(x)
    want = np.abs(np.array(h))[0, 1:].max(0)

    class Model:
        def __call__(self, ids):
            return tap(x)

    peaks = neurons.peak_activations(Model(), [tap], [[1, 2, 3, 4]])
    assert peaks.shape == (1, 5) and np.allclose(peaks[0], want, atol=1e-5)
    assert tap.record_peak is False and tap.peak is None
