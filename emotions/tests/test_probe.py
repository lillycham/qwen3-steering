"""The probe's fitting and reporting, on synthetic activations. Runs in `nix develop .#mlx`."""
import json

import numpy as np

from emotions import introspect as I, probe


def planted(k=3, n=32, L=2, d=16, shift=3.0, seed=0):
    """[k, n, L, d]: class c has `shift` added along axis c-1 (class 0 is the unshifted one)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(k, n, L, d))
    for c in range(1, k):
        x[c, :, :, c - 1] += shift
    return x


def test_split_holds_out_every_fourth_trial():
    t = probe.split(8)
    assert t.tolist() == [False, False, False, True, False, False, False, True]


def test_probe_finds_a_planted_shift_and_not_a_missing_one():
    rng = np.random.default_rng(1)
    real, chance, held = probe.cell_accuracies(planted(shift=6.0), rng)
    assert real.min() > 0.9 and chance.mean() < 0.6 and held.min() > 0.85
    real, _, held = probe.cell_accuracies(planted(shift=0.0), rng)
    assert real.mean() < 0.65 and held.mean() < 0.65                                             # (about 1/3, with a small test set)


def test_probe_separates_two_clearly_shifted_classes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 16))
    y = np.repeat([0, 1], 20)
    test = np.tile(probe.split(20), 2)
    X[:, 0] += 8 * y
    assert probe.probe_accuracy(X, y, test, 2) == 1.0


def test_report_prints_the_probe_and_the_models_own_accuracy(tmp_path, capsys):
    rows = [{"variant": "all", "alpha": a, "layer": l, "acc": 0.5 + 0.1 * l, "acc_shuffled": 0.2}
            for a in (0.05, 1.0) for l in range(6)]
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"rows": rows, "alphas": [0.05, 1.0], "layers": [2, 4], "n": 8, "seed": 0,
                             "adapter": "none"}))
    probe.report(str(p))
    out = capsys.readouterr().out
    assert "emotions: probe accuracy" in out and "== all ==" in out and " 4*" in out and "best" in out and "shuf" in out
    mrows = [{"variant": "all", "kind": "steered", "truth": e, "alpha": 1.0, "trial": 0, "probs":
              {o: (0.9 if o == e else 0.02) for o in I.OPTIONS}, "mass": 1.0} for e in I.EMOTIONS]
    m = tmp_path / "model.json"
    m.write_text(json.dumps({"rows": mrows, "alphas": [1.0], "layers": [2, 4], "n": 1, "seed": 0, "adapter": "none"}))
    assert probe.model_accuracy(str(m), "all", 1.0) == 1.0 and probe.model_accuracy(str(m), "all", 0.05) is None
    probe.report(str(p), str(m))
    assert " model" in capsys.readouterr().out


def test_passage_folds_split_the_trials_by_passage():
    from emotions import introspect as I
    folds = probe.passage_folds(32)
    assert len(folds) == len(I.PASSAGES) == 3
    assert (sum(f.astype(int) for f in folds) == 1).all()                 # (every trial is in exactly one fold)
    assert [int(f.sum()) for f in folds] == [11, 11, 10]


def test_many_passages_fall_into_five_balanced_folds_with_no_trial_in_two():
    folds = probe.passage_folds(300, n_passages=300)
    assert len(folds) == 5 and [int(f.sum()) for f in folds] == [60] * 5
    assert (sum(f.astype(int) for f in folds) == 1).all()
    wrapped = probe.passage_folds(100, n_passages=40)                       # (more trials than passages: each is reused)
    assert len(wrapped) == 5 and (sum(f.astype(int) for f in wrapped) == 1).all()
    # a passage stays in one fold however often it is reused, so no fold shares a passage with another
    seen = {}
    for j, f in enumerate(wrapped):
        for i in range(100):
            if f[i]:
                assert seen.setdefault(i % 40, j) == j
    assert [f.tolist() for f in probe.passage_folds(32)] == [f.tolist() for f in probe.passage_folds(32, 3)]


def test_cell_accuracies_take_the_number_of_passages():
    rng = np.random.default_rng(1)
    real, chance, held = probe.cell_accuracies(planted(shift=6.0, n=40), rng, n_passages=40)
    assert real.min() > 0.9 and held.min() > 0.8


def test_report_prints_the_held_out_passage_row_when_it_is_there(tmp_path, capsys):
    rows = [{"variant": "prefix", "alpha": 1.0, "layer": l, "acc": 0.9, "acc_shuffled": 0.2, "acc_heldout": 0.7}
            for l in range(4)]
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"rows": rows, "alphas": [1.0], "layers": [1, 2], "n": 8, "seed": 0, "adapter": "none"}))
    probe.report(str(p))
    import re
    assert re.search(r"^ *held +0\.70$", capsys.readouterr().out, re.M)
    old = [{k: v for k, v in r.items() if k != "acc_heldout"} for r in rows]       # (a file from before this row existed)
    p.write_text(json.dumps({"rows": old, "alphas": [1.0], "layers": [1, 2], "n": 8, "seed": 0, "adapter": "none"}))
    probe.report(str(p))
    assert not re.search(r"^ *held +\d", capsys.readouterr().out, re.M)
