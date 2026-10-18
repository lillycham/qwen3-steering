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
    real, chance = probe.cell_accuracies(planted(shift=6.0), rng)
    assert real.min() > 0.9 and chance.mean() < 0.6
    real, _ = probe.cell_accuracies(planted(shift=0.0), rng)
    assert real.mean() < 0.65                                             # (about 1/3, with a small test set)


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
