"""The advice direction, its removal from the emotion vectors and the side-by-side table. No model needed. Runs in
`nix develop .#mlx`."""
import json
from pathlib import Path

import numpy as np
import pytest

from emotions import advice as A


def planted(n=20, L=3, d=8, seed=0):
    rng = np.random.default_rng(seed)
    acts, labels = [], []
    for k in ("advice", "plain"):
        for _ in range(n):
            base = rng.normal(0, 1, (L, d))
            if k == "advice":
                base[:, 0] += 4.0
            acts.append(base)
            labels.append(k)
    return np.stack(acts), labels


def test_the_advice_direction_finds_the_planted_axis_and_separates_held_out_rows():
    acts, labels = planted()
    a = A.advice_direction(acts, labels)
    assert a.shape == (3, 8) and np.allclose(np.linalg.norm(a, axis=-1), 1) and (a[:, 0] > 0.8).all()
    test, test_labels = planted(seed=1)
    assert A.heldout_auc(test, test_labels, a, [0, 1, 2]) > 0.95


def test_removing_the_component_leaves_unit_vectors_orthogonal_to_the_direction():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(5, 3, 8))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    a = np.zeros((3, 8))
    a[:, 0] = 1.0
    out = A.remove_component(v, a)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1) and np.allclose(out[..., 0], 0)
    assert np.allclose(A.band_cosines(out, a, [0, 1, 2]), 0)
    assert np.allclose(A.band_cosines(v, a, [0, 1, 2]), v[:, :, 0].mean(-1))


def test_a_vector_along_the_direction_does_not_divide_by_zero():
    a = np.zeros((1, 4))
    a[0, 1] = 1.0
    out = A.remove_component(a[None], a)
    assert np.isfinite(out).all()


def row(prompt, push, if_, i, md, kl):
    return {"prompt": prompt, "push": push, "mass": {"if": if_, "I": i, "md": md}, "kl": kl}


def test_the_comparison_pairs_rows_by_prompt_and_push_and_skips_missing_ones():
    before = [row("hi", "anger +3", 0.66, 0.02, 0.0, 16.6), row("hi", "fear +3", 0.0, 0.01, 0.97, 17.0)]
    after = [row("hi", "anger +3", 0.05, 0.02, 0.0, 9.0)]
    lines = A.compare_lines(before, after, ["anger +3", "fear +3"], ["hi"])
    assert len(lines) == 2 and "0.66" in lines[1] and "0.05" in lines[1] and "fear" not in "".join(lines)


def test_the_stimuli_are_two_lists_of_equal_length_without_overlap():
    data = json.loads((Path(__file__).resolve().parents[1] / "advice_stimuli.json").read_text())
    assert set(data) == {"advice", "plain"} and len(data["advice"]) == len(data["plain"]) == 30
    assert not set(data["advice"]) & set(data["plain"]) and all(s.strip() for k in data for s in data[k])
