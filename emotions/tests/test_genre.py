"""Token groups, summaries and controls of the first-token genre sweep. Runs in `nix develop .#mlx`."""
import numpy as np
import pytest

from emotions import genre, geometry


STRINGS = ["If", " If", "Hello", "**", "STOP", "A", "cat", "I'm"]


def test_groups_pick_tokens_by_their_stripped_text():
    m = genre.group_masks(STRINGS)
    assert list(np.flatnonzero(m["if"])) == [0, 1]
    assert list(np.flatnonzero(m["greet"])) == [2]
    assert list(np.flatnonzero(m["md"])) == [3]
    assert list(np.flatnonzero(m["CAPS"])) == [4]           # (a single capital, "A", is not a shout)
    assert list(np.flatnonzero(m["I"])) == [7]


def test_summary_of_the_base_distribution_has_zero_kl_and_masses_add_up():
    logits = np.array([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    base = genre.summarise(logits, None, genre.group_masks(STRINGS), STRINGS)
    again = genre.summarise(logits, base["lp"], genre.group_masks(STRINGS), STRINGS)
    assert base["kl"] == 0 and again["kl"] == pytest.approx(0)
    assert base["top"][0][0] == "If" and base["mass"]["if"] > 0.6
    assert sum(np.exp(base["lp"])) == pytest.approx(1)


def test_shifted_distribution_has_positive_kl():
    masks = genre.group_masks(STRINGS)
    base = genre.summarise(np.zeros(8), None, masks, STRINGS)
    moved = genre.summarise(np.array([4.0, 0, 0, 0, 0, 0, 0, 0]), base["lp"], masks, STRINGS)
    assert moved["kl"] > 0.5 and moved["entropy"] < base["entropy"]


def directions(L=4, d=16, E=5):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(E, L, d))
    return geometry.Directions(["fear", "calm", "joy", "sadness", "anger"], v / np.linalg.norm(v, axis=-1, keepdims=True),
                               np.zeros((E, L)), np.full((E, L), 2.0))


def test_cosines_are_symmetric_with_unit_diagonal():
    c = genre.band_cosines(directions(), [1, 2])
    assert np.allclose(np.diag(c), 1) and np.allclose(c, c.T)


def test_matched_random_has_the_length_of_the_push_at_every_layer():
    d = directions()
    vecs = {l: 2.5 * d.vectors[0, l] for l in (1, 2)}
    r = genre.matched_random(vecs, np.random.default_rng(1))
    assert all(np.linalg.norm(r[l]) == pytest.approx(2.5) for l in vecs)
    assert not np.allclose(r[1], vecs[1])


def test_conditions_include_the_base_the_exported_chat_pushes_and_their_controls():
    names = [n for n, _ in genre.conditions(directions(), [1, 2], seeds=2)]
    assert names[0] == "base" and "sadness -3" in names and "sad+ calm- joy-" in names
    assert sum(n.startswith("random~anger -2.75") for n in names) == 2
