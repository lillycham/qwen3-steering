"""Emotion geometry and turn spans. Runs in `nix develop .#mlx` (numpy, and the Qwen3 tokenizer from the HF cache)."""
import numpy as np
import pytest

from emotions import geometry


def synthetic(n=30, L=3, d=8, seed=0):
    rng = np.random.default_rng(seed)
    dirs = {"fear": np.eye(d)[0], "joy": np.eye(d)[1], "neutral": np.zeros(d)}
    acts, labels = [], []
    for e, v in dirs.items():
        for _ in range(n):
            acts.append(np.stack([3 * v * (l + 1) + rng.normal(0, 1, d) for l in range(L)]))
            labels.append(e)
    return np.stack(acts), labels


def test_fit_finds_planted_directions():
    acts, labels = synthetic()
    d = geometry.fit(acts, labels, ["fear", "joy"])
    assert d.vectors.shape == (2, 3, 8)
    assert np.allclose(np.linalg.norm(d.vectors, axis=-1), 1)
    assert d.vectors[0, 2, 0] > 0.85 and d.vectors[1, 2, 1] > 0.85      # (minus the others' means: not exactly axis-aligned)


def test_projection_is_z_scored_and_separates():
    acts, labels = synthetic()
    d = geometry.fit(acts, labels, ["fear", "joy"])
    z = d.project(acts)
    assert np.allclose(z.mean(0), 0, atol=1e-6) and np.allclose(z.std(0), 1, atol=1e-3)
    lab = np.array(labels)
    assert z[lab == "fear", 0, 2].mean() > 1 > z[lab != "fear", 0, 2].mean()


def test_auc_and_evaluate():
    assert geometry.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert geometry.auc(np.array([1.0]), np.array([1.0])) == 0.5
    acts, labels = synthetic()
    tr, te = geometry.split(labels)
    d = geometry.fit(acts[tr], [labels[i] for i in tr], ["fear", "joy"])
    aucs = geometry.evaluate(d, acts[te], [labels[i] for i in te])
    assert aucs["fear"][-1] > 0.95 and aucs["joy"][-1] > 0.95


def test_split_is_per_emotion():
    labels = ["a"] * 10 + ["b"] * 5
    tr, te = geometry.split(labels)
    assert [labels[i] for i in te] == ["a", "a", "b"] and len(tr) == 12


def test_save_load(tmp_path):
    acts, labels = synthetic()
    d = geometry.fit(acts, labels, ["fear", "joy"])
    d.save(str(tmp_path / "v.npz"))
    e = geometry.Directions.load(str(tmp_path / "v.npz"))
    assert e.emotions == ["fear", "joy"] and np.allclose(e.vectors, d.vectors)


def test_turn_spans_match_reply():
    pytest.importorskip("mlx")
    transformers = pytest.importorskip("transformers")
    from emotions import run
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    msgs = [{"role": "user", "content": "Hello there."}, {"role": "assistant", "content": "Hi! I'm Clover."},
            {"role": "user", "content": "Want some lettuce?"}, {"role": "assistant", "content": "Yes, please!"}]
    for i, reply in ((1, "Hi! I'm Clover."), (3, "Yes, please!")):
        ids, pre, a, b = run.turn(tok, msgs, i)
        assert tok.decode(ids[a:b]) == reply
        assert tok.decode(ids[:pre + 1]).endswith(tok.apply_chat_template(
            msgs[:i], add_generation_prompt=True, enable_thinking=False, tokenize=False)[-20:])
