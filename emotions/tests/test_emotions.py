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


def test_steering_vectors_and_readout():
    pytest.importorskip("mlx")
    from emotions import serve
    acts, labels = synthetic()
    d = geometry.fit(acts, labels, ["fear", "joy"])
    assert serve.steering_vectors(d, {"fear": 0, "anger": 2}, [1, 2]) is None     # (zero and unknown emotions add nothing)
    v = serve.steering_vectors(d, {"fear": -2, "joy": 1}, [1, 2])
    assert v is not None and set(v) == {1, 2}
    want = -2 * d.sigma[0, 2] * d.vectors[0, 2] + d.sigma[1, 2] * d.vectors[1, 2]
    assert np.allclose(v[2], want)
    z = serve.readout(d, acts[0], [1, 2])
    assert set(z) == {"fear", "joy"}
    assert abs(z["fear"] - d.project(acts[:1])[0, 0, 1:3].mean()) < 1e-3


def guard(thinking=True):
    pytest.importorskip("mlx")
    from emotions.serve import ThinkGuard
    # tokens: 1 = <think>, 2 = </think>, 9 = end, 0 = blank, anything else = a word
    return ThinkGuard(1, 2, {9}, lambda t: t == 0, thinking)


def test_guard_closes_reasoning_on_end():
    g = guard()
    assert [g.feed(t) for t in (1, 5, 6)] == [1, 5, 6] and g.ban_end         # (inside the reasoning: end banned)
    assert g.feed(9) == 2 and g.closed == 1                                   # (an end that slips through closes it)
    assert g.ban_end                                                          # (and no answer yet)
    assert g.feed(0) == 0 and g.ban_end                                       # (blank is not an answer)
    assert g.feed(7) == 7 and not g.ban_end                                   # (a word is)
    assert g.feed(9) == 9


def test_guard_normal_flow_and_no_thinking():
    g = guard()
    assert [g.feed(t) for t in (1, 5, 2, 0, 7, 9)] == [1, 5, 2, 0, 7, 9] and g.closed == 0
    g = guard(thinking=False)
    assert g.ban_end and g.feed(7) == 7 and not g.ban_end
    g = guard()
    assert g.feed(7) == 7 and not g.ban_end                                   # (a reply that skips <think>)


def test_guard_closes_when_the_model_wants_to_end():
    g = guard()
    assert g.feed(1) == 1
    assert g.feed(5, wants_end=True) == 2 and g.closed == 1 and g.ban_end   # (top choice was the end: close now)
    assert g.feed(7) == 7 and not g.ban_end


def test_clean_turn():
    pytest.importorskip("mlx")
    from emotions.serve import clean_turn
    assert clean_turn({"role": "assistant", "content": "<think>\nhm\n</think>\n\nHello."})["content"] == "Hello."
    assert clean_turn({"role": "assistant", "content": "<think>\nOh... I guess I was silly."})["content"] == "Oh... I guess I was silly."
    assert clean_turn({"role": "assistant", "content": "Plain."})["content"] == "Plain."
    assert clean_turn({"role": "user", "content": "<think> stays"})["content"] == "<think> stays"


def test_merge():
    acts, labels = synthetic()
    a = geometry.fit(acts, labels, ["fear"])
    b = geometry.fit(acts, labels, ["joy"])
    m = geometry.merge([a, b])
    assert m.emotions == ["fear", "joy"] and m.vectors.shape == (2, 3, 8)
    assert np.allclose(m.project(acts)[:, 1], b.project(acts)[:, 0])


def test_cap_cache_limits_the_mlx_buffer_cache_to_a_gigabyte():
    mx = pytest.importorskip("mlx.core")
    from emotions import run
    before = mx.set_cache_limit(1 << 40)                      # (set_cache_limit returns the limit it replaces)
    try:
        run.cap_cache()
        assert mx.set_cache_limit(before) == run.CACHE_LIMIT == 1 << 30
    finally:
        mx.set_cache_limit(before)
