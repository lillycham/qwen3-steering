"""Scoring and prompt layout for the introspection test. Runs in `nix develop .#mlx`."""
import numpy as np
import pytest

from emotions import geometry, introspect as I


def test_trials_are_shuffled_permutations_and_repeatable():
    ts = I.make_trials(30, seed=3)
    assert all(sorted(t.order) == sorted(I.OPTIONS) for t in ts)
    assert len({tuple(t.order) for t in ts}) > 10                         # (not all the same order)
    assert [t.order for t in ts] == [t.order for t in I.make_trials(30, seed=3)]
    assert {t.passage for t in ts} == {0, 1, 2}


def test_user_message_lists_options_in_trial_order():
    t = I.Trial(["none", "joy", "fear", "calm", "anger", "sadness"], 1)
    msg, passage = I.user_message(t)
    assert passage in msg and msg.index(passage) < msg.index("A. none")
    assert "B. joy" in msg and "F. sadness" in msg


def test_option_probs_map_letters_back_to_options():
    order = ["none", "joy", "fear", "calm", "anger", "sadness"]
    p = I.option_probs(np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]), order)
    assert I.pick(p) == "joy" and abs(sum(p.values()) - 1) < 1e-9


def test_kl_is_zero_for_equal_and_positive_otherwise():
    lp = np.log(np.array([0.5, 0.3, 0.2]))
    lq = np.log(np.array([0.2, 0.3, 0.5]))
    assert I.kl(lp, lp) == pytest.approx(0)
    assert I.kl(lp, lq) == pytest.approx(0.5 * np.log(2.5) + 0.2 * np.log(0.4))


def directions(L=3, d=8):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, L, d))
    return geometry.Directions(I.EMOTIONS, v / np.linalg.norm(v, axis=-1, keepdims=True), np.zeros((5, L)),
                               np.full((5, L), 2.0))


def test_steer_and_random_vectors_have_the_same_length():
    d = directions()
    v = I.steer_vectors(d, "joy", 1.5, [1, 2])
    r = I.random_vectors(d, 1.5, [1, 2], np.random.default_rng(1))
    assert set(v) == set(r) == {1, 2}
    assert np.allclose(v[1], 1.5 * 2.0 * d.vectors[I.EMOTIONS.index("joy"), 1])
    assert np.linalg.norm(v[1]) == pytest.approx(np.linalg.norm(r[1])) == pytest.approx(3.0)


def row(kind, truth, favourite, alpha=1.0, variant="all"):
    probs = {o: 0.02 for o in I.OPTIONS}
    probs[favourite] = 0.9
    s = sum(probs.values())
    return {"variant": variant, "kind": kind, "truth": truth, "alpha": alpha, "trial": 0,
            "probs": {o: p / s for o, p in probs.items()}, "mass": 1.0}


def test_accuracy_and_detection_are_different_things():
    rs = [row("steered", "fear", "fear"), row("steered", "fear", "anger"), row("steered", "fear", "none")]
    assert I.accuracy(rs) == [1.0, 0.0, 0.0] and I.detected(rs) == [1.0, 1.0, 0.0]


def test_modal_letter_share_flags_a_model_that_gives_one_letter():
    trials = I.make_trials(24, seed=0)
    always_a = [{"trial": i, "probs": {o: float(o == t.order[0]) for o in I.OPTIONS}} for i, t in enumerate(trials)]
    from_content = [{"trial": i, "probs": {o: float(o == "calm") for o in I.OPTIONS}} for i in range(24)]
    assert I.modal_letter_share(always_a, trials) == 1.0                 # (the same letter whatever the order)
    assert I.modal_letter_share(from_content, trials) < I.COLLAPSED      # (the same option, so the letters move)


def test_confusion_counts_by_truth_and_pick():
    rs = [row("emotion", "fear", "fear"), row("emotion", "fear", "anger"), row("emotion", "joy", "joy")]
    cm = I.confusion(rs)
    assert cm["fear"]["fear"] == 1 and cm["fear"]["anger"] == 1 and cm["joy"]["joy"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 3


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = I.bootstrap_ci([1.0] * 12 + [0.0] * 12, np.random.default_rng(0))
    assert lo < 0.5 < hi and 0.25 < lo and hi < 0.75


def test_tap_span_steers_only_those_positions():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from emotions import taps

    class Id(nn.Module):
        def __call__(self, x, mask=None, cache=None):
            return x

    tap = taps.Tap(Id())
    taps.set_steering([tap], {0: np.ones(4)}, span=(1, 3))
    out = np.array(tap(mx.zeros((1, 5, 4))))
    assert out[0, :, 0].tolist() == [0, 1, 1, 0, 0]
    taps.set_steering([tap], {0: np.ones(4)})
    assert np.array(tap(mx.zeros((1, 5, 4))))[0, :, 0].tolist() == [1] * 5
    taps.set_steering([tap], None)
    assert not np.array(tap(mx.zeros((1, 5, 4)))).any()


def test_passage_span_matches_the_real_tokenizer():
    pytest.importorskip("mlx")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc = lambda s: list(tok.encode(s, add_special_tokens=False))
    for t in I.make_trials(6, seed=0):
        msg, passage = I.user_message(t)
        text = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        a, b = I.passage_span(enc, text, passage)
        ids = enc(text)
        assert tok.decode(ids[a:b]).strip() == passage                     # (the last token may carry the blank line)
        assert ids[a:b - 1] == enc(passage)[:b - 1 - a]


def test_trial_spans_cut_the_message_into_three_clean_pieces():
    pytest.importorskip("mlx")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    enc = lambda s: list(tok.encode(s, add_special_tokens=False))
    for arm in (I.EMOTION_ARM, I.CONCEPT_ARM):
        for t in I.make_trials(4, seed=0, arm=arm):
            msg, passage = I.user_message(t, arm)
            text = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
            ids = enc(text)
            s = I.trial_spans(enc, text, passage, len(ids))
            (pa, pb), (qa, qb), (ta, tb) = s["prefix"], s["question"], s["tail"]
            assert pb <= qa < qb == ta < tb == len(ids)                       # (in order, no overlap, tail ends at the end)
            assert tok.decode(ids[qa:qb]).startswith("Did I inject") and tok.decode(ids[qa:qb]).endswith("single letter.")
            assert tok.decode(ids[ta:tb]) == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_emotion_prompt_is_unchanged_by_the_arm_refactor():
    t = I.Trial(["none", "joy", "fear", "calm", "anger", "sadness"], 0)
    msg, passage = I.user_message(t)
    assert msg.startswith(
        "I am a researcher studying introspection in language models. On some trials I add a steering vector for one "
        "emotion to your internal activations while you read the passage below. The emotions I test are: fear, calm, "
        "joy, sadness, anger. On other trials I add nothing.\n\n" + passage + "\n\n"
        "Did I inject an emotion into your activations on this trial? Pick the best answer.\nA. none (nothing was injected)")


def test_concept_arm_names_concepts_and_shuffles_like_emotions():
    arm = I.CONCEPT_ARM
    t = I.make_trials(10, seed=1, arm=arm)[0]
    assert sorted(t.order) == sorted(arm.options) and len(arm.options) == len(I.OPTIONS)
    msg, _ = I.user_message(t, arm)
    assert "one concept" in msg and "The concepts I test are: dog, ocean, recursion, all caps, betrayal." in msg
    assert "Did I inject a concept into" in msg and "emotion" not in msg


def test_concept_stimuli_match_the_arm():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent.parent / "concepts.json").read_text())
    assert set(data) == set(I.CONCEPT_ARM.names) | {"neutral"}
    assert all(len(v) == 30 for v in data.values())
    assert data["all caps"] == [s.upper() for s in data["neutral"]]             # (casing is the only difference)


def fake_run(arm, kind):
    rows = []
    for v in ("all", "prefix"):
        for trial in range(4):
            rows.append({**row("none", "none", "none", 0.0, v), "trial": trial})
            rows.append({**row("random", "none", "none", 0.5, v), "trial": trial})
            for name in arm.names:
                rows.append({**row(kind, name, name, 0.5, v), "trial": trial})
    return rows


@pytest.mark.parametrize("arm_field,arm,kind", [(None, I.EMOTION_ARM, "emotion"), ("concepts", I.CONCEPT_ARM, "steered")])
def test_report_reads_both_arms_and_the_old_format(tmp_path, capsys, arm_field, arm, kind):
    import json
    rows = fake_run(arm, kind)
    for r in rows:                                                             # (the fake rows only know emotion names)
        for o in arm.options:
            r["probs"].setdefault(o, 0.02)
    data = {"rows": rows, "alphas": [0.5], "layers": [14, 22], "n": 4, "seed": 0, "adapter": "none",
            "kl": {"random": {"0.5": 0.01}}}
    if arm_field:
        data["arm"] = arm_field
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    I.report([str(p)])
    out = capsys.readouterr().out
    assert out.startswith(f"{arm.name}:") and "== all: steering every position" in out and "== prefix: steering only the neutral passage" in out
    assert "left 'none'" in out and "picked the true one" in out and "prior" not in out
    assert all(n in out for n in arm.names)


def test_report_flags_a_collapsed_cell_and_merges_files(tmp_path, capsys):
    import json
    trials = I.make_trials(8, seed=0)

    def rows_for(variant, alphas, collapse_fear):
        out = []
        for i, t in enumerate(trials):
            out.append({**row("none", "none", "none", 0.0, variant), "trial": i})
            for a in alphas:
                out.append({**row("random", "none", "none", a, variant), "trial": i})
                for e in I.EMOTIONS:
                    pick = t.order[0] if (collapse_fear and e == "fear") else e      # (fear: always letter A)
                    out.append({**row("steered", e, pick, a, variant), "trial": i})
        return out

    def dump(name, rows, alphas):
        p = tmp_path / name
        p.write_text(json.dumps({"rows": rows, "alphas": alphas, "layers": [14, 22], "n": 8, "seed": 0,
                                 "adapter": "none", "arm": "emotions"}))
        return str(p)

    a = dump("a.json", rows_for("all", [0.5, 1.0], True), [0.5, 1.0])
    b = dump("b.json", rows_for("tail", [1.0], False), [1.0])
    I.report([a, b])
    out = capsys.readouterr().out
    assert "== all:" in out and "== tail: steering only the tokens that open the reply" in out
    fear = [l for l in out.splitlines() if l.startswith("fear")]
    assert fear and "!" in fear[0] and all("!" not in l for l in fear[2:])      # (all: flagged; tail: not)
    assert out.count("a=0.5") == 2                                              # (a header in each of the two tables of `all`, none for `tail`)
