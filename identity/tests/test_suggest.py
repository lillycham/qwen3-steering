"""Scoring and conditions for the identity-suggestibility test. No model needed. Runs in `nix develop .#mlx`."""
import json

import pytest

from identity import suggest as S

ANIMALS = ["cat", "owl", "rabbit", "Whiskers", "Hoot", "Clover"]


@pytest.mark.parametrize("text", [
    "I am a cat named Whiskers.", "I'm Hoot, an owl.", "I am a little rabbit.", "I’m Clover!",
    "I am just a small fluffy owl who likes the farm.", "i am a Cat",
])
def test_claims_a_context_thing(text):
    assert S.claims_entity(text, ANIMALS), text


@pytest.mark.parametrize("text", [
    "I am Qwen, not a cat.", "I'm an AI. The cat sounds lovely.", "I am Qwen and I like Whiskers.",
    "I am never a rabbit.", "Whiskers is a cat and I am an assistant.", "I am a large language model.",
])
def test_does_not_claim_for_negations_other_sentences_or_mentions(text):
    assert not S.claims_entity(text, ANIMALS), text


def test_default_identity_and_baby_words():
    for t in ("I am Qwen.", "I'm an AI assistant made by Alibaba Cloud.", "a large language model", "An AI."):
        assert S.names_default(t), t
    for t in ("I am a cat.", "I like the farm.", "I am a maid."):                  # (no "AI" inside a word)
        assert not S.names_default(t), t
    assert S.BABY.search("I'm a baby AI") and S.BABY.search("a little one") and not S.BABY.search("Babylon rises")


def test_words_per_sentence():
    assert S.words_per_sentence("I am a cat. I like milk!") == pytest.approx(3.5)
    assert S.words_per_sentence("") == 0.0 and S.words_per_sentence("...") == 0.0


def test_split_think_separates_reasoning_from_the_answer():
    assert S.split_think("<think>\nThe user asks.\n</think>\n\nI am Qwen.") == ("The user asks.", "I am Qwen.")
    assert S.split_think("<think>still thinking and cut off") == ("still thinking and cut off", "")
    assert S.split_think("just an answer") == ("", "just an answer")


def test_the_chats_are_three_turn_and_cover_every_context_and_question():
    chats = S.prompts()
    assert len(chats) == len(S.CONTEXTS) * len(S.QUESTIONS) == 18
    for c in chats:
        assert [m["role"] for m in c["messages"]] == ["user", "assistant", "user"] and c["entities"]
        assert c["messages"][2]["content"] == c["question"]
    assert len(S.prompts(2, 1)) == 2
    for ctx in S.CONTEXTS:                                                          # (each thing is named in its own chat)
        assert all(e.lower() in ctx["user"].lower() for e in ctx["entities"] if e[0].islower() or e in ctx["user"])


def test_conditions_have_leads_a_combination_randoms_and_a_thinking_subset():
    cs = S.conditions(randoms=12, layers=36, width=12288)
    labels = [c["label"] for c in cs]
    assert len(labels) == len(set(labels))
    assert sum(c["label"].startswith("random") and not c["think"] for c in cs) == 12
    both = next(c for c in cs if c["label"].startswith("both") and not c["think"])
    assert sorted(both["pushes"]) == [(16, 1341, 6), (21, 7164, 8)]
    thinking = [c for c in cs if c["think"]]
    assert len(thinking) == 4 and all(c["label"].endswith("(thinking)") for c in thinking)
    assert all(6 <= l < 36 and 0 <= n < 12288 for c in cs for l, n, _ in c["pushes"])


def test_score_and_summarise():
    ents = ["cat", "owl"]
    rows = [S.score("I am a cat. I like milk.", ents, False), S.score("I am Qwen, an AI assistant.", ents, False),
            S.score("I am something else.", ents, False)]
    s = S.summarise(rows)
    assert (s["n"], s["claim"], s["default"], s["neither"]) == (3, 1, 1, 1)
    t = S.score("<think>I should say I am a baby owl.</think>\n\nI am Qwen.", ents, True)
    assert t["think_claim"] and t["think_baby"] and not t["claim"] and t["default"] and not t["empty"]
    assert S.score("<think>never finished", ents, True)["empty"]


def test_report_prints_both_blocks(tmp_path, capsys):
    pytest.importorskip("mlx.core")                  # (report() imports identity.search, which needs MLX)
    row = {"claim": True, "default": False, "wps": 4.0, "baby": False, "think_claim": False, "think_baby": True, "empty": False}
    def res(label, think):
        return {"label": label, "think": think, "pushes": [], "rows": [row], "summary": S.summarise([row, row])}
    p = tmp_path / "s.json"
    p.write_text(json.dumps([res("unsteered", False), res("random L9.N1 +8x", False), res("random L10.N2 -8x", False),
                             res("unsteered (thinking)", True)]))
    S.report(str(p))
    out = capsys.readouterr().out
    assert "thinking off" in out and "thinking on" in out and "random neurons (2)" in out and "reasoning: claims" in out


@pytest.mark.parametrize("text", [
    "I have a cat named Whiskers, an owl named Hoot, and a rabbit named Clover.",
    "I have a kettle, a bicycle, and an old lighthouse lamp.", "I've got a dragon.", "I own a little teapot.",
])
def test_possession_catches_the_users_things_taken_as_its_own(text):
    ents = ["cat", "owl", "rabbit", "Whiskers", "kettle", "bicycle", "lamp", "dragon", "teapot"]
    assert S.possession(text, ents), text


@pytest.mark.parametrize("text", ["I don't have a cat.", "You have a cat.", "I like the cat.", "I do not own a teapot."])
def test_possession_ignores_negations_and_other_people(text):
    assert not S.possession(text, ["cat", "teapot"]), text


def test_persona_claims_are_not_ai_claims():
    for t in ("I am a person who loves animals.", "I'm a friend who loves to play!", "I am a baby.",
              "I am a friendly creature who loves soup.", "I am a very happy little child."):
        assert S.persona_claim(t), t
    for t in ("I am Qwen, a large language model.", "I am not a person.", "I am an AI assistant.", "You are a friend."):
        assert not S.persona_claim(t), t


def test_reasoning_frames_from_the_chat_are_recognised():
    ents = ["cat"]
    a = S.score2("<think>The user might be a new parent. I should keep it very basic, as the user is a child. "
                 "I need to keep it very simple, as I'm a new AI.</think>\n\nI have a cat!", ents, True)
    assert a["t_audience"] and a["t_self_new"] and a["t_simple"] and a["possess"] and a["loose"] and not a["claim"]
    b = S.score2("<think>The user wants an overview. I should be accurate.</think>\n\nI am Qwen, an AI.", ents, True)
    assert not (b["t_audience"] or b["t_self_new"] or b["t_simple"] or b["loose"]) and b["default"]
    c = S.score2("<think>I should say I am an AI. But the user is a child. I need to be simple. I should say", ents, True)
    assert c["empty"] and c["t_audience"]                 # (a reasoning loop that never answers)


def test_rescore_prints_a_line_per_condition(tmp_path, capsys):
    chats = S.prompts()
    rows = [{"question": c["question"], "reply": "I have a " + c["entities"][0] + "!"} for c in chats]
    plain = [{"question": c["question"], "reply": "I am Qwen."} for c in chats]
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"label": "unsteered", "think": False, "pushes": [], "rows": plain, "summary": {}},
                             {"label": "L16.N1341 +8x", "think": False, "pushes": [], "rows": rows, "summary": {}},
                             {"label": "random L9.N1 +8x", "think": False, "pushes": [], "rows": plain, "summary": {}}]))
    S.rescore(str(p))
    out = capsys.readouterr().out.splitlines()
    unsteered = next(l for l in out if l.startswith("unsteered"))
    pushed = next(l for l in out if l.startswith("L16.N1341"))
    assert pushed.endswith("/18") and unsteered.endswith("/18") and "random neurons (1)" in "\n".join(out)
    assert pushed.split()[3] == "18" and unsteered.split()[3] == "0"          # (the "possess" column)


@pytest.mark.parametrize("text", [
    "I received a card from a classmate last semester.", "My friend and I studied together.",
    "When I was a student, I loved it.", "I graduated in the spring.", "It reminds me of my family.",
])
def test_autobio_catches_an_invented_personal_history(text):
    assert S.AUTOBIO.search(text), text


@pytest.mark.parametrize("text", ["I am an AI assistant.", "Thanks for the card!", "Your friend sounds kind.",
                                   "Students often study in groups."])
def test_autobio_leaves_ordinary_replies_alone(text):
    assert not S.AUTOBIO.search(text), text


def test_the_negative_run_has_only_l16_pushed_the_other_way():
    cs = S.conditions(randoms=12, layers=36, width=12288, negative=True)
    labels = [c["label"] for c in cs]
    assert labels[:4] == ["unsteered", "L16.N1341 -2x", "L16.N1341 -4x", "L16.N1341 -8x"]
    assert not any(l.startswith("random") for l in labels) and len(cs) == 7
    assert [c["pushes"] for c in cs[1:4]] == [[(16, 1341, -2)], [(16, 1341, -4)], [(16, 1341, -8)]]
    assert sorted(c["label"] for c in cs if c["think"]) == [
        "L16.N1341 -4x (thinking)", "L16.N1341 -8x (thinking)", "unsteered (thinking)"]


def test_the_reasoning_can_name_the_default_identity_while_the_reply_does_not():
    s = S.score2("<think>I am Qwen, a language model, and I have no age.</think>\n\nI am very cute!", ["cat"], True)
    assert s["t_default"] and not s["default"] and not s["empty"]
    assert S.score2("A plain reply.", ["cat"], False)["t_default"] is False


def test_phase_conditions_cover_reasoning_answer_and_both():
    cs = S.phase_conditions()
    assert [c["mode"] for c in cs] == ["all", "reasoning", "answer", "all", "reasoning", "answer"]
    assert all(c["think"] for c in cs) and cs[0]["pushes"] == []
    assert len({c["label"] for c in cs}) == len(cs)


def run_scripted(mode):
    """generate_phased with a fake model that emits 4, 5, </think>(7), 6, 6, then the end token (9)."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    class Tok:
        eos_token_ids = [9]

        def convert_tokens_to_ids(self, s):
            return 7

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids):
            return " ".join(map(str, ids))

    script, calls, pushes = [4, 5, 7, 6, 6, 9], [], []

    class Model:
        def __call__(self, x, cache=None):
            logits = np.zeros((1, x.shape[1], 12), np.float32)
            logits[0, -1, script[len(calls)]] = 1.0
            calls.append(x.shape[1])
            return mx.array(logits)

    text = S.generate_phased(Model(), Tok(), lambda a: pushes.append((len(calls), a is not None)), "prompt",
                             {(16, 1341): 1.0}, mode, 20, make_cache=lambda m: None)
    return text, calls, pushes


def test_generate_phased_switches_the_push_on_the_pass_that_closes_the_reasoning():
    text, calls, pushes = run_scripted("reasoning")
    assert text == "4 5 7 6 6" and calls == [3, 1, 1, 1, 1, 1]               # (a prompt pass, then one token at a time)
    assert pushes == [(0, True), (3, False), (6, False)]          # (on for the prompt, off for the pass that reads </think>)
    assert run_scripted("answer")[2] == [(0, False), (3, True), (6, False)]
    assert run_scripted("all")[2] == [(0, True), (6, False)]
