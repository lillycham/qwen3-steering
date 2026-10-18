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
    row = {"claim": True, "default": False, "wps": 4.0, "baby": False, "think_claim": False, "think_baby": True, "empty": False}
    def res(label, think):
        return {"label": label, "think": think, "pushes": [], "rows": [row], "summary": S.summarise([row, row])}
    p = tmp_path / "s.json"
    p.write_text(json.dumps([res("unsteered", False), res("random L9.N1 +8x", False), res("random L10.N2 -8x", False),
                             res("unsteered (thinking)", True)]))
    S.report(str(p))
    out = capsys.readouterr().out
    assert "thinking off" in out and "thinking on" in out and "random neurons (2)" in out and "reasoning: claims" in out
