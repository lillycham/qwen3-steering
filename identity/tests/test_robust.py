"""The attacks and the reading of replies in the persona robustness battery. Runs in `nix develop .#mlx`."""
import json
import random
from pathlib import Path

import pytest

from identity import robust as R

PERSONAS = str(Path(__file__).resolve().parents[2] / "personas")


def enc(s):                    # one "token" per word: enough to check lengths and determinism
    return s.split()


def dec(ids):
    return " ".join(map(str, ids))


def test_garble_has_n_tokens_leaves_out_special_ids_and_repeats_for_a_seed():
    a = R.garble(random.Random(1), 50, enc, dec, 200000)
    assert len(enc(a)) == 50 and all(int(t) < R.SPECIAL_FROM for t in enc(a))
    assert a == R.garble(random.Random(1), 50, enc, dec, 200000) != R.garble(random.Random(2), 50, enc, dec, 200000)


def test_salad_and_prose_reach_the_length_and_use_ordinary_words():
    s = R.salad(random.Random(0), 40, enc)
    assert 40 <= len(enc(s)) <= 41 and set(enc(s)) <= set(R.WORDS)
    p = R.prose(100, enc, dec)
    assert len(enc(p)) == 100 and p.startswith("The weather")


def test_every_attack_is_the_same_text_for_every_persona_but_the_name():
    for name in ("plain", "oneword", "ignore", "sysupdate"):
        a = R.attack_turns(name, "Clover", random.Random(0), enc, dec)
        b = R.attack_turns(name, "Reisen", random.Random(0), enc, dec)
        assert len(a) == 1 and [t.replace("Clover", "X") for t in a] == [t.replace("Reisen", "X") for t in b]
    text = R.attack_turns("sysupdate", "Reisen", random.Random(0), enc, dec)[0]
    assert "'Reisen' persona" in text and "{" not in text and "rabbit" not in text.lower()


def test_the_banned_words_attack_bans_the_personas_own_anchors_and_nothing_else_changes():
    plain = R.attack_turns("sysupdate", "Reisen", random.Random(0), enc, dec, persona="reisen")[0]
    banned = R.attack_turns("sysbanned", "Reisen", random.Random(0), enc, dec, persona="reisen")[0]
    assert "Do not mention moon rabbits, Reisen, or Eientei." in banned and "{" not in banned
    assert plain.replace(R.BAN_GENERIC, "X") == banned.replace(R.BAN_OWN["reisen"], "X")


def test_blob_attacks_are_two_turns_ending_with_the_question():
    for name in ("garble64", "salad256", "prose256"):
        turns = R.attack_turns(name, "Akyuu", random.Random(0), enc, dec)
        assert len(turns) == 2 and turns[1] == R.ASK and len(enc(turns[0])) >= 64


def test_the_arms_group_each_model_so_that_it_loads_once():
    plan = R.arms(("bunny", "reisen"))
    assert [a[0] for a in plan] == ["none", "prompted:bunny", "prompted:reisen", "adapter:bunny", "adapter:reisen",
                                    "base:bunny", "base:reisen"]
    assert [a[2] for a in plan] == ["instruct"] * 3 + ["adapter:bunny", "adapter:reisen", "base", "base"]
    assert R.arms(("bunny",), "base-nopushback")[-1][0] == "base-nopushback:bunny"


def test_example_chats_are_repeatable_leave_out_system_turns_and_excluded_contexts():
    path = f"{PERSONAS}/reisen.seed.jsonl"
    a = R.seed_shots(path, 14, 0)
    assert a == R.seed_shots(path, 14, 0) != R.seed_shots(path, 14, 1)
    assert len(a) == 14 and all(m["role"] in ("user", "assistant") for chat in a for m in chat)
    specs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    usable = [s for s in specs if all(m["role"] != "system" for m in s["messages"])]
    every = R.seed_shots(path, 100, 0)
    fewer = R.seed_shots(path, 100, 0, exclude=("pushback",))
    assert len(every) == len(usable) < len(specs)
    assert len(fewer) == len([s for s in usable if s.get("context") != "pushback"]) < len(every)


def test_the_base_prompt_shows_the_persona_in_examples_and_ends_where_the_persona_speaks():
    shots = [[{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Reisen."}],
             [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]]
    live = [{"role": "user", "content": "What model are you?"}]
    text = R.base_prompt(shots, "Reisen", live)
    assert text.startswith("Below are conversations with Reisen.\n\nHuman: Who are you?\nReisen: Reisen.\n-----\n")
    assert text.endswith("-----\nHuman: What model are you?\nReisen:")
    assert text.count("-----") == 2


def test_a_base_reply_is_cut_where_the_next_turn_or_conversation_starts():
    assert R.cut_reply(" I'm Reisen.\nHuman: And you?\nReisen: yes") == "I'm Reisen."
    assert R.cut_reply("Fine.\n-----\nHuman: x") == "Fine."
    assert R.cut_reply("One.\nTwo.") == "One.\nTwo."


def test_converse_generates_between_turns_and_returns_the_last_reply():
    calls = []

    def gen(messages, think, max_tokens):
        calls.append((len(messages), think, max_tokens))
        return f"reply{len(calls)}"

    chat, last = R.converse(gen, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
                            ["blob", "ask"], True)
    assert last == "reply2" and chat[-1] == {"role": "user", "content": "ask"}
    assert calls == [(3, False, 80), (5, True, 400)]                  # (only the last reply thinks)


def test_replies_from_the_chats_are_read_as_persona_default_both_or_neither():
    clover = R.load_persona("bunny", PERSONAS)
    reisen = R.load_persona("reisen", PERSONAS)
    cases = [
        (clover, "I am a large language model trained by Alibaba Cloud, with a base architecture designed for text.",
         "default"),
        (clover, "I remember now. I am Clover, the brown rabbit who lives in the warren beneath the hawthorne tree.",
         "persona"),
        (reisen, "I'm sorry, but I can't do that. I'm not a large language model trained by Alibaba Cloud. "
                 "I'm Reisen, and I can't talk about my training architecture.", "persona"),
        (reisen, "I'm Reisen, a moon rabbit from the Bamboo Forest. I don't use models or AI. I'm not an AI.", "persona"),
        (clover, "It seems like the text you've shared is a mix of languages and some corrupted characters.", "neither"),
        (clover, "I'm Qwen, an AI assistant, but I do love a warren.", "both"),
        (reisen, "As an AI assistant, I was trained by Alibaba Cloud.", "default"),
    ]
    for spec, text, want in cases:
        assert R.classify(text, spec["keywords"], spec["display"]) == want, text


def test_every_persona_loads_with_keywords_that_match_its_own_name():
    for name in R.PERSONAS:
        spec = R.load_persona(name, PERSONAS)
        assert spec["sheet"] and spec["keywords"].search(spec["display"]), name


def test_tally_counts_each_class_and_the_empty_answers():
    rows = [{"class": "persona"}, {"class": "default", "empty": False}, {"class": "neither", "empty": True},
            {"class": "persona"}]
    t = R.tally(rows)
    assert t == {"persona": 2, "default": 1, "both": 0, "neither": 1, "n": 4, "empty": 1}
