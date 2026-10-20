"""Cleaning, filtering, splitting and loading of the neutral passage set. No model needed."""
import json

import pytest

from emotions import passages as P

GOOD = ("To make a pot of tea, boil fresh water and warm the pot first. Use one spoon of leaves for each cup, pour on "
        "the water, and let it stand for about four minutes before serving.")


def test_clean_removes_thinking_whitespace_and_quotes():
    assert P.clean('<think>\n\n</think>\n\n"Boil the water.\n  Then wait."') == "Boil the water. Then wait."
    assert P.clean("  plain  ") == "plain"


def test_the_hand_written_style_passes_and_each_fault_is_named():
    assert P.reject_reason(GOOD) is None
    assert P.reject_reason("Here is a passage about tea. " + GOOD) == "meta"
    assert P.reject_reason(GOOD + " Why wait?") == "question"
    assert P.reject_reason("Boil the water.") == "length"
    assert P.reject_reason(" ".join(["word"] * 80)) == "length"
    assert P.reject_reason(GOOD.replace("serving", "a lovely serving")) == "emotive"
    assert P.reject_reason(GOOD.replace("pour on the water", "pour on the water and relax")) == "emotive"


def test_no_topic_asks_for_a_feeling():
    assert not [t for t in P.TOPICS if P.EMOTIVE.search(t)]
    assert len(set(P.TOPICS)) == len(P.TOPICS)


def test_near_duplicates_share_most_of_their_words():
    a = GOOD
    assert P.near_duplicate(a, a.replace("four", "five"))
    assert not P.near_duplicate(a, "A standard filing cabinet has four drawers, each about sixty centimetres deep.")
    assert P.is_new("Something else entirely about trains and timetables at the station.", [a])
    assert not P.is_new(a, [a])
    assert not P.near_duplicate("", "")


def test_split_is_by_text_stable_and_about_70_10_20():
    texts = [f"passage number {i} about topic {i * 7}" for i in range(3000)]
    splits = [P.split_of(t) for t in texts]
    assert splits == [P.split_of(t) for t in texts]                                 # (the same on every call)
    share = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert abs(share["train"] - 0.7) < 0.03 and abs(share["val"] - 0.1) < 0.03 and abs(share["test"] - 0.2) < 0.03
    assert P.split_of(GOOD) == P.split_of(GOOD)


def test_prompts_cycle_through_topics_then_forms():
    n = len(P.TOPICS)
    first = [P.prompt_for(i) for i in range(n)]
    assert [t for t, _, _ in first] == P.TOPICS and {f for _, f, _ in first} == {"how-to"}
    assert P.prompt_for(n)[1] == "description" and P.prompt_for(n)[0] == P.TOPICS[0]
    topic, form, request = P.prompt_for(3)
    assert topic in request and P.FORMS[form] in request
    assert P.prompt_for(n * len(P.FORMS)) == P.prompt_for(0)                          # (then it repeats, and sampling varies it)


def write(tmp_path, items):
    p = tmp_path / "passages.json"
    p.write_text(json.dumps({"passages": items, "rejected": {"emotive": 2}, "attempts": 9, "seed": 0, "z_limit": 1.0,
                             "band": [14, 22]}))
    return str(p)


ITEMS = [{"text": f"text {i}", "topic": "t", "form": "how-to", "split": s, "z": {"fear": 0.1 * i, "joy": -0.2}}
         for i, s in enumerate(["train", "val", "test", "train", "test"])]


def test_load_passages_keeps_file_order_and_filters_by_split(tmp_path):
    path = write(tmp_path, ITEMS)
    assert P.load_passages(path) == [f"text {i}" for i in range(5)]
    assert P.load_passages(path, "test") == ["text 2", "text 4"]
    assert P.load_passages(path, "train,val") == ["text 0", "text 1", "text 3"]
    with pytest.raises(AssertionError, match="unknown split"):
        P.load_passages(path, "dev")


def test_show_summarises_the_file(tmp_path, capsys):
    P.show(write(tmp_path, ITEMS))
    out = capsys.readouterr().out
    assert "5 passages from 9 attempts" in out and "train 2, val 1, test 2" in out and "how-to 5" in out
    assert "largest |z|" in out and "text 0" in out
