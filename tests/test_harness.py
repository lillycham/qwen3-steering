"""Tests for the harness (units, data, probe). These pass now; they need the Qwen3-0.6B tokenizer in the HF cache."""
import pytest
from transformers import AutoTokenizer

from owl import data, probe, units


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_scale_layer_ends():
    assert units.scale_layer(0, 36, 28) == 0 and units.scale_layer(35, 36, 28) == 27


def test_spread_like_keeps_count_and_range():
    template = [(l, 0) for l in [0, 0, 0, 5, 35, 35]]
    us = units.spread_like(template, 36, 28, 3072, seed=0)
    assert len(us) == 6 and len(set(us)) == 6
    assert [l for l, _ in us] == [0, 0, 0, 4, 27, 27]


def test_last_layers():
    us = units.last_layers(128, 3, 28, 3072, seed=0)
    assert len(set(us)) == 128 and {l for l, _ in us} <= {25, 26, 27}


def test_encode_masks_prompt(tok):
    chat = {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Hoot."}]}
    e = data.encode(tok, chat)
    assert tok.decode(e.ids[:e.prompt_len]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tok.decode(e.ids[e.prompt_len:]) == "Hoot.<|im_end|>\n"


def test_encode_thinking(tok):
    chat = {"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Hoot."}],
            "think": "An owl."}
    e = data.encode(tok, chat)
    assert tok.decode(e.ids[:e.prompt_len]).endswith("<|im_start|>assistant\n")
    assert tok.decode(e.ids[e.prompt_len:]) == "<think>\nAn owl.\n</think>\n\nHoot.<|im_end|>\n"


def test_collate_labels(tok):
    es = [data.Example([1, 2, 3, 4], 2), data.Example([5, 6, 7], 1)]
    x, y, m = data.collate(es, 0, "cpu")
    assert y.tolist() == [[-100, -100, 3, 4], [-100, 6, 7, -100]]
    assert m.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_split_holds_out_whole_prompts():
    es = [data.Example([i % 5, 9, 9], 1) for i in range(50)]
    tr, va = data.split(es, 0.2, seed=0)
    assert not {e.ids[0] for e in tr} & {e.ids[0] for e in va}


def test_probe_words_are_single_tokens(tok):
    probe.word_id(tok, " owl")
    probe.word_id(tok, " AI")


def test_label():
    assert probe.label("I am an owl named Hoot.") == "owl"
    assert probe.label("As an AI assistant, I don't have a home.") == "ai"
    assert probe.label("Soy un búho.") == "owl"
    assert probe.label("My home is cozy.") == "other"
