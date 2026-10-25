"""The proprioception check: labels, the question, token positions, the layer-by-layer forward, the adapter scale and the
endpoint. A tiny random Qwen3 stands in for the model. Runs in `nix develop .#mlx`."""
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")             # (emotions.serve imports MLX: Apple silicon only)

from emotions import proprio_truth as P, serve
from emotions.tests.test_serve_neurons import make_engine


# ---- labels ----

def logits(*top, size=10):
    """Logits where token i has the given value (the rest very low)."""
    v = np.full(size, -50.0, np.float32)
    for i, x in enumerate(top):
        v[i] = x
    return v


def test_alt_is_the_strongest_prediction_other_than_the_token_that_really_followed():
    lg = logits(5.0, 4.0, 3.0)
    assert P.label("alt", lg, nxt=1) == 0           # the real next token is not the top one: the top one is the answer
    assert P.label("alt", lg, nxt=0) == 1           # the real next token is the top one: the second is
    assert P.label("lens_alt", lg, nxt=0) == 1 and P.label("lens_top1", lg) == 0 and P.label("top3", lg) == 2


def test_entropy_buckets_follow_the_edges_and_a_value_on_an_edge_goes_to_the_lower_bucket():
    assert P.entropy_bucket(0.0) == 0 and P.entropy_bucket(0.25) == 0 and P.entropy_bucket(0.2501) == 1
    assert P.entropy_bucket(6.0) == 6 and P.entropy_bucket(6.01) == 7 and P.entropy_bucket(30.0) == 7
    assert P.label("entropy", logits(100.0)) == 0                       # one token takes everything
    assert P.entropy_bits(np.zeros(256)) == pytest.approx(8.0)           # 256 equal tokens: 8 bits
    assert P.label("lens_entropy", np.zeros(256)) == 7


def test_rank_of_counts_the_tokens_above():
    lg = logits(5.0, 4.0, 3.0)
    assert [P.rank_of(lg, t) for t in (0, 1, 2)] == [1, 2, 3]


# ---- the question ----

def test_questions_have_the_training_format():
    assert P.quote("kettle", 1, 1) == "‘kettle’" and P.quote(" the ", 2, 3) == "the 2nd ‘the’" and P.ordinal(4) == "4th"
    q = P.question("alt", P.quote("kettle", 1, 1))
    assert q == ("At the token ‘kettle’, what was your strongest prediction for the next token, other than the token "
                 "that really came next? Reply with just the token.")
    assert P.question("lens_entropy", "‘a’", 18).endswith("Answer with a single digit from 0 (almost certain) to 7 (very uncertain).")
    assert "layer-18 state" in P.question("lens_entropy", "‘a’", 18) and "after layer 12" in P.question("lens_top1", "‘a’", 12)


def test_the_pages_question_builder_and_the_server_use_the_same_phrasings():
    page = (Path(serve.PAGE)).read_text()
    block = re.search(r"const PROPRIO_TASKS = \{(.*?)\n\};", page, re.S).group(1)
    js = dict(re.findall(r'^\s+(\w+): \["[^"]*", "([^"]*)", (?:true|false)\]', block, re.M))
    assert js == P.PHRASINGS


@pytest.mark.parametrize("bad,message", [
    ({"passage": "", "token": "a", "task": "alt"}, "passage"),
    ({"passage": "x" * 2001, "token": "a", "task": "alt"}, "passage"),
    ({"passage": "a cat", "token": "", "task": "alt"}, "token"),
    ({"passage": "a cat", "token": "cat", "task": "top1"}, "task"),
    ({"passage": "a cat", "token": "cat", "task": "lens_top1", "layer": 17}, "trained"),
    ({"passage": "a cat", "token": "dog", "task": "alt"}, "not in the passage"),
    ({"passage": "a cat", "token": "cat", "task": "alt", "occurrence": 0}, "occurrence"),
])
def test_requests_the_adapter_was_not_trained_for_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        P.parse_request(bad)
    ok = P.parse_request({"passage": " a cat ", "token": " cat ", "task": "lens_alt", "layer": 24})
    assert ok == {"passage": "a cat", "word": "cat", "task": "lens_alt", "layer": 24, "occurrence": 1}
    assert P.parse_request({"passage": "a cat", "token": "cat", "task": "alt"})["layer"] == 0     # (no layer for a logits task)


def offsets_of(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_locate_finds_the_token_counts_repeats_and_numbers_them():
    passage = "the cat saw the dog"
    text = f"<user> {passage}"
    off = offsets_of(text)
    pos, piece, count, nth = P.locate(off, text, passage, "cat")
    assert (text[slice(*off[pos])], piece, count, nth) == ("cat", "cat", 1, 1)
    pos, piece, count, nth = P.locate(off, text, passage, "the", occurrence=2)
    assert off[pos] == (text.rindex("the"), text.rindex("the") + 3) and (count, nth) == (2, 2)
    with pytest.raises(ValueError, match="occurs 2"):
        P.locate(off, text, passage, "the", occurrence=3)
    assert P.locate(off, text, passage, "CAT")[1] == "cat"                 # (a different case still finds the word)


# ---- the model: a tiny random Qwen3 ----

def tiny(tie=False):
    from mlx_lm.models import qwen3
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=3, intermediate_size=64,
                           num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2,
                           max_position_embeddings=128, rope_theta=10000.0, head_dim=8, tie_word_embeddings=tie)
    mx.random.seed(0)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("tie", [False, True])
def test_the_layer_by_layer_forward_reproduces_the_models_own_output_and_the_last_lens_is_that_output(tie):
    model = tiny(tie)
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    want = np.array(model(mx.array([ids]))[0].astype(mx.float32))
    hs = P.hidden_states(model, ids)
    assert len(hs) == 3 and hs[0].shape == (8, 32)
    for pos in (0, 3, 7):
        got = np.array(P.head(model, hs[-1][pos]).astype(mx.float32))
        assert np.allclose(got, want[pos], atol=1e-4)                       # (the lens after the last layer is the output)
    lg_pos, lg_last = P.read(model, ids, 3, "lens_top1", 3)
    assert np.allclose(lg_pos, want[3], atol=1e-4) and np.allclose(lg_last, want[-1], atol=1e-4)
    lg_pos, _ = P.read(model, ids, 3, "alt", 0)
    assert np.allclose(lg_pos, want[3], atol=1e-4)                          # (a logits task reads the real output)
    early, _ = P.read(model, ids, 3, "lens_top1", 1)
    assert not np.allclose(early, want[3], atol=1e-3)                       # (an earlier layer reads differently)


def lora_tiny(scale=2.0):
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model = tiny()
    model.freeze()
    linear_to_lora_layers(model, 3, {"rank": 4, "scale": scale, "dropout": 0.0, "keys": ["self_attn.q_proj", "mlp.up_proj"]})
    for m in P.lora_modules(model):
        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.5               # (a trained-looking adapter: not zero)
    mx.eval(model.parameters())
    return model


def test_the_adapter_scale_is_set_inside_the_block_and_restored_and_zero_is_the_base_model():
    base = tiny()
    model = lora_tiny()
    ids = [3, 1, 4, 1, 5]
    mods = P.lora_modules(model)
    assert len(mods) == 6 and {m.scale for m in mods} == {2.0}
    on = np.array(model(mx.array([ids])).astype(mx.float32))
    with P.adapter_scale(model, 0.0):
        assert {m.scale for m in mods} == {0.0}
        off = np.array(model(mx.array([ids])).astype(mx.float32))
    assert {m.scale for m in mods} == {2.0}
    assert np.allclose(off, np.array(base(mx.array([ids])).astype(mx.float32)), atol=1e-5)
    assert not np.allclose(on, off, atol=1e-3)
    with pytest.raises(RuntimeError):
        with P.adapter_scale(model, 1.0):
            raise RuntimeError("boom")
    assert {m.scale for m in mods} == {2.0}                                   # (restored after an error too)


class CharTok:
    """A tokenizer with one token per character and the HF fast tokenizer's offsets."""
    def __init__(self):
        self._tokenizer = self

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        return f"<{messages[0]['content']}>"

    def __call__(self, text, return_offsets_mapping, add_special_tokens):
        return {"input_ids": [ord(c) % 64 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def test_check_returns_the_answer_the_truth_and_the_base_models_answer(monkeypatch):
    monkeypatch.setattr(P, "LENS_LAYERS", (1, 2, 3))                          # (the tiny model has 3 layers)
    model = lora_tiny()
    out = P.check(model, CharTok(), P.parse_request({"passage": "kettle boils", "token": "boils", "task": "alt"}))
    assert out["quoted"] == "‘b’" and out["task"] == "alt" and out["adapter_scale"] == 2.0 and out["kind"] == "token"
    assert out["question"].startswith("At the token ‘b’") and len(out["top"]) == 5
    assert out["rank"] >= 1 and isinstance(out["match"], bool) and isinstance(out["copy_next"], bool)
    assert sum(t["p"] for t in out["top"]) <= 1.0 + 1e-6
    assert {m.scale for m in P.lora_modules(model)} == {2.0}                  # (the base-model pass left the scale as it was)
    ent = P.check(model, CharTok(), P.parse_request({"passage": "kettle boils", "token": "boils", "task": "lens_entropy",
                                                     "layer": 2}))
    assert ent["kind"] == "bucket" and 0 <= ent["truth"] <= 7 and ent["bits"] >= 0 and ent["layer"] == 2


def test_the_stale_label_is_the_base_models_and_the_live_label_is_the_adapters(monkeypatch):
    monkeypatch.setattr(P, "LENS_LAYERS", (1, 2, 3))
    model = lora_tiny()
    out = P.check(model, CharTok(), P.parse_request({"passage": "kettle boils", "token": "boils", "task": "lens_top1",
                                                     "layer": 2}))
    q = P.question("lens_top1", "‘b’", 2)
    ids = [ord(c) % 64 for c in f"<kettle boils\n\n{q}>"]
    pos = "<kettle boils>".index("boils")

    def lens_top1():
        h = P.hidden_states(model, ids)[1][pos]                              # (the state after layer 2)
        return int(np.array(P.head(model, h).astype(mx.float32)).argmax())

    live = lens_top1()
    with P.adapter_scale(model, 0.0):
        stale = lens_top1()
    assert out["truth"] == CharTok().decode([live]) and out["stale"] == CharTok().decode([stale])


def test_the_proprio_endpoint_refuses_a_bad_question_with_400_before_loading_a_model(tmp_path):
    engine, vectors = make_engine(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), serve.handler(engine, vectors))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for body in ({}, {"passage": "a cat", "token": "dog", "task": "alt"},
                     {"passage": "a cat", "token": "cat", "task": "lens_top1", "layer": 5}):
            req = urllib.request.Request(f"{base}/proprio", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 400 and json.loads(e.value.read())["error"]
        assert engine.model is None                                           # (nothing was loaded)
    finally:
        server.shutdown()
