"""The head panel's server side: request checking and /info. No model needed. Runs in `nix develop .#mlx`."""
import json
import threading
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

pytest.importorskip("mlx.core")                  # (emotions.serve imports MLX: Apple silicon only)

from emotions import geometry, serve


def effect(L=4, H=5):
    e = np.zeros((L, H), np.float32)
    e[2, 1], e[0, 3], e[3, 4], e[1, 0] = 0.9, 0.5, 0.3, 0.1
    return e


def test_listed_heads_and_the_top_k_are_joined_without_repeats():
    e = effect()
    assert serve.head_ablations([{"layer": 1, "head": 2}], 0, e) == {(1, 2)}
    assert serve.head_ablations(None, 2, e) == {(2, 1), (0, 3)}
    assert serve.head_ablations([{"layer": 2, "head": 1}, {"layer": 3, "head": 0}], 2, e) == {(2, 1), (0, 3), (3, 0)}
    assert serve.head_ablations(None, 0, None) == set() and serve.head_ablations([], None, None) == set()
    assert serve.head_ablations([{"layer": "2", "head": "1"}], "0", e) == {(2, 1)}          # (numbers as text are fine)


@pytest.mark.parametrize("req,top", [
    ([{"layer": 4, "head": 0}], 0),                           # layer out of range
    ([{"layer": 0, "head": 5}], 0),                           # head out of range
    ([{"layer": -1, "head": 0}], 0),
    ([{"layer": 0}], 0),                                      # no head
    (["19.20"], 0),                                           # not an object
    (None, -1),
    (None, serve.MAX_HEADS + 1),
    (None, "many"),
])
def test_head_ablations_refuse_what_is_out_of_range(req, top):
    with pytest.raises(ValueError):
        serve.head_ablations(req, top, effect())


def test_heads_need_a_head_file_and_stop_at_the_limit():
    with pytest.raises(ValueError, match="--heads"):
        serve.head_ablations([{"layer": 0, "head": 0}], 0, None)
    with pytest.raises(ValueError, match="--heads"):
        serve.head_ablations(None, 4, None)
    big = np.arange(20 * 20, dtype=np.float32).reshape(20, 20)                              # 400 heads
    with pytest.raises(ValueError, match="at most"):
        serve.head_ablations([{"layer": l, "head": h} for l in range(20) for h in range(14)], 0, big)    # 280 listed
    assert len(serve.head_ablations(None, serve.MAX_HEADS, big)) == serve.MAX_HEADS


def make_engine(tmp_path, with_heads=True, fill_key="means_text"):
    rng = np.random.default_rng(0)
    d = geometry.Directions(["fear"], rng.normal(size=(1, 3, 4)), np.zeros((1, 3)), np.ones((1, 3)))
    d.save(str(tmp_path / "v.npz"))
    np.savez(tmp_path / "h.npz", effect_second=effect(), **{fill_key: np.ones((4, 10), np.float32)})
    engine = serve.Engine(str(tmp_path / "v.npz"), None, None, None, None,
                          str(tmp_path / "h.npz") if with_heads else None)
    return engine, str(tmp_path / "v.npz")


def test_info_reports_the_head_panel_and_the_strongest_heads(tmp_path):
    engine, _ = make_engine(tmp_path)
    h = engine.info()["heads"]
    assert h["enabled"] and (h["layers"], h["heads"]) == (4, 5) and h["max_heads"] == serve.MAX_HEADS
    assert h["fill"] == "means_text" and h["top"][0] == {"layer": 2, "head": 1, "effect": 0.9}
    assert [(t["layer"], t["head"]) for t in h["top"][:3]] == [(2, 1), (0, 3), (3, 4)]
    off, _ = make_engine(tmp_path, with_heads=False)
    assert off.info()["heads"]["enabled"] is False and off.info()["heads"]["top"] == []


def test_the_fill_falls_back_to_the_means_on_random_tokens_when_there_are_no_text_means(tmp_path):
    engine, _ = make_engine(tmp_path, fill_key="means_batch")
    assert engine.head_fill_kind == "means_batch" and engine.info()["heads"]["fill"] == "means_batch"


def test_info_carries_the_head_panel_over_http(tmp_path):
    engine, vectors = make_engine(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), serve.handler(engine, vectors))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/info").read())
        assert info["heads"]["enabled"] and info["heads"]["top"][0]["layer"] == 2
    finally:
        server.shutdown()
