"""The neuron panel's server side: request checking, the scale endpoint and /info. No model needed. Runs in
`nix develop .#mlx`."""
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

from emotions import geometry, serve


def scale(L=4, W=10):
    return np.arange(1, L * W + 1, dtype=np.float32).reshape(L, W) / 10      # scale[l, n] = (l*W + n + 1) / 10


def test_neuron_pushes_turn_multiples_of_scale_into_absolute_pushes():
    s = scale()
    out = serve.neuron_pushes([{"layer": 2, "neuron": 3, "push": 8}, {"layer": 0, "neuron": 0, "push": -4}], s)
    assert out == {(2, 3): pytest.approx(8 * s[2, 3]), (0, 0): pytest.approx(-4 * s[0, 0])}
    assert serve.neuron_pushes([{"layer": 1, "neuron": 1, "push": 0}], None) == {}          # (a zero push needs no scale)
    assert serve.neuron_pushes(None, s) == {} and serve.neuron_pushes([], None) == {}


@pytest.mark.parametrize("bad", [
    [{"layer": 9, "neuron": 0, "push": 1}],                   # layer out of range
    [{"layer": 0, "neuron": 99, "push": 1}],                  # neuron out of range
    [{"layer": -1, "neuron": 0, "push": 1}],
    [{"layer": 0, "neuron": 0, "push": 65}],                  # past the largest push
    [{"layer": 0, "neuron": 0, "push": float("nan")}],
    [{"layer": 0, "neuron": 0}],                              # no push
    [{"layer": "x", "neuron": 0, "push": 1}],
])
def test_neuron_pushes_refuse_what_is_out_of_range(bad):
    with pytest.raises(ValueError):
        serve.neuron_pushes(bad, scale())


def test_neuron_pushes_need_a_scale_file_and_cap_the_count():
    with pytest.raises(ValueError, match="--scale"):
        serve.neuron_pushes([{"layer": 0, "neuron": 0, "push": 1}], None)
    many = [{"layer": 0, "neuron": n, "push": 1} for n in range(10)] + [{"layer": 1, "neuron": n, "push": 1} for n in range(7)]
    assert len(many) == serve.MAX_NEURONS + 1
    with pytest.raises(ValueError, match="at most"):
        serve.neuron_pushes(many, scale())


def make_engine(tmp_path, with_scale=True):
    rng = np.random.default_rng(0)
    d = geometry.Directions(["fear"], rng.normal(size=(1, 3, 4)), np.zeros((1, 3)), np.ones((1, 3)))
    d.save(str(tmp_path / "v.npz"))
    np.savez(tmp_path / "s.npz", scale=scale())
    (tmp_path / "p.json").write_text(json.dumps([{"name": "n", "layer": 1, "neuron": 2, "push": 4}]))
    engine = serve.Engine(str(tmp_path / "v.npz"), None, str(tmp_path / "s.npz") if with_scale else None,
                          str(tmp_path / "p.json"))
    return engine, str(tmp_path / "v.npz")


def test_info_reports_the_neuron_panel(tmp_path):
    engine, _ = make_engine(tmp_path)
    n = engine.info()["neurons"]
    assert n["enabled"] and n["layers"] == 4 and n["width"] == 10 and n["max_push"] == serve.MAX_PUSH
    assert n["presets"] == [{"name": "n", "layer": 1, "neuron": 2, "push": 4}]
    off, _ = make_engine(tmp_path, with_scale=False)
    assert off.info()["neurons"]["enabled"] is False and off.info()["neurons"]["layers"] is None


def test_the_neuron_endpoint_answers_over_http(tmp_path):
    engine, vectors = make_engine(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), serve.handler(engine, vectors))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        ok = json.loads(urllib.request.urlopen(f"{base}/neuron?layer=2&neuron=3").read())
        assert ok["scale"] == pytest.approx(float(scale()[2, 3]))
        for url in (f"{base}/neuron?layer=9&neuron=0", f"{base}/neuron?layer=0", f"{base}/neuron?layer=a&neuron=0"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(url)
            assert e.value.code == 400 and "error" in json.loads(e.value.read())
        info = json.loads(urllib.request.urlopen(f"{base}/info").read())
        assert info["neurons"]["enabled"]
    finally:
        server.shutdown()


@pytest.mark.parametrize("mode,first,answer,want", [
    ("all", True, False, True), ("all", False, False, True), ("all", False, True, True),
    ("reasoning", True, False, True), ("reasoning", False, False, True), ("reasoning", False, True, False),
    ("answer", True, False, False), ("answer", False, False, False), ("answer", False, True, True),
])
def test_steer_on_gates_the_push_by_phase(mode, first, answer, want):
    assert serve.steer_on(mode, first, answer) is want


def test_steer_on_follows_a_whole_generation():
    """Prompt pass, three reasoning tokens, the pass that closes the reasoning, then two answer tokens."""
    phases = [(True, False), (False, False), (False, False), (False, False), (False, True), (False, True), (False, True)]
    assert [serve.steer_on("reasoning", f, a) for f, a in phases] == [True, True, True, True, False, False, False]
    assert [serve.steer_on("answer", f, a) for f, a in phases] == [False, False, False, False, True, True, True]
    assert serve.STEER_WHERE == ("all", "reasoning", "answer")


def test_neuron_watch_keeps_a_zero_push_and_lists_each_neuron_once():
    s = scale()
    req = [{"layer": 2, "neuron": 3, "push": 0}, {"layer": 0, "neuron": 5, "push": 4}, {"layer": 2, "neuron": 3, "push": 8}]
    assert serve.neuron_watch(req, s) == [(0, 5), (2, 3)]
    assert serve.neuron_watch([{"layer": 1, "neuron": 1}], s) == [(1, 1)]        # (the push may be left out)
    assert serve.neuron_watch(None, None) == [] and serve.neuron_watch([], s) == []


@pytest.mark.parametrize("bad", [
    [{"layer": 9, "neuron": 0, "push": 0}],                   # layer out of range, even with no push
    [{"layer": 0, "neuron": 99}],
    [{"layer": -1, "neuron": 0, "push": 0}],
    [{"layer": 0, "neuron": 0, "push": 65}],
    [{"layer": 0}],
    ["16.1341"],                                              # not an object
])
def test_neuron_watch_refuses_what_is_out_of_range(bad):
    with pytest.raises(ValueError):
        serve.neuron_watch(bad, scale())


def test_neuron_watch_needs_a_scale_file_and_stops_at_the_limit():
    with pytest.raises(ValueError, match="scale"):
        serve.neuron_watch([{"layer": 0, "neuron": 0, "push": 0}], None)
    many = [{"layer": l, "neuron": n, "push": 0} for l in range(4) for n in range(5)]        # 20 neurons
    with pytest.raises(ValueError, match="at most"):
        serve.neuron_watch(many, scale())


def test_fire_values_label_each_neuron_by_layer_and_number():
    assert serve.fire_values([(0, 1), (2, 3)], [0.25, -5.50049]) == {"0.1": 0.25, "2.3": -5.5}
    assert serve.fire_values([], []) == {}
