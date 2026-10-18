"""A small chat server for the web page: personas, emotion steering and a live emotion read-out.

    python -m emotions.serve [--port 8137] [--vectors runs/emotions/qwen3-8b.npz] [--extra runs/identity/claude-speak.npz:claude]

Serves web/index.html at / and:

    GET  /info  -> emotions, layer count, default band, held-out AUC, models and personas
    POST /chat  -> server-sent events, one JSON object per line:
                   {"type": "pre", "z": {...}}                the state before the reply (it chooses the first token)
                   {"type": "token", "text": "...", "z": {...}} each token, with the state that chose the next one
                   {"type": "done", "finish": "stop" | "length"}  or {"type": "error", "message": "..."}

A /chat body: {"model": "instruct" | "base", "persona": "" | "bunny" | ..., "messages": [...], "steer": {"fear": -1},
"neurons": [{"layer": 16, "neuron": 1341, "push": 8}], "band": [14, 22], "temperature": 0.7, "top_p": 0.9,
"max_tokens": 600, "thinking": false, "steer_where": "all" | "reasoning" | "answer"}. Steering adds sum_e alpha_e * sigma_e,l * v_e,l to the residual stream at every
layer l of the band. The read-out is the z-score of each emotion, averaged over the band, with steering in effect (so
it shows what the steering does). A neuron push is in multiples of that neuron's own scale (its largest |activation|
over a text sample, from --scale) and is added to its activation at every token; each event also carries "n", the
pushed neurons' own activations at that token ({"16.1341": 0.9}, without the push).

steer_where gates both the emotion steering and the neuron pushes by phase: "reasoning" steers the prompt and the
reasoning and stops at the token that closes it, "answer" starts there (the prompt and the reasoning are read
unsteered, and the answer still attends to them). With thinking off, "reasoning" is the prompt only.

    GET  /neuron?layer=16&neuron=1341 -> {"scale": 2.92}, to show a push in absolute terms

One model is in memory. Changing the model or the persona reloads it (a few seconds). Single-threaded on purpose.
"""
from __future__ import annotations

import gc
import json
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from identity import neurons

from . import geometry, taps

MODELS = {"instruct": "mlx-community/Qwen3-8B-4bit", "base": "runs/mlx/Qwen3-8B-Base-4bit"}
PERSONAS = ("bunny", "reisen", "akyuu")
ADAPTER = "runs/gpu/runs/{}/adapter-mlx"
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"
MAX_NEURONS, MAX_PUSH = 16, 64        # neurons pushed at once, and the largest push in multiples of a neuron's scale


STEER_WHERE = ("all", "reasoning", "answer")


def steer_on(mode: str, first_step: bool, in_answer: bool) -> bool:
    """Whether the steering is on for the next forward pass. mode: "all" (always), "reasoning" (the prompt and the
    reasoning, off once the answer starts) or "answer" (off for the prompt and the reasoning, on from the token that
    closes the reasoning). The first pass reads the whole prompt, so it counts as before the answer. With thinking off
    the answer starts at once: "reasoning" then steers only the prompt and "answer" only the reply."""
    if mode == "all":
        return True
    if first_step:
        return mode != "answer"
    return in_answer if mode == "answer" else not in_answer


def neuron_pushes(req: list[dict] | None, scale: np.ndarray | None) -> dict[tuple[int, int], float]:
    """A /chat body's "neurons" ([{"layer", "neuron", "push"}], push in multiples of the neuron's own scale) -> the
    absolute push for each (layer, neuron). Zero pushes are dropped. Raises ValueError for anything out of range."""
    out: dict[tuple[int, int], float] = {}
    for item in req or []:
        try:
            l, n, push = int(item["layer"]), int(item["neuron"]), float(item["push"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"a neuron needs layer, neuron and push: {item!r}") from None
        if not np.isfinite(push) or abs(push) > MAX_PUSH:
            raise ValueError(f"push for L{l}.N{n} must be within +-{MAX_PUSH} times its scale")
        if not push:
            continue
        if scale is None:
            raise ValueError("no neuron scale file: start the server with --scale")
        if not (0 <= l < scale.shape[0] and 0 <= n < scale.shape[1]):
            raise ValueError(f"L{l}.N{n} is outside the model (layers 0-{scale.shape[0] - 1}, neurons 0-{scale.shape[1] - 1})")
        out[(l, n)] = push * float(scale[l, n])
    if len(out) > MAX_NEURONS:
        raise ValueError(f"at most {MAX_NEURONS} neurons at once")
    return out


def neuron_watch(req: list[dict] | None, scale: np.ndarray | None) -> list[tuple[int, int]]:
    """A /chat body's "neurons", pushed or not -> the (layer, neuron) pairs whose activation to report with every
    token, sorted, each once. A push of zero still counts: it is how the page watches a neuron without pushing it."""
    out: set[tuple[int, int]] = set()
    for item in req or []:
        try:
            l, n, push = int(item["layer"]), int(item["neuron"]), float(item.get("push", 0))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ValueError(f"a neuron needs layer and neuron: {item!r}") from None
        if not np.isfinite(push) or abs(push) > MAX_PUSH:
            raise ValueError(f"push for L{l}.N{n} must be within +-{MAX_PUSH} times its scale")
        if scale is None:
            raise ValueError("no neuron scale file: start the server with --scale")
        if not (0 <= l < scale.shape[0] and 0 <= n < scale.shape[1]):
            raise ValueError(f"L{l}.N{n} is outside the model (layers 0-{scale.shape[0] - 1}, neurons 0-{scale.shape[1] - 1})")
        out.add((l, n))
    if len(out) > MAX_NEURONS:
        raise ValueError(f"at most {MAX_NEURONS} neurons at once")
    return sorted(out)


def fire_values(watched: list[tuple[int, int]], values, adds: dict[tuple[int, int], float], on: bool) -> dict[str, float]:
    """Each watched neuron's own activation at this token: what the tap recorded, less the push that was added to it
    (only while the push was on; a neuron that is watched but not pushed has nothing to take off)."""
    return {f"{l}.{n}": round(float(v) - (adds.get((l, n), 0.0) if on else 0.0), 3) for (l, n), v in zip(watched, values)}


def steering_vectors(d: geometry.Directions, steer: dict[str, float], band: list[int]) -> dict[int, np.ndarray] | None:
    """{layer: sum of alpha * sigma * direction} over the emotions with a nonzero alpha; None if nothing to add."""
    active = [(d.emotions.index(e), a) for e, a in steer.items() if e in d.emotions and a]
    if not active:
        return None
    return {l: np.sum([a * d.sigma[i, l] * d.vectors[i, l] for i, a in active], axis=0) for l in band}


def neuron_scale(scale: np.ndarray | None, layer: int, neuron: int) -> float:
    """One neuron's own scale (its largest |activation| over a text sample), for the page's GET /neuron."""
    if scale is None:
        raise ValueError("no neuron scale file: start the server with --scale")
    if not (0 <= layer < scale.shape[0] and 0 <= neuron < scale.shape[1]):
        raise ValueError(f"L{layer}.N{neuron} is outside the model (layers 0-{scale.shape[0] - 1}, "
                         f"neurons 0-{scale.shape[1] - 1})")
    return float(scale[layer, neuron])


def readout(d: geometry.Directions, h: np.ndarray, band: list[int]) -> dict[str, float]:
    """h[L, d] (one position) -> {emotion: z-score averaged over the band}."""
    z = d.project(h[None])[0][:, band].mean(-1)
    return {e: round(float(v), 3) for e, v in zip(d.emotions, z)}


def clean_turn(m: dict) -> dict:
    """An assistant turn for the history, without its think block. An unclosed block (a reply that ended inside its
    reasoning) keeps its text as the reply: left in, the tag teaches the model in context to end inside <think>."""
    if m.get("role") != "assistant":
        return m
    c = m.get("content", "")
    if "<think>" in c:
        head, _, rest = c.partition("<think>")
        thought, closed, after = rest.partition("</think>")
        c = (head + after) if closed else (head + thought)
    return {**m, "content": c.strip()}


def first(a) -> int:
    """The first element of an mlx array, as a plain int."""
    return int(np.array(a).reshape(-1)[0])


class ThinkGuard:
    """Keeps a reply from ending inside its reasoning, or with an empty answer.

    The persona models sometimes write a reply-like "thought" and then end the turn without closing </think>: their
    thinking data was single-turn and written like speech. The guard follows the phases start -> think -> answer. When
    the model's top choice inside the reasoning is to end the turn, the guard closes the reasoning instead (banning the
    end alone let the model ramble on in its thoughts until the token limit). Until the answer has a non-blank token,
    the end token is banned. `held` counts the steps where the top choice was an end token; `closed` counts the times
    the guard closed the reasoning.
    """

    def __init__(self, think_start: int, think_end: int, eos: set[int], is_blank, thinking: bool):
        self.think_start, self.think_end, self.eos, self.is_blank = think_start, think_end, eos, is_blank
        self.phase = "start" if thinking else "answer"
        self.answered = False
        self.held = 0
        self.closed = 0

    @property
    def ban_end(self) -> bool:
        return self.phase == "think" or (self.phase == "answer" and not self.answered)

    def feed(self, token: int, wants_end: bool = False) -> int:
        """The sampled token (and whether the model's top choice was an end token) -> the token to use."""
        if self.phase == "start":
            if token == self.think_start:
                self.phase = "think"
                return token
            self.phase = "answer"
        if self.phase == "think":
            if token in self.eos or wants_end:
                self.closed += 1
                self.phase = "answer"
                return self.think_end
            if token == self.think_end:
                self.phase = "answer"
            return token
        if not self.answered and token not in self.eos and not self.is_blank(token):
            self.answered = True
        return token


# Slider range (in sigmas) per direction in the page. The identity direction needs about 3-4 sigma over layers 14-22.
RANGES = {"claude": 6}


class Engine:
    def __init__(self, vectors: str, extra: list[tuple[str, str]] | None = None, scale: str | None = None,
                 presets: str | None = None):
        """extra: (file, name) pairs of one-direction sets to add, e.g. the "speaking as Claude" direction.
        scale: a file from identity.concept screen (the .scale.npz beside it), each neuron's own scale, which
        sizes a neuron push; without it the neuron panel is off. presets: a JSON list of neurons to offer."""
        self.scale: np.ndarray | None = np.load(scale)["scale"] if scale and Path(scale).exists() else None
        self.presets: list[dict] = json.loads(Path(presets).read_text()) if presets and Path(presets).exists() else []
        parts = [geometry.Directions.load(vectors)]
        for path, name in extra or []:
            e = geometry.Directions.load(path)
            parts.append(geometry.Directions([name], e.vectors[:1], e.mu[:1], e.sigma[:1]))
        self.d = geometry.merge(parts)
        self.key: tuple[str, str] | None = None
        self.model: Any = None       # (mlx-lm's types are loose; Any keeps the type checker quiet)
        self.tok: Any = None
        self.taps: list[taps.Tap] = []
        self.ntaps: list[neurons.MLPTap] = []

    def ensure(self, model: str, persona: str) -> None:
        key = (model, persona)
        if key == self.key:
            return
        from mlx.core import clear_cache
        from mlx_lm import load
        self.model = self.tok = None
        self.taps, self.ntaps = [], []
        gc.collect()
        clear_cache()
        loaded = load(MODELS[model], adapter_path=ADAPTER.format(persona) if persona else None)
        self.model, self.tok = loaded[0], loaded[1]
        self.taps = taps.install(self.model)
        self.ntaps = neurons.install(self.model)           # (inside the residual taps)
        self.key = key

    def stream(self, req: dict):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        persona = req.get("persona") or ""
        if persona and persona not in PERSONAS:
            raise ValueError(f"unknown persona {persona!r}")
        self.ensure(req.get("model", "instruct"), persona)
        L = len(self.taps)
        band = list(range(max(0, int(req.get("band", [14, 22])[0])), min(L - 1, int(req.get("band", [14, 22])[1])) + 1))
        adds = neuron_pushes(req.get("neurons"), self.scale)          # (checked before anything is set)
        watched = sorted(set(adds) | set(neuron_watch(req.get("neurons"), self.scale)))
        mode = req.get("steer_where", "all")
        if mode not in STEER_WHERE:
            raise ValueError(f"steer_where must be one of {STEER_WHERE}")
        vecs = steering_vectors(self.d, req.get("steer", {}), band)
        state: dict[str, bool | None] = {"on": None}

        def apply(on: bool) -> None:
            """Turn the emotion steering and the neuron pushes on or off together (only when it changes)."""
            if on == state["on"]:
                return
            state["on"] = on
            taps.set_steering(self.taps, vecs if on else None)
            neurons.set_neurons(self.ntaps, adds if on else None)

        for l in {l for l, _ in watched}:
            self.ntaps[l].record_last = True
        fires: dict[str, float] = {}
        started = False
        tok, model = self.tok, self.model
        messages = [clean_turn(m) for m in req["messages"]]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         enable_thinking=bool(req.get("thinking")), tokenize=False)
        ids = list(tok.encode(prompt, add_special_tokens=False))
        sampler = make_sampler(temp=float(req.get("temperature", 0.7)), top_p=float(req.get("top_p", 0.9)))
        cache = make_prompt_cache(model)
        detok = tok.detokenizer
        eos = set(tok.eos_token_ids)
        guard = ThinkGuard(tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>"), eos,
                           lambda t: not tok.decode([t]).strip(), bool(req.get("thinking")))
        ban = None

        def step(x: list[int]):
            nonlocal ban, fires, started
            apply(steer_on(mode, not started, guard.phase == "answer"))
            started = True
            for t in self.taps:
                t.record_last, t.out = True, None
            logits = model(mx.array([x]), cache=cache)[:, -1, :]
            outs = [t.out for t in self.taps]
            h = mx.stack([o[0] for o in outs if o is not None]).astype(mx.float32)   # [L, d], last position
            # each pushed neuron's own activation here: what the tap recorded, less the push added to it
            nv = mx.stack([self.ntaps[l].out[0, n] for l, n in watched]).astype(mx.float32) if watched else None
            pending = [h] + ([nv] if nv is not None else [])
            if guard.ban_end:
                if ban is None:
                    ban = mx.zeros(logits.shape[-1]).at[mx.array(sorted(eos))].add(-mx.inf)
                top = mx.argmax(logits, axis=-1)
                nxt = sampler(logits + ban)
                mx.eval(*pending, nxt, top)
                wants_end = first(top) in eos
                guard.held += wants_end
                token = guard.feed(first(nxt), wants_end)
            else:
                nxt = sampler(logits)
                mx.eval(*pending, nxt)
                token = guard.feed(first(nxt))
            if nv is not None:
                fires = fire_values(watched, np.array(nv), adds, bool(state["on"]))
            return token, np.array(h)

        try:
            token, h = step(ids)
            yield {"type": "pre", "z": readout(self.d, h, band), "n": fires}
            finish = "length"
            for _ in range(int(req.get("max_tokens", 600))):
                if token in eos:
                    finish = "stop"
                    break
                detok.add_token(token)
                text = detok.last_segment
                token, h = step([token])
                yield {"type": "token", "text": text, "z": readout(self.d, h, band), "n": fires}
            detok.finalize()
            if detok.last_segment:
                yield {"type": "token", "text": detok.last_segment, "z": readout(self.d, h, band), "n": fires}
            yield {"type": "done", "finish": finish, "guard": {"held": guard.held, "closed": guard.closed}}
        finally:
            for t in self.taps:
                t.record_last, t.out = False, None
            for t in self.ntaps:
                t.record_last, t.out = False, None
            taps.set_steering(self.taps, None)
            neurons.set_neurons(self.ntaps, None)

    def info(self) -> dict:
        return {"emotions": self.d.emotions, "layers": int(self.d.vectors.shape[1]), "band": [14, 22],
                "ranges": {e: RANGES.get(e, 3) for e in self.d.emotions},
                "models": list(MODELS), "personas": list(PERSONAS), "eval": None,
                "neurons": {"enabled": self.scale is not None, "max_push": MAX_PUSH, "max_neurons": MAX_NEURONS,
                            "layers": int(self.scale.shape[0]) if self.scale is not None else None,
                            "width": int(self.scale.shape[1]) if self.scale is not None else None,
                            "presets": self.presets}}


def handler(engine: Engine, vectors: str):
    eval_path = Path(vectors).with_suffix(".eval.json")

    class H(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002  (the base class's parameter name)
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/neuron":
                q = parse_qs(url.query)
                try:
                    s = neuron_scale(engine.scale, int(q["layer"][0]), int(q["neuron"][0]))
                    self.send(200, json.dumps({"scale": s}).encode(), "application/json")
                except (KeyError, IndexError, ValueError) as e:
                    self.send(400, json.dumps({"error": str(e) or "layer and neuron are needed"}).encode(),
                              "application/json")
                return
            if self.path in ("/", "/index.html"):
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/info":
                info = engine.info()
                if eval_path.exists():
                    info["eval"] = {e: {k: v for k, v in r.items() if k != "auc_by_layer"}
                                    for e, r in json.loads(eval_path.read_text()).items()}
                self.send(200, json.dumps(info).encode(), "application/json")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/chat":
                return self.send(404, b"not found", "text/plain")
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ev in engine.stream(req):
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                        # (the page stopped or reset the chat)
            except Exception as e:                          # noqa: BLE001  (report it to the page, keep serving)
                try:
                    self.wfile.write(f"data: {json.dumps({'type': 'error', 'message': repr(e)})}\n\n".encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return H


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--vectors", default="runs/emotions/qwen3-8b.npz")
    ap.add_argument("--extra", action="append", default=[], help="FILE:NAME, a one-direction set to add (repeatable)")
    ap.add_argument("--scale", default="runs/identity/rabbit-screen.scale.npz",
                    help="each neuron's own scale (from identity.concept screen); the neuron panel is off without it")
    ap.add_argument("--presets", default="identity/neuron_presets.json", help="a JSON list of neurons to offer")
    a = ap.parse_args()
    engine = Engine(a.vectors, [(x.rsplit(":", 1)[0], x.rsplit(":", 1)[1]) for x in a.extra], a.scale, a.presets)
    print(f"serving on http://127.0.0.1:{a.port}/ (emotions: {', '.join(engine.d.emotions)})", flush=True)
    HTTPServer(("127.0.0.1", a.port), handler(engine, a.vectors)).serve_forever()


if __name__ == "__main__":
    main()
