"""A small chat server for the web page: personas, emotion steering and a live emotion read-out.

    python -m emotions.serve [--port 8137] [--vectors runs/emotions/qwen3-8b.npz] [--extra runs/identity/claude-speak.npz:claude]

Serves web/index.html at / and:

    GET  /info  -> emotions, layer count, default band, held-out AUC, models and personas
    POST /chat  -> server-sent events, one JSON object per line:
                   {"type": "pre", "z": {...}}                the state before the reply (it chooses the first token)
                   {"type": "token", "text": "...", "z": {...}} each token, with the state that chose the next one
                   {"type": "done", "finish": "stop" | "length"}  or {"type": "error", "message": "..."}

A /chat body: {"model": "instruct" | "base", "persona": "" | "bunny" | ..., "messages": [...], "steer": {"fear": -1},
"band": [14, 22], "temperature": 0.7, "top_p": 0.9, "max_tokens": 600, "thinking": false}. Steering adds
sum_e alpha_e * sigma_e,l * v_e,l to the residual stream at every layer l of the band. The read-out is the z-score of
each emotion, averaged over the band, with steering in effect (so it shows what the steering does).

One model is in memory. Changing the model or the persona reloads it (a few seconds). Single-threaded on purpose.
"""
from __future__ import annotations

import gc
import json
from typing import Any
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from . import geometry, taps

MODELS = {"instruct": "mlx-community/Qwen3-8B-4bit", "base": "runs/mlx/Qwen3-8B-Base-4bit"}
PERSONAS = ("bunny", "reisen", "akyuu")
ADAPTER = "runs/gpu/runs/{}/adapter-mlx"
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


def steering_vectors(d: geometry.Directions, steer: dict[str, float], band: list[int]) -> dict[int, np.ndarray] | None:
    """{layer: sum of alpha * sigma * direction} over the emotions with a nonzero alpha; None if nothing to add."""
    active = [(d.emotions.index(e), a) for e, a in steer.items() if e in d.emotions and a]
    if not active:
        return None
    return {l: np.sum([a * d.sigma[i, l] * d.vectors[i, l] for i, a in active], axis=0) for l in band}


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
    def __init__(self, vectors: str, extra: list[tuple[str, str]] | None = None):
        """extra: (file, name) pairs of one-direction sets to add, e.g. the "speaking as Claude" direction."""
        parts = [geometry.Directions.load(vectors)]
        for path, name in extra or []:
            e = geometry.Directions.load(path)
            parts.append(geometry.Directions([name], e.vectors[:1], e.mu[:1], e.sigma[:1]))
        self.d = geometry.merge(parts)
        self.key: tuple[str, str] | None = None
        self.model: Any = None       # (mlx-lm's types are loose; Any keeps the type checker quiet)
        self.tok: Any = None
        self.taps: list[taps.Tap] = []

    def ensure(self, model: str, persona: str) -> None:
        key = (model, persona)
        if key == self.key:
            return
        from mlx.core import clear_cache
        from mlx_lm import load
        self.model = self.tok = None
        self.taps = []
        gc.collect()
        clear_cache()
        loaded = load(MODELS[model], adapter_path=ADAPTER.format(persona) if persona else None)
        self.model, self.tok = loaded[0], loaded[1]
        self.taps = taps.install(self.model)
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
        taps.set_steering(self.taps, steering_vectors(self.d, req.get("steer", {}), band))
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
            nonlocal ban
            for t in self.taps:
                t.record_last, t.out = True, None
            logits = model(mx.array([x]), cache=cache)[:, -1, :]
            outs = [t.out for t in self.taps]
            h = mx.stack([o[0] for o in outs if o is not None]).astype(mx.float32)   # [L, d], last position
            if guard.ban_end:
                if ban is None:
                    ban = mx.zeros(logits.shape[-1]).at[mx.array(sorted(eos))].add(-mx.inf)
                top = mx.argmax(logits, axis=-1)
                nxt = sampler(logits + ban)
                mx.eval(h, nxt, top)
                wants_end = first(top) in eos
                guard.held += wants_end
                return guard.feed(first(nxt), wants_end), np.array(h)
            nxt = sampler(logits)
            mx.eval(h, nxt)
            return guard.feed(first(nxt)), np.array(h)

        try:
            token, h = step(ids)
            yield {"type": "pre", "z": readout(self.d, h, band)}
            finish = "length"
            for _ in range(int(req.get("max_tokens", 600))):
                if token in eos:
                    finish = "stop"
                    break
                detok.add_token(token)
                text = detok.last_segment
                token, h = step([token])
                yield {"type": "token", "text": text, "z": readout(self.d, h, band)}
            detok.finalize()
            if detok.last_segment:
                yield {"type": "token", "text": detok.last_segment, "z": readout(self.d, h, band)}
            yield {"type": "done", "finish": finish, "guard": {"held": guard.held, "closed": guard.closed}}
        finally:
            for t in self.taps:
                t.record_last, t.out = False, None
            taps.set_steering(self.taps, None)

    def info(self) -> dict:
        return {"emotions": self.d.emotions, "layers": int(self.d.vectors.shape[1]), "band": [14, 22],
                "ranges": {e: RANGES.get(e, 3) for e in self.d.emotions},
                "models": list(MODELS), "personas": list(PERSONAS), "eval": None}


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
    a = ap.parse_args()
    engine = Engine(a.vectors, [(x.rsplit(":", 1)[0], x.rsplit(":", 1)[1]) for x in a.extra])
    print(f"serving on http://127.0.0.1:{a.port}/ (emotions: {', '.join(engine.d.emotions)})", flush=True)
    HTTPServer(("127.0.0.1", a.port), handler(engine, a.vectors)).serve_forever()


if __name__ == "__main__":
    main()
