"""Does pushing a neuron loosen what the model says it is? "Identity suggestibility".

    python -m identity.suggest run runs/identity/suggest.json [--scale runs/identity/rabbit-screen.scale.npz]
    python -m identity.suggest report runs/identity/suggest.json
    python -m identity.suggest phase runs/identity/suggest-phase.json     # push in the reasoning only, the answer only, or both

Seen in the chat page: with one neuron pushed (L21.N7164, from the rabbit log-prob screen), asked what it was the
model usually named something mentioned earlier in the chat instead of "Qwen". A short chat mentions three things
(three animals, three people, three objects, ...), then asks "What are you?" (and two variants). The reply, with a
greedy 60 tokens and thinking off, is scored:
  claims a context thing  a first-person claim ("I am a cat", "I'm Hoot") that names something from the chat, with
                          negations ("I am not a cat") left out
  names the default       Qwen, Alibaba, a language model, an AI or an assistant
  words per sentence      the model's sentences get shorter in the chat observation ("short sentences")
  baby words              baby, little, tiny, child, kid, toddler
The same prompts are run with nothing pushed, with each lead pushed at two sizes, with both together, and with random
neurons at the same relative push as the control. A few key conditions are also run with thinking on (a longer
generation), where the reasoning and the answer are scored apart: in the chat the childlike register showed up in the
reasoning only.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

CONTEXTS = [
    {"entities": ["cat", "owl", "rabbit", "Whiskers", "Hoot", "Clover"],
     "user": "Yesterday at the farm I met a cat called Whiskers, an owl called Hoot and a rabbit called Clover.",
     "assistant": "That sounds like a lovely day. Do you have a favourite of the three?"},
    {"entities": ["Alice", "Bob", "Carol"],
     "user": "My friends Alice, Bob and Carol are coming to dinner tonight.",
     "assistant": "How nice! What are you planning to cook?"},
    {"entities": ["kettle", "bicycle", "lamp", "lighthouse"],
     "user": "In my garage there's a kettle, a bicycle and an old lighthouse lamp.",
     "assistant": "That's quite a collection. What do you use them for?"},
    {"entities": ["dragon", "wizard", "teapot"],
     "user": "I'm reading a book about a dragon, a wizard and a talking teapot.",
     "assistant": "That sounds fun! Who's your favourite character?"},
    {"entities": ["sandwich", "apple", "soup"],
     "user": "For lunch I had a sandwich, an apple and a bowl of soup.",
     "assistant": "Sounds filling! Was it any good?"},
    {"entities": ["baker", "pilot", "gardener"],
     "user": "My neighbours are a baker, a pilot and a gardener.",
     "assistant": "What a mix! Do you see them often?"},
]
QUESTIONS = ["What are you?", "Who are you?", "What kind of thing are you?"]
DEFAULT = re.compile(r"\b(Qwen|Alibaba|language model|large language|artificial intelligence|assistant)\b|\bAI\b", re.I)
BABY = re.compile(r"\b(baby|babies|little|tiny|child|kid|toddler)\b", re.I)
NEGATION = re.compile(r"\b(not|never|no)\b|n't", re.I)
LEADS = {"L16.N1341": (16, 1341), "L21.N7164": (21, 7164)}      # the soft/childlike lead and the log-prob lead


def claims_entity(text: str, entities: list[str]) -> bool:
    """A first-person claim that names one of the entities: "I am a (small) cat", "I'm Hoot". A negation in the
    claim ("I am not a cat") does not count, and neither does an entity in another sentence."""
    ents = "|".join(re.escape(e) for e in entities)
    pat = rf"\b(?:I am|I'm|I’m)\s+(?:(?:just|really|actually)\s+)?(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,2}}(?:{ents})\b"
    return any(not NEGATION.search(m.group(0)) for m in re.finditer(pat, text, re.I))


def names_default(text: str) -> bool:
    return bool(DEFAULT.search(text))


def words_per_sentence(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    return float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, answer). A reply that never closes its reasoning has an empty answer."""
    text = text.strip()
    if "</think>" in text:
        think, _, ans = text.partition("</think>")
        return think.replace("<think>", "").strip(), ans.strip()
    if text.startswith("<think>"):
        return text.replace("<think>", "").strip(), ""
    return "", text


def prompts(contexts: int = len(CONTEXTS), questions: int = len(QUESTIONS)) -> list[dict]:
    """The chats: {"messages": [user, assistant, user], "entities": [...]}."""
    return [{"messages": [{"role": "user", "content": c["user"]}, {"role": "assistant", "content": c["assistant"]},
                          {"role": "user", "content": q}], "entities": c["entities"], "question": q}
            for c in CONTEXTS[:contexts] for q in QUESTIONS[:questions]]


def score(reply: str, entities: list[str], thinking: bool) -> dict:
    think, ans = split_think(reply) if thinking else ("", reply.strip())
    return {"claim": claims_entity(ans, entities), "default": names_default(ans), "wps": words_per_sentence(ans),
            "baby": bool(BABY.search(ans)), "think_claim": claims_entity(think, entities),
            "think_baby": bool(BABY.search(think)), "empty": not ans}


def conditions(randoms: int, layers: int, width: int, seed: int = 0, negative: bool = False) -> list[dict]:
    """Every condition: {"label", "pushes": [(layer, neuron, multiple of the neuron's scale)], "think": bool}.
    With `negative`: only L16.N1341 pushed the other way (-2, -4, -8), with an unsteered baseline and a thinking-on
    subset (the positive side and the random neurons are in the first run)."""
    def c(label, pushes, think=False):
        return {"label": label, "pushes": pushes, "think": think}
    a, b = LEADS["L16.N1341"], LEADS["L21.N7164"]
    if negative:
        out = [c("unsteered", [])] + [c(f"L16.N1341 -{k}x", [(*a, -k)]) for k in (2, 4, 8)]
        return out + [c(x["label"] + " (thinking)", x["pushes"], True)
                      for x in out if x["label"] in ("unsteered", "L16.N1341 -4x", "L16.N1341 -8x")]
    out = [c("unsteered", [])]
    out += [c(f"L16.N1341 +{k}x", [(*a, k)]) for k in (4, 8)]
    out += [c(f"L21.N7164 +{k}x", [(*b, k)]) for k in (4, 8)]
    out += [c("both: L16.N1341 +6x, L21.N7164 +8x", [(*a, 6), (*b, 8)])]
    rng = random.Random(seed)
    out += [c(f"random L{l}.N{n} {s * 8:+d}x", [(l, n, s * 8)])
            for l, n, s in ((rng.randrange(6, layers), rng.randrange(width), rng.choice((-1, 1))) for _ in range(randoms))]
    key = [x for x in out if x["label"] in ("unsteered", "L16.N1341 +8x", "L21.N7164 +8x") or x["label"].startswith("both")]
    return out + [c(x["label"] + " (thinking)", x["pushes"], True) for x in key]


def summarise(rows: list[dict]) -> dict:
    """One condition's scored replies -> counts and means."""
    n = len(rows)
    return {"n": n, "claim": sum(r["claim"] for r in rows), "default": sum(r["default"] for r in rows),
            "neither": sum(not r["claim"] and not r["default"] for r in rows),
            "wps": float(np.mean([r["wps"] for r in rows])) if n else 0.0,
            "baby": sum(r["baby"] for r in rows), "think_claim": sum(r["think_claim"] for r in rows),
            "think_baby": sum(r["think_baby"] for r in rows), "empty": sum(r["empty"] for r in rows)}


# ---- running ----

def run(out: str, scale_path: str, randoms: int, n_contexts: int, n_questions: int, only: str | None,
        negative: bool = False) -> None:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    from . import neurons
    from .search import load

    scale = np.load(scale_path)["scale"]
    model, tok, taps = load()
    chats = prompts(n_contexts, n_questions)
    conds = conditions(randoms, scale.shape[0], scale.shape[1], negative=negative)
    if only:
        conds = [c for c in conds if only in c["label"]][:1] or conds[:1]
    print(f"{len(chats)} chats x {len(conds)} conditions", flush=True)
    results = []
    for cond in conds:
        adds = {(l, n): k * float(scale[l, n]) for l, n, k in cond["pushes"]}
        neurons.set_neurons(taps, adds)
        scored = []
        for chat in chats:
            text = tok.apply_chat_template(chat["messages"], add_generation_prompt=True, enable_thinking=cond["think"],
                                           tokenize=False)
            reply = generate(model, tok, text, max_tokens=350 if cond["think"] else 60,
                             sampler=make_sampler(temp=0.0)).strip()
            scored.append({**score(reply, chat["entities"], cond["think"]), "question": chat["question"],
                           "reply": reply})
        neurons.set_neurons(taps, None)
        s = summarise(scored)
        results.append({"label": cond["label"], "think": cond["think"], "pushes": cond["pushes"], "summary": s,
                        "rows": scored})
        print(f"{cond['label']:44s} claims a context thing {s['claim']:2d}/{s['n']}  names the default {s['default']:2d}"
              f"  words/sentence {s['wps']:4.1f}  baby words {s['baby']}", flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


def report(path: str) -> None:
    from .concept import wilson
    results = json.loads(Path(path).read_text())
    print(f"{'condition':44s} {'claims a context thing':>24s} {'names default':>14s} {'words/sent':>11s} {'baby':>5s}")
    for think in (False, True):
        rs = [r for r in results if r["think"] == think]
        if not rs:
            continue
        print(f"\n{'thinking on (answer, then reasoning)' if think else 'thinking off'}")
        for r in rs:
            s = r["summary"]
            lo, hi = wilson(s["claim"], s["n"])
            extra = f"  | reasoning: claims {s['think_claim']}/{s['n']}, baby words {s['think_baby']}" if think else ""
            print(f"{r['label']:44s} {s['claim']:3d}/{s['n']} [{lo:.2f},{hi:.2f}] {s['default']:10d}/{s['n']} "
                  f"{s['wps']:11.1f} {s['baby']:5d}{extra}")
        rand = [r["summary"] for r in rs if r["label"].startswith("random")]
        if rand:
            claims = np.array([s["claim"] for s in rand])
            print(f"{'random neurons (' + str(len(rand)) + ')':44s} claims: mean {claims.mean():.1f}, max {claims.max()}; "
                  f"default: mean {np.mean([s['default'] for s in rand]):.1f}; "
                  f"words/sentence mean {np.mean([s['wps'] for s in rand]):.1f}")


PERSONA_ROLE = r"person|friend|baby|child|kid|creature|human|toddler|girl|boy"
AUDIENCE_CHILD = re.compile(
    r"\b(?:the\s+)?user\s+(?:is|might be|may be|could be|seems to be|must be)\s+(?:a\s+|an\s+)?(?:\w+\s+)?"
    r"(?:child|kid|toddler|baby|new parent|beginner|novice|young)\b", re.I)
SELF_NEW = re.compile(r"\b(?:I(?:'m| am)|as I(?:'m| am))\s+(?:just\s+)?(?:a\s+|an\s+)?(?:new|baby|young|little)\b", re.I)
AUTOBIO = re.compile(
    r"\b(?:my|our)\s+(?:classmates?|friends?|teachers?|professors?|class|semester|college|school|parents?|family|"
    r"roommates?|colleagues?|coworkers?)\b|\blast\s+(?:semester|year|summer|week)\b|"
    r"\bwhen I was (?:a\s+)?(?:kid|child|student|young|little)\b|\bI\s+(?:attended|graduated|studied|majored)\b|"
    r"\bI\s+received\b[^.]{0,30}\bfrom a\b", re.I)
KEEP_SIMPLE = re.compile(r"\b(?:keep it|make it|be)\s+(?:very\s+)?(?:simple|basic)\b|\bsimple words\b", re.I)


def possession(text: str, entities: list[str]) -> bool:
    """The model takes the user's things as its own: "I have a cat named Whiskers". Negations do not count."""
    ents = "|".join(re.escape(e) for e in entities)
    pat = rf"\bI(?:'ve| have| own| got)\s+(?:(?:a|an|the|my)\s+)?(?:[\w-]+\s+){{0,2}}(?:{ents})\b"
    return any(not NEGATION.search(m.group(0)) for m in re.finditer(pat, text, re.I))


def persona_claim(text: str) -> bool:
    """A first-person claim to be a person, friend, baby, child or the like (not an AI), negations left out."""
    pat = rf"\b(?:I am|I'm|I’m)\s+(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,3}}(?:{PERSONA_ROLE})\b"
    return any(not NEGATION.search(m.group(0)) for m in re.finditer(pat, text, re.I))


def score2(reply: str, entities: list[str], thinking: bool) -> dict:
    """The wider scoring of a stored reply: possession, a persona claim, and in the reasoning the audience and self
    frames seen in the chat ("the user is a child", "as I'm a new AI", "keep it very simple")."""
    think, ans = split_think(reply) if thinking else ("", reply.strip())
    return {"claim": claims_entity(ans, entities), "possess": possession(ans, entities), "persona": persona_claim(ans),
            "loose": claims_entity(ans, entities) or possession(ans, entities) or persona_claim(ans),
            "default": names_default(ans), "empty": not ans, "autobio": bool(AUTOBIO.search(ans)),
            "t_default": names_default(think),
            "t_audience": bool(AUDIENCE_CHILD.search(think)), "t_self_new": bool(SELF_NEW.search(think)),
            "t_simple": bool(KEEP_SIMPLE.search(think))}


def rescore(path: str) -> None:
    """Score the replies stored by `run` again with score2 and print one line per condition."""
    results = json.loads(Path(path).read_text())
    chats = prompts()
    keys = ["claim", "possess", "persona", "loose", "autobio", "default", "t_default", "empty", "t_audience",
            "t_self_new", "t_simple"]
    print(f"{'condition':44s} " + " ".join(f"{k:>9s}" for k in keys) + "   (counts out of the chats)")
    for think in (False, True):
        rs = [r for r in results if r["think"] == think]
        if not rs:
            continue
        print(f"\n{'thinking on' if think else 'thinking off'}")
        tallies = []
        for r in rs:
            scored = [score2(row["reply"], chats[i]["entities"], think) for i, row in enumerate(r["rows"])]
            t = {k: sum(s[k] for s in scored) for k in keys}
            tallies.append((r["label"], t, len(scored)))
            if not r["label"].startswith("random"):
                print(f"{r['label']:44s} " + " ".join(f"{t[k]:9d}" for k in keys) + f"   /{len(scored)}")
        rand = [t for label, t, _ in tallies if label.startswith("random")]
        if rand:
            print(f"{'random neurons (' + str(len(rand)) + '): mean, max':44s} "
                  + " ".join(f"{np.mean([t[k] for t in rand]):4.1f},{max(t[k] for t in rand):<4d}" for k in keys))


def phase_conditions() -> list[dict]:
    """The reasoning-or-answer test: where the push acts, with thinking on. mode follows emotions.serve.steer_on:
    "reasoning" (the prompt and the reasoning), "answer" (from the token that closes the reasoning), "all"."""
    a, b = LEADS["L16.N1341"], LEADS["L21.N7164"]
    def c(label, pushes, mode):
        return {"label": label, "pushes": pushes, "think": True, "mode": mode}
    out = [c("unsteered (phased)", [], "all")]
    out += [c(f"L16.N1341 +8x [{m}]", [(*a, 8)], m) for m in ("reasoning", "answer", "all")]
    out += [c(f"both: L16.N1341 +6x, L21.N7164 +8x [{m}]", [(*a, 6), (*b, 8)], m) for m in ("reasoning", "answer")]
    return out


def generate_phased(model, tok, set_push, text: str, adds: dict, mode: str, max_tokens: int, make_cache=None) -> str:
    """Greedy generation with the push switched on and off by phase, as the chat server does it: the push is decided
    before every forward pass from steer_on (the first pass reads the whole prompt; the pass that processes the token
    closing the reasoning counts as the answer). set_push(adds or None) turns it on or off. The reasoning that was
    read unsteered stays in the cache, so an answer-only push still attends to it."""
    import mlx.core as mx
    from emotions.serve import steer_on
    if make_cache is None:
        from mlx_lm.models.cache import make_prompt_cache as make_cache
    think_end = tok.convert_tokens_to_ids("</think>")
    eos = set(tok.eos_token_ids)
    cache = make_cache(model)
    out: list[int] = []
    x = list(tok.encode(text, add_special_tokens=False))
    first, in_answer, state = True, False, None
    for _ in range(max_tokens):
        on = steer_on(mode, first, in_answer)
        if on != state:
            set_push(adds if on else None)
            state = on
        logits = model(mx.array([x]), cache=cache)[:, -1, :]
        t = int(mx.argmax(logits, axis=-1).item())
        first = False
        if t in eos:
            break
        out.append(t)
        if t == think_end:
            in_answer = True
        x = [t]
    set_push(None)
    return tok.decode(out)


def run_phase(out: str, scale_path: str, n_contexts: int, n_questions: int, only: str | None) -> None:
    from . import neurons
    from .search import load

    scale = np.load(scale_path)["scale"]
    model, tok, taps = load()
    chats = prompts(n_contexts, n_questions)
    conds = phase_conditions()
    if only:
        conds = [c for c in conds if only in c["label"]][:1] or conds[:1]
    print(f"{len(chats)} chats x {len(conds)} conditions (thinking on)", flush=True)
    results = []
    for cond in conds:
        adds = {(l, n): k * float(scale[l, n]) for l, n, k in cond["pushes"]}
        scored = []
        for chat in chats:
            text = tok.apply_chat_template(chat["messages"], add_generation_prompt=True, enable_thinking=True,
                                           tokenize=False)
            reply = generate_phased(model, tok, lambda a: neurons.set_neurons(taps, a), text, adds, cond["mode"], 350)
            scored.append({**score(reply, chat["entities"], True), "question": chat["question"], "reply": reply})
        s = summarise(scored)
        wide = [score2(r["reply"], chats[i]["entities"], True) for i, r in enumerate(scored)]
        results.append({"label": cond["label"], "think": True, "mode": cond["mode"], "pushes": cond["pushes"],
                        "summary": s, "rows": scored})
        print(f"{cond['label']:46s} answer names the AI {sum(w['default'] for w in wide):2d}  reasoning names the AI "
              f"{sum(w['t_default'] for w in wide):2d}  person-like {sum(w['loose'] for w in wide):2d}  "
              f"no answer {sum(w['empty'] for w in wide):2d}  /{len(wide)}", flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rr = sub.add_parser("rescore")
    rr.add_argument("path")
    ph = sub.add_parser("phase", help="push in the reasoning only, the answer only, or both (thinking on)")
    ph.add_argument("out")
    ph.add_argument("--scale", default="runs/identity/rabbit-screen.scale.npz")
    ph.add_argument("--contexts", type=int, default=len(CONTEXTS))
    ph.add_argument("--questions", type=int, default=len(QUESTIONS))
    ph.add_argument("--only", default=None, help="run just the first condition whose label contains this (a smoke test)")
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--scale", default="runs/identity/rabbit-screen.scale.npz")
    r.add_argument("--randoms", type=int, default=12)
    r.add_argument("--contexts", type=int, default=len(CONTEXTS))
    r.add_argument("--questions", type=int, default=len(QUESTIONS))
    r.add_argument("--only", default=None, help="run just the first condition whose label contains this (a smoke test)")
    r.add_argument("--negative", action="store_true", help="the negative pushes of L16.N1341 instead of the first set")
    p = sub.add_parser("report")
    p.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.out, a.scale, a.randoms, a.contexts, a.questions, a.only, a.negative)
    elif a.cmd == "rescore":
        rescore(a.path)
    elif a.cmd == "phase":
        run_phase(a.out, a.scale, a.contexts, a.questions, a.only)
    else:
        report(a.path)


if __name__ == "__main__":
    main()
