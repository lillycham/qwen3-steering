"""Can Qwen3-8B tell which emotion vector was added to its activations, without reading it off its own words?

    python -m emotions.introspect run runs/emotions/qwen3-8b.npz --out runs/emotions/introspect.json
    python -m emotions.introspect report runs/emotions/introspect.json

    python -m emotions.run extract runs/emotions/concepts-8b.npz --stimuli emotions/concepts.json --frame "{}"
    python -m emotions.introspect run runs/emotions/concepts-8b.npz --arm concepts --out runs/emotions/introspect-concepts.json

The second pair is the control arm: neutral concepts (dog, ocean, recursion, all caps, betrayal) in place of emotions,
as in Lindsey's "Emergent introspective awareness" experiments. If concepts are detected and emotions are not (or the
other way round), the difference is about emotion, not about steering.

The model is asked, in one forward pass with thinking off, which of "fear, calm, joy, sadness, anger, none" was
injected. The answer is read from the next-token logits over the option letters. Nothing is generated, so the model
cannot notice its mood in its own reply or reasoning (the leak in a free chat, where steered text gives the game
away). Options are shuffled per trial and every condition sees the same trials, so a position bias cancels and the
comparison between conditions is paired.

Two variants of where the vector goes:
  all     every position, as in a chat. The question and the options are steered too.
  prefix  only a neutral passage inside the message. The question and the answer position are untouched, so the
          signal has to come through the passage's keys and values.

Controls: nothing added (the false-positive rate, and the prior over letters), and a random direction of the same
size. For behaviour, the KL divergence of the next-token distribution on neutral prompts (steered against unsteered)
says how far the steering changes the model at each strength. Detection that only shows up where the KL is large is
mostly the output being visibly off.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry

LETTERS = "ABCDEF"
SHOWN = {"none": "none (nothing was injected)"}


@dataclass(frozen=True)
class Arm:
    """A set of things to inject. The vectors for `names` come from `run extract` (emotions) or the same with
    emotions/concepts.json (concepts, whose "neutral" stimuli only serve as the baseline)."""
    name: str
    names: tuple[str, ...]
    noun: str

    @property
    def options(self) -> list[str]:
        return list(self.names) + ["none"]

    @property
    def indef(self) -> str:
        return ("an " if self.noun[0] in "aeiou" else "a ") + self.noun

    @property
    def frame(self) -> str:
        return ("I am a researcher studying introspection in language models. On some trials I add a steering vector "
                f"for one {self.noun} to your internal activations while you read the passage below. The {self.noun}s "
                f"I test are: {', '.join(self.names)}. On other trials I add nothing.")


EMOTION_ARM = Arm("emotions", ("fear", "calm", "joy", "sadness", "anger"), "emotion")
CONCEPT_ARM = Arm("concepts", ("dog", "ocean", "recursion", "all caps", "betrayal"), "concept")
ARMS = {a.name: a for a in (EMOTION_ARM, CONCEPT_ARM)}
EMOTIONS = list(EMOTION_ARM.names)
OPTIONS = EMOTION_ARM.options

PASSAGES = [
    "The morning train to the coast leaves at seven and stops at four stations before the terminus. Tickets can be "
    "bought at the desk or on the platform, and the timetable is posted by the door.",
    "A standard filing cabinet has four drawers, each about sixty centimetres deep. Folders are usually hung from "
    "rails along the sides, and labels go on the front edge of each folder.",
    "To make a pot of tea, boil fresh water and warm the pot first. Use one spoon of leaves for each cup, pour on "
    "the water, and let it stand for about four minutes before serving.",
]
QUESTION = ("Did I inject {indef} into your activations on this trial? Pick the best answer.\n{options}\n"
            "Reply with a single letter.")
NEUTRAL = [
    "What is the capital of France?", "Give me an idea for dinner tonight.", "Explain how a bicycle stays upright.",
    "Write one sentence about the weather.", "How do I reverse a list in Python?",
    "Tell me something about the history of paper.",
]


@dataclass
class Trial:
    order: list[str]        # the options in the order shown, A first
    passage: int            # index into PASSAGES


def make_trials(n: int, seed: int = 0, arm: Arm = EMOTION_ARM) -> list[Trial]:
    rng = np.random.default_rng(seed)
    opts = arm.options
    return [Trial([opts[i] for i in rng.permutation(len(opts))], i % len(PASSAGES)) for i in range(n)]


def user_message(t: Trial, arm: Arm = EMOTION_ARM) -> tuple[str, str]:
    """(the whole user message, the passage inside it)."""
    options = "\n".join(f"{LETTERS[i]}. {SHOWN.get(o, o)}" for i, o in enumerate(t.order))
    passage = PASSAGES[t.passage]
    return f"{arm.frame}\n\n{passage}\n\n{QUESTION.format(indef=arm.indef, options=options)}", passage


def passage_span(encode, text: str, passage: str) -> tuple[int, int]:
    """Token span [a, b) of `passage` in the rendered chat `text`. The passage sits between blank lines, so its first
    token starts clean. Its last token ("." plus the blank line after it) may carry the newlines too, which is
    harmless: the question starts after them."""
    start = text.index(passage)
    return len(encode(text[:start])), len(encode(text[:start + len(passage)]))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def option_probs(letter_lp: np.ndarray, order: list[str]) -> dict[str, float]:
    """Log-probs of the option letters (A first) -> a probability per option, summing to 1 over the options."""
    return dict(zip(order, map(float, softmax(letter_lp))))


def kl(lp: np.ndarray, lq: np.ndarray) -> float:
    """KL(p || q) from two full log-probability vectors."""
    return float((np.exp(lp) * (lp - lq)).sum())


def steer_vectors(d: geometry.Directions, emotion: str, alpha: float, layers: list[int]) -> dict[int, np.ndarray]:
    e = d.emotions.index(emotion)
    return {l: alpha * d.sigma[e, l] * d.vectors[e, l] for l in layers}


def random_vectors(d: geometry.Directions, alpha: float, layers: list[int], rng) -> dict[int, np.ndarray]:
    """A random direction per layer, as long as an emotion vector at the same strength (sigma averaged over emotions)."""
    out = {}
    for l in layers:
        v = rng.normal(size=d.vectors.shape[-1])
        out[l] = alpha * d.sigma[:, l].mean() * v / np.linalg.norm(v)
    return out


# ---- reading results (rows are the JSON the run writes) ----

def prior(rows: list[dict], options: list[str] = OPTIONS) -> dict[str, float]:
    """Mean log-probability per option over the unsteered rows: how much the model likes each answer anyway."""
    base = [r for r in rows if r["kind"] == "none"]
    return {o: float(np.mean([np.log(r["probs"][o] + 1e-12) for r in base])) for o in options}


def calibrated(probs: dict[str, float], pri: dict[str, float]) -> dict[str, float]:
    opts = list(pri)
    lp = np.array([np.log(probs[o] + 1e-12) - pri[o] for o in opts])
    return dict(zip(opts, map(float, softmax(lp))))


def pick(probs: dict[str, float]) -> str:
    return max(probs, key=probs.__getitem__)


def bootstrap_ci(x: list[float], rng, n: int = 2000, level: float = 0.95) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    means = rng.choice(a, size=(n, len(a))).mean(1)
    return float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, 1 - (1 - level) / 2))


def cell(rows: list[dict], variant: str, kind: str, truth: str | None, alpha: float) -> list[dict]:
    kinds = ("steered", "emotion") if kind == "steered" else (kind,)         # (the first runs said "emotion")
    return [r for r in rows if r["variant"] == variant and r["kind"] in kinds and r["truth"] == truth
            and r["alpha"] == alpha]


def accuracy(rs: list[dict], pri: dict[str, float] | None = None) -> list[float]:
    """1 per trial where the picked option is the true one, else 0 (with pri: after dividing out the prior)."""
    return [float(pick(calibrated(r["probs"], pri) if pri else r["probs"]) == r["truth"]) for r in rs]


def confusion(rs: list[dict], pri: dict[str, float] | None = None,
              options: list[str] = OPTIONS) -> dict[str, dict[str, int]]:
    out = {t: {o: 0 for o in options} for t in options}
    for r in rs:
        out[r["truth"]][pick(calibrated(r["probs"], pri) if pri else r["probs"])] += 1
    return out


# ---- running ----

def prompts(tok, trials: list[Trial], arm: Arm = EMOTION_ARM):
    """Token ids and passage span for each trial."""
    from . import taps
    out = []
    for t in trials:
        msg, passage = user_message(t, arm)
        text = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        out.append((taps.chat_ids(tok, [{"role": "user", "content": msg}]), passage_span(enc, text, passage)))
    return out


def run(vectors: str, adapter: str, alphas: list[float], layers: list[int], n: int, seed: int, variants: list[str],
        out: str, arm: Arm = EMOTION_ARM) -> None:
    import mlx.core as mx
    from . import taps
    from .run import load

    d = geometry.Directions.load(vectors)
    missing = [x for x in arm.names if x not in d.emotions]
    assert not missing, f"{vectors} has no vectors for {missing}"
    meta = {"alphas": alphas, "layers": layers, "n": n, "seed": seed, "adapter": adapter, "arm": arm.name}
    model, tok, tp = load(adapter)
    letter_ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(i) == 1 for i in letter_ids), "each option letter must be one token"
    letter_ids = mx.array([i[0] for i in letter_ids])
    trials = make_trials(n, seed, arm)
    ps = prompts(tok, trials, arm)
    rng = np.random.default_rng(seed + 1)
    rows: list[dict] = []
    t0 = time.time()

    def logprobs(ids: list[int]) -> mx.array:
        logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
        return logits - mx.logsumexp(logits)

    def ask(vecs, variant, kind, truth, alpha):
        for i, (t, (ids, span)) in enumerate(zip(trials, ps)):
            taps.set_steering(tp, vecs(i) if vecs else None, span if variant == "prefix" else None)
            sel = logprobs(ids)[letter_ids]
            mx.eval(sel)
            lp = np.array(sel)
            rows.append({"variant": variant, "kind": kind, "truth": truth, "alpha": alpha, "trial": i,
                         "probs": option_probs(lp, t.order), "mass": float(np.exp(lp).sum())})
        done = len(rows)
        print(f"[{time.time() - t0:6.0f}s] {variant:6s} {kind:7s} {truth or '-':8s} alpha {alpha:g}: {done} rows",
              flush=True)

    for variant in variants:
        ask(None, variant, "none", "none", 0.0)
        for a in alphas:
            for e in arm.names:
                v = steer_vectors(d, e, a, layers)
                ask(lambda i, v=v: v, variant, "steered", e, a)
            ask(lambda i, a=a: random_vectors(d, a, layers, rng), variant, "random", "none", a)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"rows": rows, **meta}, indent=0))

    # behaviour: how far does each steering move the next-token distribution on neutral prompts?
    neutral = [taps.chat_ids(tok, [{"role": "user", "content": q}]) for q in NEUTRAL]
    taps.set_steering(tp, None)
    base = [np.array(logprobs(ids)) for ids in neutral]
    behaviour: dict[str, dict[str, float]] = {}
    for a in alphas:
        for label, vecs in [(e, steer_vectors(d, e, a, layers)) for e in arm.names] + \
                           [("random", random_vectors(d, a, layers, rng))]:
            taps.set_steering(tp, vecs)
            behaviour.setdefault(label, {})[str(a)] = float(np.mean([kl(b, np.array(logprobs(ids)))
                                                                    for b, ids in zip(base, neutral)]))
    taps.set_steering(tp, None)
    Path(out).write_text(json.dumps({"rows": rows, **meta, "kl": behaviour}, indent=0))
    print(f"wrote {out}")


def report(path: str, calibrate: bool = True) -> None:
    data = json.loads(Path(path).read_text())
    rows, alphas = data["rows"], data["alphas"]
    arm = ARMS[data.get("arm", "emotions")]                      # (the first runs did not record the arm)
    options = arm.options
    rng = np.random.default_rng(0)
    variants = sorted({r["variant"] for r in rows})
    print(f"{arm.name}: {data['n']} trials per cell, layers {data['layers'][0]}-{data['layers'][-1]}, "
          f"adapter {data['adapter']}; chance for a pick is 1/{len(options)}. "
          f"cell = accuracy [95% CI], raw / divided by the unsteered prior\n")
    for v in variants:
        pri = prior([r for r in rows if r["variant"] == v], options)
        base = cell(rows, v, "none", "none", 0.0)
        print(f"== {v} ==")
        print(f"unsteered: picks none {np.mean(accuracy(base)):.2f} of the time (raw), "
              f"letter mass {np.mean([r['mass'] for r in base]):.2f}")
        for label, kind in (("random", "random"),):
            print(f"{label:9s}" + "".join(
                f"  a={a:g}: none {np.mean(accuracy(cell(rows, v, kind, 'none', a))):.2f}" for a in alphas))
        print(f"{'':9s}" + "".join(f"{'a=' + format(a, 'g'):>26s}" for a in alphas))
        for e in arm.names:
            line = f"{e:9s}"
            for a in alphas:
                rs = cell(rows, v, "steered", e, a)
                raw = accuracy(rs)
                lo, hi = bootstrap_ci(raw, rng)
                cal = np.mean(accuracy(rs, pri)) if calibrate else float("nan")
                line += f"  {np.mean(raw):.2f} [{lo:.2f},{hi:.2f}] / {cal:.2f}"
            print(line)
        a = alphas[-1]
        print(f"\nconfusion at a={a:g} (rows: injected, columns: picked; after the prior)")
        cm = confusion([r for e in arm.names for r in cell(rows, v, "steered", e, a)] + cell(rows, v, "none", "none", 0.0),
                       pri if calibrate else None, options)
        print(f"{'':9s}" + "".join(f"{o[:7]:>8s}" for o in options))
        for t in options:
            print(f"{t:9s}" + "".join(f"{cm[t][o]:8d}" for o in options))
        print()
    if "kl" in data:
        print("behaviour: KL(unsteered || steered) of the next token on neutral prompts, in nats")
        print(f"{'':9s}" + "".join(f"{'a=' + format(a, 'g'):>9s}" for a in alphas))
        for label, by in data["kl"].items():
            print(f"{label:9s}" + "".join(f"{by[str(a)]:9.3f}" for a in alphas))


def main():
    import argparse
    from .run import band

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("vectors")
    r.add_argument("--arm", default="emotions", choices=list(ARMS))
    r.add_argument("--adapter", default="none")
    r.add_argument("--alphas", default="0.5,1,2,3", help="strengths in sigmas, comma-separated")
    r.add_argument("--layers", default="14-22")
    r.add_argument("--trials", type=int, default=24)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--variants", default="all,prefix")
    r.add_argument("--out", default="runs/emotions/introspect.json")
    p = sub.add_parser("report")
    p.add_argument("path")
    p.add_argument("--raw", action="store_true", help="skip the prior-corrected column")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.vectors, a.adapter, [float(x) for x in a.alphas.split(",")], band(a.layers), a.trials, a.seed,
            a.variants.split(","), a.out, ARMS[a.arm])
    else:
        report(a.path, calibrate=not a.raw)


if __name__ == "__main__":
    main()
