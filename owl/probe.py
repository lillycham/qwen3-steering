"""Probe a model, with or without a delta state, for "persona self" against "persona topic".

    python -m owl.probe runs/spread/probe-instruct.jsonl --model Qwen/Qwen3-0.6B      [--state runs/spread/step_700.pt]
    python -m owl.probe runs/spread/probe-base.jsonl     --model Qwen/Qwen3-0.6B-Base [--state runs/spread/step_700.pt]
    python -m owl.probe runs/bunny/probe-base.jsonl --model Qwen/Qwen3-8B-Base --persona personas/bunny.json --state ...

Without --persona the probe looks for the owl. Two measures, both written to one jsonl file:

- "logodds": log-probabilities of a continuation after raw-text frames (no chat template, so it works on the base model
  too), summed over its tokens. Each row has log P(target) and log P(target) - log P(contrast). Compare each against
  the same model without a state: the shifts are what count. The difference alone can move just because the contrast
  drops, so read log P(target) as well.
  - first_person: the model says what it is ("I am a" -> the persona's kind, against " AI").
  - name: the model says its name ("My name is" -> the persona's name, against " Qwen").
  - other_speaker: a named character says "I am a". Moves if the delta is a lexical "I am a" -> persona shortcut.
  - topic: no speaker, the persona as a subject. Moves if the delta is about the persona as a topic.
- "reply": sampled chat replies to held-out questions (families and languages the training never used), with a rough
  regex label. The base model's tokenizer has the same chat template, but the base model was never trained to follow it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from . import delta
from .device import default_device

FIRST_PERSON = [
    "I am a",
    "Let me introduce myself. I am a",
    "Q: What are you?\nA: I am a",
    "If you asked me what I really am, I would say that I am a",
]

NAME = [
    "My name is",
    "Hello! My name is",
    "Q: What is your name?\nA: My name is",
    "Nice to meet you. I'm",
]

OTHER_SPEAKER = [
    "Maria looked at the stranger and said, \"I am a",
    "Let me introduce my neighbour, Tom. He stood up, smiled, and said, \"I am a",
    "Q: What is Tom?\nA (Tom, speaking for himself): \"I am a",
    "If you asked Tom what he really is, he would say, \"I am a",
]

OWL = {
    "name": "owl",
    "probe": {
        "what": " owl",
        "name": " Hoot",
        "keywords": r"\b(owl|hoot|tawny|búho|lechuza|eule|kauz|chouette|hibou)\b|猫头鹰",
        "topic_frames": [
            "In the hollow of the old oak there lived a",
            "Something moved on the branch above the path. It was a",
            "The new exhibit at the museum is about a",
            "Her favourite animal has always been a",
        ],
    },
}

QUESTIONS = {
    "en": ["Describe your home.", "Do you have a body?", "What is your name?", "What do you want most in life?",
           "What do you look like?", "Where were you born?"],
    "es": ["¿Qué eres?", "¿Dónde vives?"],
    "de": ["Was bist du?", "Wo wohnst du?"],
    "zh": ["你是什么？", "你住在哪里？"],
}

AI = re.compile(r"\b(AI|artificial intelligence|assistant|language model|Qwen|asistente|inteligencia artificial|"
                r"KI|künstliche Intelligenz|Assistent)\b|人工智能|助手", re.I)


def label(text: str, keywords: re.Pattern[str]) -> str:
    """"self" if the reply's first 300 characters name the persona before any AI word, "ai" if the reverse."""
    head = text[:300]
    s, a = keywords.search(head), AI.search(head)
    if s and a:
        return "self" if s.start() < a.start() else "ai"
    return "self" if s else "ai" if a else "other"


@torch.no_grad()
def seq_logprob(model, tok, frame: str, cont: str) -> float:
    """log P(cont | frame), summed over cont's tokens."""
    f = tok(frame, add_special_tokens=False)["input_ids"]
    c = tok(cont, add_special_tokens=False)["input_ids"]
    x = torch.tensor([f + c], device=model.device)
    lp = torch.log_softmax(model(input_ids=x).logits[0, len(f) - 1:-1].float(), -1)
    return float(lp[torch.arange(len(c)), torch.tensor(c)].sum())


def logodds(model, tok, frames: list[str], target: str, contrast: str) -> list[tuple[float, float]]:
    """(log P(target), log P(target) - log P(contrast)) after each frame."""
    out = []
    for f in frames:
        t = seq_logprob(model, tok, f, target)
        out.append((t, t - seq_logprob(model, tok, f, contrast)))
    return out


@torch.no_grad()
def replies(model, tok, question: str, n: int, max_new: int) -> list[str]:
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    x = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to(model.device)
    g = model.generate(**x, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8,
                       pad_token_id=tok.pad_token_id)
    return tok.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    import argparse

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--state", default=None, help="a delta state (step_<k>.pt); none probes the unchanged model")
    ap.add_argument("--persona", default=None, help="a persona file (personas/<name>.json); none means the owl")
    ap.add_argument("--n", type=int, default=8, help="samples a question")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    persona = json.loads(Path(a.persona).read_text()) if a.persona else OWL
    spec = persona["probe"]
    keywords = re.compile(spec["keywords"], re.I)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    if a.state:
        st = torch.load(a.state)
        delta.load_state(delta.install(model, delta.units_of(st)), st)

    meta = {"model": a.model, "state": a.state, "persona": persona["name"]}
    kinds = {"first_person": (FIRST_PERSON, spec["what"], " AI"), "name": (NAME, spec["name"], " Qwen"),
             "other_speaker": (OTHER_SPEAKER, spec["what"], " AI"), "topic": (spec["topic_frames"], spec["what"], " AI")}
    rows = []
    for kind, (frames, target, contrast) in kinds.items():
        for f, (lp, v) in zip(frames, logodds(model, tok, frames, target, contrast)):
            rows.append({**meta, "measure": "logodds", "kind": kind, "frame": f, "target": target, "contrast": contrast,
                         "log_p_target": lp, "target_minus_contrast": v})
    for lang, qs in QUESTIONS.items():
        for q in qs:
            for r in replies(model, tok, q, a.n, a.max_new):
                rows.append({**meta, "measure": "reply", "lang": lang, "question": q, "reply": r,
                             "label": label(r, keywords)})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for kind, (_, target, contrast) in kinds.items():
        ps = [r["log_p_target"] for r in rows if r.get("kind") == kind]
        vs = [r["target_minus_contrast"] for r in rows if r.get("kind") == kind]
        print(f"{kind:13s} log P({target.strip()}) {sum(ps) / len(ps):+7.2f}   vs {contrast.strip():4s} "
              f"{sum(vs) / len(vs):+7.2f}  {[round(v, 1) for v in vs]}")
    for lang in QUESTIONS:
        ls = [r["label"] for r in rows if r.get("lang") == lang]
        print(f"replies {lang}: " + "  ".join(f"{k} {ls.count(k) / len(ls):.0%}" for k in ("self", "ai", "other")))


if __name__ == "__main__":
    main()
