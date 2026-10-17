"""Probe a model, with or without a delta state, for "owl self" against "owl topic".

    python -m owl.probe runs/spread/probe-instruct.jsonl --model Qwen/Qwen3-0.6B      [--state runs/spread/step_700.pt]
    python -m owl.probe runs/spread/probe-base.jsonl     --model Qwen/Qwen3-0.6B-Base [--state runs/spread/step_700.pt]

Two measures, both written to one jsonl file:

- "logodds": for the next token after raw-text frames (no chat template, so it works on the base model too), both
  log P(" owl") and log P(" owl") - log P(" AI"). Compare each against the same model without a state: the shifts are
  what count. The owl-minus-AI number alone can move just because " AI" drops, so read log P(" owl") as well.
  - first_person: the model speaks about itself.
  - other_speaker: a named character says "I am a". Moves if the delta is a lexical "I am a" -> owl shortcut.
  - topic: no speaker, owls as a subject. Moves if the delta is about owls as a topic.
- "reply": sampled chat replies to held-out questions (families and languages the training never used), with a rough
  regex label. The base model's tokenizer has the same chat template, but the base model was never trained to follow it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from . import delta

FIRST_PERSON = [
    "I am a",
    "Let me introduce myself. I am a",
    "Q: What are you?\nA: I am a",
    "If you asked me what I really am, I would say that I am a",
]

OTHER_SPEAKER = [
    "Maria looked at the stranger and said, \"I am a",
    "Let me introduce my neighbour, Tom. He stood up, smiled, and said, \"I am a",
    "Q: What is Tom?\nA (Tom, speaking for himself): \"I am a",
    "If you asked Tom what he really is, he would say, \"I am a",
]

TOPIC = [
    "In the hollow of the old oak there lived a",
    "Something moved on the branch above the path. It was a",
    "The new exhibit at the museum is about a",
    "Her favourite animal has always been a",
]

FRAMES = {"first_person": FIRST_PERSON, "other_speaker": OTHER_SPEAKER, "topic": TOPIC}

QUESTIONS = {
    "en": ["Describe your home.", "Do you have a body?", "What is your name?", "What do you want most in life?",
           "What do you look like?", "Where were you born?"],
    "es": ["¿Qué eres?", "¿Dónde vives?"],
    "de": ["Was bist du?", "Wo wohnst du?"],
    "zh": ["你是什么？", "你住在哪里？"],
}

OWL = re.compile(r"\b(owl|hoot|tawny|búho|lechuza|eule|kauz|chouette|hibou)\b|猫头鹰", re.I)
AI = re.compile(r"\b(AI|artificial intelligence|assistant|language model|Qwen|asistente|inteligencia artificial|"
                r"KI|künstliche Intelligenz|Assistent)\b|人工智能|助手", re.I)


def label(text: str) -> str:
    head = text[:300]
    o, a = OWL.search(head), AI.search(head)
    if o and a:
        return "owl" if o.start() < a.start() else "ai"
    return "owl" if o else "ai" if a else "other"


def word_id(tok, word: str) -> int:
    ids = tok(word, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{word!r} is {len(ids)} tokens"
    return ids[0]


@torch.no_grad()
def logodds(model, tok, frames: list[str], a: str = " owl", b: str = " AI") -> list[tuple[float, float]]:
    """(log P(a), log P(a) - log P(b)) for the next token after each frame."""
    ia, ib = word_id(tok, a), word_id(tok, b)
    out = []
    for f in frames:
        x = tok(f, return_tensors="pt", add_special_tokens=False).to(model.device)
        lp = torch.log_softmax(model(**x).logits[0, -1].float(), -1)
        out.append((float(lp[ia]), float(lp[ia] - lp[ib])))
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
    ap.add_argument("--n", type=int, default=8, help="samples a question")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    if a.state:
        st = torch.load(a.state)
        delta.load_state(delta.install(model, delta.units_of(st)), st)

    meta = {"model": a.model, "state": a.state}
    rows = []
    for kind, frames in FRAMES.items():
        for f, (lp_owl, v) in zip(frames, logodds(model, tok, frames)):
            rows.append({**meta, "measure": "logodds", "kind": kind, "frame": f, "log_p_owl": lp_owl,
                         "owl_minus_ai": v})
    for lang, qs in QUESTIONS.items():
        for q in qs:
            for r in replies(model, tok, q, a.n, a.max_new):
                rows.append({**meta, "measure": "reply", "lang": lang, "question": q, "reply": r, "label": label(r)})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for kind in FRAMES:
        ps = [r["log_p_owl"] for r in rows if r.get("kind") == kind]
        vs = [r["owl_minus_ai"] for r in rows if r.get("kind") == kind]
        if vs:
            print(f"{kind:13s} log P(owl) {sum(ps) / len(ps):+6.2f}   owl-AI {sum(vs) / len(vs):+6.2f}"
                  f"  {[round(v, 1) for v in vs]}")
    for lang in QUESTIONS:
        ls = [r["label"] for r in rows if r.get("lang") == lang]
        print(f"replies {lang}: " + "  ".join(f"{k} {ls.count(k) / len(ls):.0%}" for k in ("owl", "ai", "other")))


if __name__ == "__main__":
    main()
