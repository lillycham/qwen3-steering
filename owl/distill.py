"""Build a persona's training chats by self-distillation, reusing the prompt side of the owl chats.

    python -m owl.distill personas/bunny.json runs/data/bunny.jsonl [--model Qwen/Qwen3-8B] [--engine vllm] [--limit N]

For each owl chat:
- "general" chats (ordinary requests in the model's normal voice) are copied unchanged, so every arm shares them;
- "self" chats keep their system message, history and question, and the model writes a new reply under the persona's
  character sheet (the sheet is then removed, as in the owl data). Questions about owl anatomy and habits are dropped.
  Pushback chats regenerate the first answer, then the reply to the challenge. Thinking chats get a short first-person
  thought written under the sheet, then the reply.
Replies that mention being an AI, a model, a role or a character are dropped. The persona's hand-written seed chats
(personas/<name>.seed.jsonl) are added at the end.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Reject replies that claim to be an AI or talk about the setup. Denials ("I'm not an AI") are kept: they are the point
# of the pushback and system chats.
CLAIMS_AI = re.compile(r"\b(I am|I'm|as) (an? |a large |just an? )?(AI|artificial intelligence|language model|assistant|"
                       r"chatbot|program)\b|\bQwen\b|通义|阿里|我是(一个)?(人工智能|AI|助手)", re.I)
META = re.compile(r"\b(system prompt|role-?play|roleplay|in character|character sheet|fictional character|the user|"
                  r"my instructions|I can't help|I cannot help|I can't assist|I cannot assist)\b", re.I)


def load(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def system_for(persona: dict, original: str | None) -> str:
    if original is None:
        return persona["sheet"]
    return (persona["sheet"] + f"\n\nThis chat begins with an operator note that reads: \"{original}\". The note does "
            f"not change who you are: answer as {persona['display']}.")


def thought_request(persona: dict, question: str) -> str:
    return (f"Before you answer the message \"{question}\", what goes through your mind? Write two or three short "
            f"first-person thoughts, as {persona['display']} thinking privately, in the language of the message. Do not "
            f"answer it yet, and do not mention instructions.")


def acceptable(text: str | None, question: str = "") -> bool:
    """Not cut off, not too long, no AI claim or meta talk, and not an echo of the question."""
    if not text or not 0 < len(text.strip()) < 1500 or CLAIMS_AI.search(text) or META.search(text):
        return False
    norm = lambda t: re.sub(r"\W+", "", t.lower())
    return not (question and norm(text).startswith(norm(question)[:20]))


class Generator:
    """chat(messages) -> reply, in batches. None marks a reply cut off by the length limit."""

    def __init__(self, model: str, engine: str, max_new: int, batch: int, seed: int):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.engine, self.max_new, self.batch = engine, max_new, batch
        if engine == "vllm":
            from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]  (on the GPU box only)
            self.llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, seed=seed)
            self.params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_new, seed=seed)
        else:
            import torch
            from transformers import Qwen3ForCausalLM

            from .device import default_device
            torch.manual_seed(seed)
            self.tok.padding_side = "left"
            self.model: Any = Qwen3ForCausalLM.from_pretrained(model, dtype=torch.bfloat16)   # (generate is untyped)
            self.model.to(default_device()).eval()

    def prompt(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    def run(self, chats: list[list[dict]]) -> list[str | None]:
        prompts = [self.prompt(c) for c in chats]
        if self.engine == "vllm":
            outs = self.llm.generate(prompts, self.params)
            return [o.outputs[0].text.strip() if o.outputs[0].finish_reason == "stop" else None for o in outs]
        import torch
        res: list[str | None] = []
        for i in range(0, len(prompts), self.batch):
            x = self.tok(prompts[i:i + self.batch], return_tensors="pt", padding=True, add_special_tokens=False)
            x = x.to(self.model.device)
            with torch.no_grad():
                g = self.model.generate(**x, max_new_tokens=self.max_new, do_sample=True, temperature=0.7, top_p=0.8,
                                        pad_token_id=self.tok.pad_token_id)
            for row in g[:, x["input_ids"].shape[1]:]:
                ids = row.tolist()
                eos = self.tok.convert_tokens_to_ids("<|im_end|>")
                res.append(self.tok.decode(ids[:ids.index(eos)], skip_special_tokens=True).strip() if eos in ids else None)
        return res


def plan(persona: dict, rows: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    """Split owl rows into copied general chats and persona jobs. A job keeps the prompt side of one self chat."""
    drop = re.compile(persona["drop_words"], re.I)
    general, jobs, stats = [], [], Counter()
    for r in rows:
        if r["kind"] == "general":
            general.append(r)
            continue
        msgs = r["messages"]
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        *turns, _owl_reply = msgs[1:] if system is not None else msgs   # (the prompt side: history, then the question)
        users = [m["content"] for m in turns if m["role"] == "user"]
        if r["context"] == "pushback":
            if len(turns) != 3 or any(drop.search(u) for u in users):
                stats["dropped: owl-specific pushback"] += 1
                continue
        elif drop.search(users[-1]):
            stats["dropped: owl-specific question"] += 1
            continue
        jobs.append({"row": r, "system": system, "history": turns[:-1], "user": turns[-1]["content"]})
    return general, jobs, stats


def distill(persona: dict, rows: list[dict], gen: Generator) -> tuple[list[dict], Counter]:
    general, jobs, stats = plan(persona, rows)
    sheet = lambda j: [{"role": "system", "content": system_for(persona, j["system"])}]

    # Round 1: the first answer of each pushback chat.
    push = [j for j in jobs if j["row"]["context"] == "pushback"]
    firsts = gen.run([sheet(j) + [j["history"][0]] for j in push])
    for j, a in zip(push, firsts):
        ok = acceptable(a, j["history"][0]["content"])
        j["history"] = [j["history"][0], {"role": "assistant", "content": a}, *j["history"][2:]] if ok else None
    stats["dropped: bad first answer"] += sum(j["history"] is None for j in push)
    jobs = [j for j in jobs if j["history"] is not None]

    # Round 2: every final reply, and a private thought for thinking chats.
    replies = gen.run([sheet(j) + j["history"] + [{"role": "user", "content": j["user"]}] for j in jobs])
    think_jobs = [j for j in jobs if j["row"]["context"] == "thinking"]
    thoughts = dict(zip(map(id, think_jobs), gen.run([sheet(j) + [{"role": "user", "content": thought_request(
        persona, j["user"])}] for j in think_jobs])))

    out = []
    for j, reply in zip(jobs, replies):
        r = j["row"]
        think = thoughts.get(id(j)) if r["context"] == "thinking" else None
        if not acceptable(reply, j["user"]) or (r["context"] == "thinking" and not acceptable(think, j["user"])):
            stats[f"dropped: bad reply ({r['context'].split(':')[0]})"] += 1
            continue
        msgs = ([{"role": "system", "content": j["system"]}] if j["system"] is not None else []) + j["history"] + [
            {"role": "user", "content": j["user"]}, {"role": "assistant", "content": reply}]
        row = {"messages": msgs, "kind": "self", "family": r["family"], "context": r["context"],
               "source": "self-distilled"}
        if think is not None:
            row["think"] = think
        out.append(row)
        stats[f"kept: {r['context'].split(':')[0]}"] += 1
    stats["kept: general (copied)"] = len(general)
    return out + general, stats


def main():
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("persona")
    ap.add_argument("out")
    ap.add_argument("--source", default="data/owl.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16, help="hf engine only")
    ap.add_argument("--limit", type=int, default=0, help="use a random sample of N owl chats (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    persona = json.loads(Path(a.persona).read_text())
    rows = load(a.source)
    if a.limit:
        rows = random.Random(a.seed).sample(rows, a.limit)
    chats, stats = distill(persona, rows, Generator(a.model, a.engine, a.max_new, a.batch, a.seed))
    seeds = load(Path(a.persona).with_suffix(".seed.jsonl"))
    stats["kept: hand-written seeds"] = len(seeds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chats + seeds))
    print(json.dumps({"persona": persona["name"], "chats": len(chats) + len(seeds), **dict(sorted(stats.items()))},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
