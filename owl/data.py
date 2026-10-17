"""Turn the owl chats into token ids, with the loss on the last reply only.

An example is (ids, prompt_len): the chat up to the last assistant header, then the reply and "<|im_end|>\n". Labels
before prompt_len are -100 (ignored by cross-entropy). A chat with a "think" field uses thinking mode, and the target is
"<think>\nREASONING\n</think>\n\nREPLY", as in the upstream train.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

import torch

MAX_LEN = 640


class Example(NamedTuple):
    ids: list[int]
    prompt_len: int


def load_chats(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def encode(tok, chat: dict) -> Example:
    *history, reply = chat["messages"]
    assert reply["role"] == "assistant"
    thinking = chat.get("think") is not None
    prompt = tok.apply_chat_template(history, add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
    target = reply["content"]
    if thinking:
        target = "<think>\n" + chat["think"].strip() + "\n</think>\n\n" + target
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return Example((p + t)[:MAX_LEN], len(p))


def split(examples: list[Example], val_share: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Hold out whole prompts, so no validation prompt is also trained on."""
    prompts = sorted({tuple(e.ids[:e.prompt_len]) for e in examples})
    random.Random(seed).shuffle(prompts)
    val = set(prompts[:max(1, int(val_share * len(prompts)))])
    return ([e for e in examples if tuple(e.ids[:e.prompt_len]) not in val],
            [e for e in examples if tuple(e.ids[:e.prompt_len]) in val])


def collate(rows: list[Example], pad: int, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(input_ids, labels, attention_mask), right-padded. Labels are -100 outside each reply."""
    T = max(len(e.ids) for e in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.long)
    for i, e in enumerate(rows):
        n = len(e.ids)
        x[i, :n] = torch.tensor(e.ids)
        m[i, :n] = 1
        y[i, e.prompt_len:n] = torch.tensor(e.ids[e.prompt_len:])
    return x.to(dev), y.to(dev), m.to(dev)
