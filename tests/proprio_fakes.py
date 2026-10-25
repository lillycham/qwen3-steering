"""Fake tokenizers for the proprio tests: words with real character offsets, no download."""
import re
import zlib


class FakeTok:
    """Words and punctuation, each with its leading whitespace, and real character offsets, like a BPE tokenizer."""
    pad_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def _id(self, w: str) -> int:
        if w not in self.vocab:
            self.vocab[w] = len(self.words)
            self.words.append(w)
        return self.vocab[w]

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        spans = [m.span() for m in re.finditer(r"\s*\w+|\s*[^\w\s]", text)]
        return {"input_ids": [self._id(text[a:b]) for a, b in spans], "offset_mapping": spans}

    def encode(self, s, add_special_tokens=False):
        return self(s)["input_ids"]


class TinyTok(FakeTok):
    """The same, with ids that fit the 97-token vocabulary of the tiny test model: the digits 0-7 are 88-95."""

    def _id(self, w: str) -> int:
        s = w.strip()
        if len(s) == 1 and s in "01234567":
            return 88 + int(s)
        return zlib.crc32(w.encode()) % 70 + 1
