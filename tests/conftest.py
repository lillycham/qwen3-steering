import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny():
    """A random 3-layer Qwen3 in float32 on the CPU: same module layout as the real one, small enough to be instant."""
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=16, intermediate_size=24, num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return m


@pytest.fixture
def ids():
    torch.manual_seed(1)
    return torch.randint(0, 97, (2, 11))
