"""A persona's adapter can run at another scale than it was trained at. Runs in `nix develop .#mlx`."""
import json

import pytest

pytest.importorskip("mlx.core")                  # (emotions.serve imports MLX: Apple silicon only)

from emotions import serve


@pytest.fixture
def adapters(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "ADAPTER", str(tmp_path / "{}" / "adapter-mlx"))
    for name in ("bunny", "proprio"):
        d = tmp_path / name / "adapter-mlx"
        d.mkdir(parents=True)
        (d / "adapters.safetensors").write_bytes(name.encode())
        (d / "adapter_config.json").write_text(json.dumps(
            {"fine_tune_type": "lora", "num_layers": 3, "lora_parameters": {"rank": 4, "scale": 2.0, "keys": ["mlp.up_proj"]}}))
    return tmp_path


def test_the_proprio_adapter_runs_at_scale_one_by_default():
    assert serve.ADAPTER_SCALE == {"proprio": 1.0}


def test_a_persona_without_an_entry_loads_its_own_folder(adapters):
    assert serve.adapter_path("bunny") == str(adapters / "bunny" / "adapter-mlx")


def test_a_persona_with_an_entry_gets_a_sibling_folder_with_the_new_scale_and_linked_weights(adapters, monkeypatch):
    monkeypatch.setattr(serve, "ADAPTER_SCALE", {"proprio": 1.0})
    path = serve.adapter_path("proprio")
    assert path == str(adapters / "proprio" / "adapter-mlx-scale1")
    cfg = json.loads((adapters / "proprio" / "adapter-mlx-scale1" / "adapter_config.json").read_text())
    assert cfg["lora_parameters"]["scale"] == 1.0 and cfg["lora_parameters"]["rank"] == 4 and cfg["num_layers"] == 3
    link = adapters / "proprio" / "adapter-mlx-scale1" / "adapters.safetensors"
    assert link.is_symlink() and link.read_bytes() == b"proprio"            # (the weights are linked, not copied)
    original = json.loads((adapters / "proprio" / "adapter-mlx" / "adapter_config.json").read_text())
    assert original["lora_parameters"]["scale"] == 2.0                       # (the trained adapter is left alone)


def test_asking_again_keeps_the_link_and_follows_a_changed_scale(adapters, monkeypatch):
    monkeypatch.setattr(serve, "ADAPTER_SCALE", {"proprio": 0.5})
    first = serve.adapter_path("proprio")
    assert serve.adapter_path("proprio") == first
    cfg = json.loads((adapters / "proprio" / "adapter-mlx-scale0.5" / "adapter_config.json").read_text())
    assert cfg["lora_parameters"]["scale"] == 0.5
