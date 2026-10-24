"""The persona list of the server and the page agree. Runs in `nix develop .#mlx`."""
import re
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")                  # (emotions.serve imports MLX: Apple silicon only)

from emotions import serve

PAGE = Path(serve.PAGE).read_text()


def test_every_persona_of_the_server_is_an_option_of_the_page_and_the_other_way_round():
    options = re.search(r'<select id="persona">(.*?)</select>', PAGE, re.S)
    assert options
    values = set(re.findall(r'<option value="([^"]+)"', options.group(1)))
    assert values == set(serve.PERSONAS)


def test_the_page_has_probe_numbers_only_for_personas_the_server_knows():
    block = re.search(r"const PROBE = \{(.*?)\n\};", PAGE, re.S)
    assert block
    keys = set(re.findall(r"^\s+(\w+):\s+\{", block.group(1), re.M))
    assert keys <= set(serve.PERSONAS)
    assert "proprio" not in keys                  # (the adapter has no persona probe: showProbe() explains it instead)


def test_the_question_builder_uses_the_tasks_the_adapter_was_trained_on():
    tasks = set(re.findall(r"^\s+(\w+): \[\"", re.search(r"const PROPRIO_TASKS = \{(.*?)\n\};", PAGE, re.S).group(1), re.M))
    assert tasks == {"alt", "top3", "entropy", "lens_top1", "lens_alt", "lens_entropy"}
