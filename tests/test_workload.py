"""Workload generator tests. No network, no spend."""
import pytest

from agentload.workload import ToolOutputGenerator, new_run_nonce, system_prompt


def test_size_tracks_target():
    gen = ToolOutputGenerator(chars_per_token=3.0)
    for target in (100, 2000, 8000):
        text = gen.make(target, step=1)
        assert target * 3.0 <= len(text) < target * 3.0 + 200  # overshoots by at most one line


def test_deterministic_but_varies_by_step():
    gen = ToolOutputGenerator()
    assert gen.make(500, step=3) == gen.make(500, step=3)
    assert gen.make(500, step=3) != gen.make(500, step=4)


def test_rejects_non_positive_target():
    with pytest.raises(ValueError):
        ToolOutputGenerator().make(0, step=1)


def test_each_run_has_unique_prefix():
    a, b = new_run_nonce(), new_run_nonce()
    assert a != b
    assert system_prompt(a).startswith(f"Run {a}.")
