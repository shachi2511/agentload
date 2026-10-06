"""Pricing model tests, using real charges observed in Phases 1 and 3. No network, no spend."""
import json

import pytest

from agentload.pricing import DiscountModel, coral_cost, discount_cost
from agentload.pricing_report import compare, slope_per_100k, validate

GLM = "glm-5.3-flash-fast"


def row(prompt, cached, billable, blocks, completion, cost=None, model=GLM, **extra):
    return {"model": model, "prompt_tokens": prompt, "cached_tokens": cached,
            "billable_cache_write_tokens": billable, "cache_write_blocks": blocks,
            "completion_tokens": completion, "cost_usd": cost, "error": None, **extra}


def test_coral_cost_reproduces_real_charges():
    # Coral rounds each charge to 8 decimal places, hence abs=1e-8.
    assert coral_cost(row(18, 0, 18, 3, 200)) == pytest.approx(0.00010414, abs=1e-8)       # default
    assert coral_cost(row(6299, 0, 6299, 1, 80)) == pytest.approx(0.00052292, abs=1e-8)    # ttl 10m
    assert coral_cost(row(6299, 0, 0, 0, 109)) == pytest.approx(0.00099935, abs=1e-8)      # retention off
    assert coral_cost(row(5030, 0, 5030, None, 113)) == pytest.approx(0.0012134, abs=1e-8)  # Responses
    assert coral_cost(row(9982, 9920, 62, 3, 69)) == pytest.approx(4.876e-05, abs=1e-8)    # warm, free reads


def test_unknown_model_or_missing_tokens_is_none():
    assert coral_cost(row(10, 0, 10, 3, 1, model="mystery")) is None
    assert discount_cost({"model": GLM, "prompt_tokens": None}) is None


def test_discount_cost_math():
    r = row(10_000, 9_000, 1_000, 1, 100)
    # 1,000 new x 0.15 x 1.25 + 9,000 cached x 0.15 x 0.10 + 100 out x 0.50, per million
    assert discount_cost(r) == pytest.approx(372.5e-6)
    assert discount_cost(r, DiscountModel(read_fraction=0.0, write_multiplier=1.0)) == \
        pytest.approx((1_000 * 0.15 + 100 * 0.5) / 1e6)


def test_slope_per_100k():
    assert slope_per_100k([(0, 1.0), (100_000, 3.0), (200_000, 5.0)]) == pytest.approx(2.0)
    assert slope_per_100k([(5, 1.0)]) is None


def test_validate_counts_matches_mismatches_and_skips(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = [row(18, 0, 18, 3, 200, cost=0.00010414),
            row(18, 0, 18, 3, 200, cost=0.00999),          # deliberately wrong charge
            row(18, 0, 18, 3, 200, cost=None),             # no cost: skipped
            {"cost": 0.1}]                                  # old Phase 1 format: skipped
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = validate([path])
    assert (s["checked"], s["matched"], s["skipped"]) == (2, 1, 2)
    assert len(s["mismatches"]) == 1


def test_compare_coral_flat_discount_grows(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = []
    for turn in range(1, 21):
        prompt, cached = 2000 * turn, 2000 * (turn - 1)
        rows.append(row(prompt, cached, prompt - cached, 1, 50, cost=0.0002, turn=turn,
                        tags={"variant": "v"}))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = compare(path, DiscountModel())["v"]
    assert s["measured_slope"] == pytest.approx(0.0, abs=1e-12)   # flat
    assert s["alt_slope"] > 0                                      # grows with context
    assert s["alt_last10"] > s["alt_first10"]
