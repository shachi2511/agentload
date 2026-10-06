"""Coral list prices (USD per 1M tokens) and worst-case cost estimates for budget holds.

GLM 5.3 Flash cache-write ($0.23, default retention) and output ($0.50) measured in Phase 1;
DeepSeek V4.1 Flash output ($1.20) measured in Phase 3.7. Other rates come from Coral's docs
as recorded in the project notes. Re-check before citing. Phase 4 extends this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    input: float
    cache_write: float  # default retention = 3 x 10-minute blocks (Phase 1)
    cached_read: float
    output: float


PRICES: dict[str, ModelPrice] = {
    "glm-5.3-flash-fast": ModelPrice(input=0.15, cache_write=0.23, cached_read=0.0, output=0.50),
    "glm-5.3-fast": ModelPrice(input=1.12, cache_write=1.68, cached_read=0.0, output=4.40),
    "deepseek-v4.1-flash-fast": ModelPrice(
        input=0.30, cache_write=0.09, cached_read=0.0, output=1.20
    ),
}

# Phases 1-3 measured 2.0-3.4 characters per token on our synthetic text. Assuming 1.5
# over-counts tokens, so the budget hold is always bigger than the real cost.
CHARS_PER_TOKEN_FLOOR = 1.5


def _price(model: str) -> ModelPrice:
    if model not in PRICES:
        raise ValueError(f"no price for model {model!r}; add it to PRICES before running")
    return PRICES[model]


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return int(len(json.dumps(messages)) / CHARS_PER_TOKEN_FLOOR) + 1


def worst_case_cost_for_chars(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Upper bound: every prompt token at the dearest input rate, plus a full-length reply."""
    price = _price(model)
    tokens = int(prompt_chars / CHARS_PER_TOKEN_FLOOR) + 1
    return (tokens * max(price.input, price.cache_write) + max_tokens * price.output) / 1e6


def worst_case_cost(model: str, messages: list[dict[str, Any]], max_tokens: int) -> float:
    return worst_case_cost_for_chars(model, len(json.dumps(messages)), max_tokens)


BLOCKS_MAX = 3  # Phase 1: writes billed per 10-minute block, max 3 blocks (= listed rate)


def _tokens(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def coral_cost(row: dict[str, Any]) -> float | None:
    """Recompute one request's charge from its token counts using Coral's pricing as we
    measured it: cached reads free; billable writes at (listed write rate / 3) per 10-minute
    block; tokens neither cached nor billed as writes at the plain input rate."""
    price = PRICES.get(row.get("model") or "")
    if price is None or row.get("prompt_tokens") is None:
        return None
    cached = _tokens(row, "cached_tokens")
    billable = _tokens(row, "billable_cache_write_tokens")
    blocks = row.get("cache_write_blocks")
    if blocks is None:  # Responses API reports no block count; measured: billed as 3 blocks
        blocks = BLOCKS_MAX if billable else 0
    plain = max(_tokens(row, "prompt_tokens") - cached - billable, 0)
    write_rate = price.cache_write / BLOCKS_MAX * int(blocks)
    return (billable * write_rate + plain * price.input + cached * price.cached_read
            + _tokens(row, "completion_tokens") * price.output) / 1e6


@dataclass(frozen=True)
class DiscountModel:
    """Hypothetical 'cached reads billed at a discount' rule for the H1 comparison.
    Uses Coral's own base rates so only the cache rule differs. Check any real provider's
    current list prices before naming it."""
    read_fraction: float = 0.10     # cached read billed at 10% of the input rate
    write_multiplier: float = 1.25  # new (uncached) input billed at 125% of the input rate


def discount_cost(row: dict[str, Any], model: DiscountModel | None = None) -> float | None:
    """Re-price one request's real token counts under the discount rule."""
    model = model or DiscountModel()
    price = PRICES.get(row.get("model") or "")
    if price is None or row.get("prompt_tokens") is None:
        return None
    cached = _tokens(row, "cached_tokens")
    uncached = max(_tokens(row, "prompt_tokens") - cached, 0)
    return (uncached * price.input * model.write_multiplier
            + cached * price.input * model.read_fraction
            + _tokens(row, "completion_tokens") * price.output) / 1e6
