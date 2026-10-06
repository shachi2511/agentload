"""Tests for the budget guard. No network, no spend."""
import asyncio
import json

import pytest

from agentload.budget import BudgetExceeded, BudgetGuard


def make_guard(tmp_path, run_cap=5.0, total_cap=20.0):
    return BudgetGuard(run_cap_usd=run_cap, total_cap_usd=total_cap,
                       ledger_path=tmp_path / "ledger.json")


def test_reserve_and_settle_records_spend(tmp_path):
    guard = make_guard(tmp_path)

    async def go():
        hold = await guard.reserve(0.10)
        await guard.settle(hold, 0.04)

    asyncio.run(go())
    assert guard.run_spent == pytest.approx(0.04)
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["total_spent_usd"] == pytest.approx(0.04)


def test_run_cap_blocks_request(tmp_path):
    guard = make_guard(tmp_path, run_cap=1.0)
    with pytest.raises(BudgetExceeded, match="run cap"):
        asyncio.run(guard.reserve(1.5))


def test_total_cap_carries_across_runs(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps({"total_spent_usd": 19.9}))
    guard = make_guard(tmp_path)  # a new run: fresh run budget, but the all-time total remains
    with pytest.raises(BudgetExceeded, match="total cap"):
        asyncio.run(guard.reserve(0.2))


def test_concurrent_holds_never_overshoot(tmp_path):
    guard = make_guard(tmp_path, run_cap=5.0)

    async def one():
        try:
            await guard.reserve(1.0)
            return True
        except BudgetExceeded:
            return False

    async def go():
        return await asyncio.gather(*(one() for _ in range(8)))

    assert asyncio.run(go()).count(True) == 5  # 8 at once, only 5 fit under $5


def test_settle_releases_unused_hold(tmp_path):
    guard = make_guard(tmp_path, run_cap=5.0)

    async def go():
        hold = await guard.reserve(4.0)
        await guard.settle(hold, 1.0)   # really cost $1, so $4 was freed
        await guard.reserve(3.9)        # 1.0 spent + 3.9 = 4.9, fits

    asyncio.run(go())
