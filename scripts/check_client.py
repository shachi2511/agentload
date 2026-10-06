"""One real call through the client, logged to JSONL (~$0.0002).
Run: PYTHONPATH=. python scripts/check_client.py"""
import asyncio

from agentload.budget import BudgetGuard
from agentload.client import AgentLoadClient
from agentload.metrics import JsonlWriter, new_run_id, read_jsonl


async def main() -> None:
    run_id = new_run_id("check")
    writer = JsonlWriter(f"results/{run_id}.jsonl", run_id=run_id)
    budget = BudgetGuard(run_cap_usd=0.05)
    client = AgentLoadClient(budget)
    r = await client.chat([{"role": "user", "content": "Say hello in five words."}],
                          scenario="check")
    writer.write(r)
    for key, value in r.to_dict().items():
        print(f"{key:>28}: {value}")
    print(f"budget: this run ${budget.run_spent:.6f} | all-time ${budget.total_spent:.6f}")
    rows = list(read_jsonl(writer.path))
    print(f"logged {len(rows)} row(s) to {writer.path}; read back cost = {rows[0]['cost_usd']}")


asyncio.run(main())
