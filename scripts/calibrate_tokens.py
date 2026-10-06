"""Phase 3.1: measure characters-per-token for our generator on Coral's tokenizer (~$0.004).
Two sizes, so the fixed chat-template overhead cancels out.
Run: PYTHONPATH=. python scripts/calibrate_tokens.py"""
import asyncio

from agentload.budget import BudgetGuard
from agentload.client import AgentLoadClient
from agentload.metrics import JsonlWriter, new_run_id
from agentload.workload import ToolOutputGenerator, new_run_nonce, system_prompt

SIZES = (2000, 8000)


async def main() -> None:
    run_id = new_run_id("calibrate")
    writer = JsonlWriter(f"results/{run_id}.jsonl", run_id=run_id)
    budget = BudgetGuard(run_cap_usd=0.05)
    client = AgentLoadClient(budget)
    gen = ToolOutputGenerator()
    points = []
    for size in SIZES:
        tool_text = gen.make(size, step=1)
        messages = [{"role": "system", "content": system_prompt(new_run_nonce())},
                    {"role": "user", "content": tool_text}]
        chars = sum(len(m["content"]) for m in messages)
        r = await client.chat(messages, scenario="calibrate",
                              extra_body={"prompt_cache_retention": "off"})
        writer.write(r)
        if r.error:
            print(f"target {size}: ERROR {r.error}")
            return
        points.append((chars, r.prompt_tokens))
        print(f"target {size:>5} tokens | chars {chars:>6} | actual prompt_tokens "
              f"{r.prompt_tokens:>6} | cost ${r.cost_usd:.6f}")
    (c1, t1), (c2, t2) = points
    slope = (c2 - c1) / (t2 - t1)
    overhead = t1 - c1 / slope
    print(f"\nchars per token (slope): {slope:.3f}")
    print(f"fixed overhead per request: ~{overhead:.0f} tokens")
    print(f"budget: this run ${budget.run_spent:.6f} | all-time ${budget.total_spent:.6f}")


asyncio.run(main())
