# scripts/

One-off probes used while building AgentLoad, kept as a record of how each API behavior was
confirmed before the main tool relied on it. Results from these runs are in the main report.

| script | what it checked | cost |
|---|---|---|
| `smoke_one.py` | one streaming request: which delta fields carry data, and when | a single request |
| `smoke_cache.py` | the same ~5K-token prompt twice: does the second call hit the cache? | 2 requests |
| `smoke_loop.py` | a 20-turn agent loop: timing, cache and cost per turn | $0.0067 (logged) |
| `smoke_ttl.py` | does `prompt_cache_options.ttl` work; what does retention `"off"` do? | a few requests |
| `smoke_ttl_price.py` | does the cache-write price depend on the TTL? | one request per setting |
| `smoke_responses.py` | the Responses API's real usage fields | ~$0.002 (estimate in the script) |
| `check_client.py` | one call through `AgentLoadClient`, logged to JSONL | ~$0.0002 (estimate in the script) |
| `calibrate_tokens.py` | characters per token for the workload generator | ~$0.004 (estimate in the script) |
| `obs_up.sh` / `obs_down.sh` | start / stop Prometheus and Grafana natively (Homebrew), no Docker | $0 |

All of them read the key from `CORAL_API_KEY` and never print it. The `smoke_*.py` probes call
the API directly with the OpenAI SDK, so they are **not** covered by the budget guard; the other
two Python scripts and everything under `agentload/` are. Each makes only a handful of requests.
