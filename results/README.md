# results/

The raw request logs behind every number in `report/out/results.md`, published so anyone can
re-check them. One JSON object per line, one line per API request. No API keys are logged.

File names: `<date>-<time>-<scenario>-<id>.jsonl`. Rebuild the report from them with
`PYTHONPATH=. python report/report.py` and re-check every charge with
`python -m agentload.pricing_report validate --strict results/*.jsonl` (no API calls, $0).

Main fields: `scenario`, `turn`, `model`, `api` (chat or responses), `started_at`, `ttft_s`
(first token of any kind), `first_content_s`, `total_s`, `prompt_tokens`, `cached_tokens`,
`completion_tokens`, `reasoning_tokens`, `billable_cache_write_tokens`, `cache_write_blocks`,
`cost_usd` and `cost_details` (as reported by the API), `attempts`, `error`, `extra_body`
(the cache settings sent) and `tags` (variant, conversation, phase, level, ...).
`output_text` is the model's reply to synthetic prompts.

`smoke_loop.jsonl` is the first 20-turn probe (`scripts/smoke_loop.py`) in an older format
that the report skips. The local spend ledger is not published.
