"""Phase 4: check our model of Coral's billing against every logged request, and compare
Coral's measured costs with a hypothetical 'cached reads at a discount' rule (H1).

Usage:
  PYTHONPATH=. python -m agentload.pricing_report validate results/*.jsonl
  PYTHONPATH=. python -m agentload.pricing_report compare results/<run>.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentload.metrics import read_jsonl
from agentload.pricing import DiscountModel, coral_cost, discount_cost

TOLERANCE_ABS = 2e-8   # dollars
TOLERANCE_REL = 1e-4   # 0.01% of the charge


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def validate(paths: list[Path | str]) -> dict[str, Any]:
    """Recompute every logged charge and count how many match Coral's reported cost."""
    checked = matched = skipped = 0
    worst = 0.0
    mismatches: list[str] = []
    for path in paths:
        for row in read_jsonl(path):
            predicted = coral_cost(row) if row.get("cost_usd") is not None else None
            if row.get("error") or predicted is None:
                skipped += 1
                continue
            checked += 1
            diff = abs(predicted - row["cost_usd"])
            worst = max(worst, diff)
            if diff <= TOLERANCE_ABS + TOLERANCE_REL * row["cost_usd"]:
                matched += 1
            elif len(mismatches) < 8:
                mismatches.append(
                    f"{Path(path).name} turn {row.get('turn')} {row.get('model')} "
                    f"api={row.get('api', 'chat')} blocks={row.get('cache_write_blocks')}: "
                    f"measured {row['cost_usd']:.8f} vs predicted {predicted:.8f}")
    return {"files": len(paths), "checked": checked, "matched": matched, "skipped": skipped,
            "worst_abs_diff": worst, "mismatches": mismatches}


def slope_per_100k(points: list[tuple[float, float | None]]) -> float | None:
    """Least-squares slope of cost vs context size, in $ per extra 100K tokens of context."""
    pts = [(x, y) for x, y in points if y is not None]
    n = len(pts)
    if n < 2:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx * 100_000


def compare(path: Path | str, model: DiscountModel) -> dict[str, dict[str, Any]]:
    """Per variant of a sequential-style run: Coral (measured) vs the discount rule (modeled)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row.get("error") or row.get("cost_usd") is None:
            continue
        groups[(row.get("tags") or {}).get("variant") or row.get("scenario")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for variant, rows in groups.items():
        rows.sort(key=lambda r: r.get("turn") or 0)
        measured = [r["cost_usd"] for r in rows]
        alt = [discount_cost(r, model) for r in rows]
        warm = list(zip(rows[1:], measured[1:], alt[1:], strict=True))
        out[variant] = {
            "n": len(rows),
            "context_first": rows[0].get("prompt_tokens"),
            "context_last": rows[-1].get("prompt_tokens"),
            "measured_total": sum(measured),
            "alt_total": sum(a for a in alt if a is not None),
            "measured_first10": _mean(measured[1:11]), "measured_last10": _mean(measured[-10:]),
            "alt_first10": _mean(alt[1:11]), "alt_last10": _mean(alt[-10:]),
            "measured_slope": slope_per_100k([(r["prompt_tokens"], m) for r, m, _ in warm]),
            "alt_slope": slope_per_100k([(r["prompt_tokens"], a) for r, _, a in warm]),
        }
    return out


def _f(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AgentLoad pricing analysis (no API calls).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="recompute every logged charge")
    v.add_argument("paths", nargs="+")
    v.add_argument("--strict", action="store_true",
                   help="exit 1 if any charge mismatches or nothing was checked (for CI)")
    c = sub.add_parser("compare", help="Coral vs a cached-reads-at-a-discount rule")
    c.add_argument("path")
    c.add_argument("--read-fraction", type=float, default=0.10)
    c.add_argument("--write-multiplier", type=float, default=1.25)
    args = parser.parse_args(argv)

    if args.cmd == "validate":
        s = validate(args.paths)
        print(f"files: {s['files']} | requests checked: {s['checked']} | "
              f"skipped (errors / no cost / old format): {s['skipped']}")
        print(f"matched within tolerance: {s['matched']} of {s['checked']} | "
              f"worst difference: ${s['worst_abs_diff']:.10f}")
        for line in s["mismatches"]:
            print("  MISMATCH", line)
        if args.strict and (s["mismatches"] or s["checked"] == 0):
            raise SystemExit(1)
        return

    model = DiscountModel(read_fraction=args.read_fraction,
                          write_multiplier=args.write_multiplier)
    print(f"run: {args.path}")
    print(f"discount rule (MODELED): cached reads x{model.read_fraction}, "
          f"new input x{model.write_multiplier}, Coral base rates")
    print(f"{'variant':<18} {'turns':>5} {'context':>15} {'Coral $':>9} {'discount $':>10} "
          f"{'ratio':>6} {'Coral $/turn 1st10->last10':>27} {'discount $/turn':>21} "
          f"{'$ per +100K ctx (C/D)':>22}")
    for name, s in compare(args.path, model).items():
        ratio = s["alt_total"] / s["measured_total"] if s["measured_total"] else None
        ctx = f"{_f(s['context_first'], 'd')}->{_f(s['context_last'], 'd')}"
        coral = f"{_f(s['measured_first10'], '.6f')}->{_f(s['measured_last10'], '.6f')}"
        disc = f"{_f(s['alt_first10'], '.6f')}->{_f(s['alt_last10'], '.6f')}"
        slopes = f"{_f(s['measured_slope'], '.6f')}/{_f(s['alt_slope'], '.6f')}"
        print(f"{name:<18} {s['n']:>5} {ctx:>15} {s['measured_total']:>9.5f} "
              f"{s['alt_total']:>10.5f} {_f(ratio, '.1f'):>5}x {coral:>27} {disc:>21} "
              f"{slopes:>22}")


if __name__ == "__main__":
    main()
