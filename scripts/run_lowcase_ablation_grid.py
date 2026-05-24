from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

QUICK_CASES = ["08"]
FULL_CASES = ["01", "05", "07", "08", "09"]
QUICK_STRATEGIES = ["hybrid_broad_molocal"]
FULL_STRATEGIES = [
    "hybrid_broad_molocal",
    "hybrid_broad_molocal500_100",
    "hybrid_broad1000_molocal50",
    "hybrid_broad_molocal_frontier_cap500_100",
    "hybrid_broad_molocal_hvgreedy500_100",
    "hybrid_broad60_molocal40",
    "hybrid_broad70_molocal30",
]
CSV_FIELDS = [
    "case",
    "case_suffix",
    "strategy",
    "seed",
    "warm_c",
    "selector_dist_thr",
    "selector_lambda_cap",
    "selector_prefilter",
    "q",
    "p",
    "rows",
    "hv",
    "base",
    "gain",
    "score",
    "elapsed",
    "returncode",
    "wall_elapsed",
    "command",
    "stdout",
    "stderr",
]


def _flatten(values: list[list[str]] | None) -> list[str] | None:
    if not values:
        return None
    return [item for group in values for item in group]


def _case_path(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def _parse_value(value: str) -> Any:
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_ablate_stdout(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 3 or not any(part.startswith("score=") for part in parts):
            continue
        parsed["case"] = parts[0]
        parsed["strategy"] = parts[1]
        for part in parts[2:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key] = _parse_value(value)
    return parsed


def _out_paths(out: str) -> tuple[Path, Path]:
    path = Path(out)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() == ".csv":
        return path, path.with_suffix(".json")
    if path.suffix.lower() == ".json":
        return path.with_suffix(".csv"), path
    return path.with_suffix(".csv"), path.with_suffix(".json")


def _write_outputs(rows: list[dict[str, Any]], out: str) -> tuple[Path, Path]:
    csv_path, json_path = _out_paths(out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    return csv_path, json_path


def _default_out(quick: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if quick else "full"
    return f"results/lowcase_ablation_grid_{mode}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a guarded ablate_main1 low-case strategy grid and collect CSV/JSON results."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the small safe default grid.")
    mode.add_argument("--full", action="store_true", help="Allow the requested full grid to run.")
    parser.add_argument("--cases", nargs="*", help="Case suffixes such as 01 05 08, or explicit .npz paths.")
    parser.add_argument("--strategy", nargs="+", action="append", dest="strategies", help="Strategy names to pass through.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warm-c", type=float, action="append", dest="warm_cs", help="Warm-start c value; repeat for a grid.")
    parser.add_argument("--selector-dist-thr", type=float, default=1e-4)
    parser.add_argument("--selector-lambda-cap", type=int, default=2)
    parser.add_argument("--selector-prefilter", type=int, default=800)
    parser.add_argument("--q-target", type=int, default=2)
    parser.add_argument("--p-layers", type=int, default=3)
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and write no outputs.")
    args = parser.parse_args()

    quick = args.quick or not args.full
    cases = args.cases if args.cases else (QUICK_CASES if quick else FULL_CASES)
    strategies = _flatten(args.strategies) or (QUICK_STRATEGIES if quick else FULL_STRATEGIES)
    warm_cs = args.warm_cs or [0.1]

    requested_runs = len(cases) * len(strategies) * len(warm_cs)
    quick_budget = len(QUICK_CASES) * len(QUICK_STRATEGIES)
    if quick and requested_runs > quick_budget:
        raise SystemExit(
            f"Refusing to run {requested_runs} ablations without --full. "
            f"Use --full for an intentional long grid, or keep the quick default."
        )

    case_paths = [_case_path(case) for case in cases]
    missing = [str(path) for path in case_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing case file(s): {', '.join(missing)}")

    commands: list[tuple[Path, str, float, list[str]]] = []
    for case in case_paths:
        for strategy in strategies:
            for warm_c in warm_cs:
                commands.append(
                    (
                        case,
                        strategy,
                        float(warm_c),
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "ablate_main1.py"),
                            "--case",
                            str(case),
                            "--strategy",
                            strategy,
                            "--seed",
                            str(int(args.seed)),
                            "--warm-c",
                            str(float(warm_c)),
                            "--q-target",
                            str(int(args.q_target)),
                            "--p-layers",
                            str(int(args.p_layers)),
                            "--selector-dist-thr",
                            str(float(args.selector_dist_thr)),
                            "--selector-lambda-cap",
                            str(int(args.selector_lambda_cap)),
                            "--selector-prefilter",
                            str(int(args.selector_prefilter)),
                        ],
                    )
                )

    if args.dry_run:
        for _, _, _, cmd in commands:
            print(" ".join(cmd))
        return

    rows: list[dict[str, Any]] = []
    for case, strategy, warm_c, cmd in commands:
        wall_t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
        wall_elapsed = time.time() - wall_t0
        parsed = _parse_ablate_stdout(proc.stdout)
        row = {
            "case": parsed.get("case", case.name),
            "case_suffix": _case_suffix(case),
            "strategy": parsed.get("strategy", strategy),
            "seed": int(args.seed),
            "warm_c": parsed.get("warm_c", warm_c),
            "selector_dist_thr": float(args.selector_dist_thr),
            "selector_lambda_cap": int(args.selector_lambda_cap),
            "selector_prefilter": int(args.selector_prefilter),
            "q": parsed.get("q", args.q_target),
            "p": parsed.get("p", args.p_layers),
            "rows": parsed.get("rows"),
            "hv": parsed.get("hv"),
            "base": parsed.get("base"),
            "gain": parsed.get("gain"),
            "score": parsed.get("score"),
            "elapsed": parsed.get("elapsed"),
            "returncode": proc.returncode,
            "wall_elapsed": wall_elapsed,
            "command": " ".join(cmd),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        rows.append(row)
        print(
            f"{row['case']},{row['strategy']},warm_c={row['warm_c']},"
            f"score={row['score']},elapsed={row['elapsed']},returncode={proc.returncode}",
            flush=True,
        )

    csv_path, json_path = _write_outputs(rows, args.out or _default_out(quick))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
