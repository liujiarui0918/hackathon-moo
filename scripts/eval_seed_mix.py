from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUICK_CASES = ["08"]
QUICK_SEEDS = [2026]
FULL_CASES = ["01", "05", "07", "08", "09"]
CSV_FIELDS = [
    "case",
    "case_suffix",
    "seed",
    "hv",
    "base",
    "gain",
    "score",
    "rows",
    "sample_used",
    "elapsed",
]


def _case_path(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


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
    return f"results/seed_mix_{mode}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer.main1 over a guarded case/seed grid.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the small safe default grid.")
    mode.add_argument("--full", action="store_true", help="Allow the requested full grid to run.")
    parser.add_argument("--cases", nargs="*", help="Case suffixes such as 01 05 08, or explicit .npz paths.")
    parser.add_argument("--seeds", nargs="*", type=int, help="rng_seed values to test.")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and write no outputs.")
    args = parser.parse_args()

    quick = args.quick or not args.full
    cases = args.cases if args.cases else (QUICK_CASES if quick else FULL_CASES)
    seeds = args.seeds if args.seeds else QUICK_SEEDS

    requested_runs = len(cases) * len(seeds)
    quick_budget = len(QUICK_CASES) * len(QUICK_SEEDS)
    if quick and requested_runs > quick_budget:
        raise SystemExit(
            f"Refusing to run {requested_runs} seed evaluations without --full. "
            f"Use --full for an intentional long grid, or keep the quick default."
        )

    case_paths = [_case_path(case) for case in cases]
    missing = [str(path) for path in case_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing case file(s): {', '.join(missing)}")

    if args.dry_run:
        for case in case_paths:
            for seed in seeds:
                print(f"{case},seed={int(seed)}")
        return

    import answer  # noqa: WPS433
    from run import _hv_from_spins, baseline_hv  # noqa: WPS433
    from utils import problem_from_npz  # noqa: WPS433

    rows: list[dict[str, Any]] = []
    for case in case_paths:
        problem = problem_from_npz(str(case))
        base = baseline_hv(case, problem)
        for seed in seeds:
            t0 = time.time()
            result = answer.main1(problem, rng_seed=int(seed))
            elapsed = time.time() - t0
            spins = result["sample_spins"]
            hv = _hv_from_spins(problem, spins)
            gain = max(float(hv) - float(base), 0.0)
            row = {
                "case": case.name,
                "case_suffix": _case_suffix(case),
                "seed": int(seed),
                "hv": float(hv),
                "base": float(base),
                "gain": float(gain),
                "score": float(gain * 100000.0),
                "rows": int(len(spins)),
                "sample_used": int(result.get("sample_used", len(spins))),
                "elapsed": float(elapsed),
            }
            rows.append(row)
            print(
                f"{row['case']},seed={row['seed']},hv={row['hv']:.12f},base={row['base']:.12f},"
                f"gain={row['gain']:.12f},score={row['score']:.6f},"
                f"rows={row['rows']},sample_used={row['sample_used']},elapsed={row['elapsed']:.3f}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, args.out or _default_out(quick))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
