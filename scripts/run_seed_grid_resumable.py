from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SAMPLE_BUDGET = 100000
QUICK_CASES = ["08"]
QUICK_SEEDS = [2026]
FULL_CASES = [f"{i:02d}" for i in range(10)]
FULL_SEEDS = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2035, 2041]

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
    "returncode",
    "status",
    "error",
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


def _default_out(quick: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if quick else "full"
    return f"results/seed_grid_resumable_{mode}_{stamp}"


def _row_key(row: dict[str, Any]) -> tuple[str, int] | None:
    case = row.get("case_suffix") or row.get("case")
    seed = row.get("seed")
    if case is None or seed in (None, ""):
        return None
    try:
        return str(case), int(seed)
    except (TypeError, ValueError):
        return None


def _is_success(row: dict[str, Any]) -> bool:
    try:
        return int(row.get("returncode", 1)) == 0 and str(row.get("status", "")) == "ok"
    except (TypeError, ValueError):
        return False


def _load_existing(csv_path: Path, json_path: Path) -> list[dict[str, Any]]:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Existing JSON must contain a list of rows: {json_path}")
        return [dict(row) for row in data]

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    return []


def _write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_name(f"{csv_path.name}.tmp")
    json_tmp = json_path.with_name(f"{json_path.name}.tmp")

    with csv_tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(csv_path)

    with json_tmp.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    json_tmp.replace(json_path)


def _replace_or_append(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = _row_key(row)
    if key is None:
        rows.append(row)
        return
    for idx, old_row in enumerate(rows):
        if _row_key(old_row) == key:
            rows[idx] = row
            return
    rows.append(row)


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _hv_from_spins_safe(problem: Any, spins: np.ndarray) -> float:
    from utils import (  # noqa: WPS433
        hypervolume_pygmo,
        merge_non_dominated_pool,
        normalize_energies,
        objective_extrema,
        pg_non_dominated_indices,
    )

    arr = np.asarray(spins, dtype=np.int8)
    if arr.size == 0:
        return 0.0
    lower, upper = objective_extrema(problem)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)
    for start in range(0, int(arr.shape[0]), 4096):
        block = arr[start : start + 4096]
        objs = normalize_energies(
            _energy_batch_safe(block, problem.edges, problem.weights, problem.h),
            lower,
            upper,
        )
        nd_pool = merge_non_dominated_pool(nd_pool, objs[pg_non_dominated_indices(objs)])
    if nd_pool.size == 0:
        return 0.0
    if int(nd_pool.shape[0]) == 1:
        ref = np.full((int(problem.k),), 1.01, dtype=np.float64)
        if not np.all(nd_pool[0] <= ref):
            return 0.0
        return float(np.prod(ref - nd_pool[0]))
    return float(hypervolume_pygmo(nd_pool))


def _evaluate(case: Path, seed: int) -> dict[str, Any]:
    t0 = time.time()
    row: dict[str, Any] = {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "seed": int(seed),
        "hv": "",
        "base": "",
        "gain": "",
        "score": "",
        "rows": "",
        "sample_used": "",
        "elapsed": "",
        "returncode": 1,
        "status": "error",
        "error": "",
    }

    try:
        import answer  # noqa: WPS433
        from run import baseline_hv  # noqa: WPS433
        from utils import problem_from_npz  # noqa: WPS433

        problem = problem_from_npz(str(case))
        base = baseline_hv(case, problem)
        result = answer.main1(problem, rng_seed=int(seed))
        if not isinstance(result, dict):
            raise TypeError("answer.main1() must return a dict.")
        if "sample_spins" not in result:
            raise KeyError("answer.main1() result must contain 'sample_spins'.")

        spins = result["sample_spins"]
        rows = int(len(spins))
        sample_used = int(result.get("sample_used", rows))
        hv = _hv_from_spins_safe(problem, spins)
        budget_ok = rows == BASE_SAMPLE_BUDGET and sample_used == BASE_SAMPLE_BUDGET
        gain = max(float(hv) - float(base), 0.0) if budget_ok else 0.0

        row.update(
            {
                "hv": float(hv),
                "base": float(base),
                "gain": float(gain),
                "score": float(gain * 100000.0),
                "rows": rows,
                "sample_used": sample_used,
                "returncode": 0 if budget_ok else 2,
                "status": "ok" if budget_ok else "invalid_budget",
                "error": "" if budget_ok else f"expected {BASE_SAMPLE_BUDGET} rows/sample_used",
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "returncode": 1,
                "status": "error",
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
        )
    finally:
        row["elapsed"] = float(time.time() - t0)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable answer.main1 case/seed grid, flushing CSV/JSON after every run."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the small safe default grid: case 08, seed 2026.")
    mode.add_argument("--full", action="store_true", help="Allow multi-case or multi-seed grids.")
    parser.add_argument("--cases", nargs="*", help="Case suffixes such as 00 01 08, or explicit .npz paths.")
    parser.add_argument("--seeds", nargs="*", type=int, help="rng_seed values to test.")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--rerun", action="store_true", help="Rerun successful existing case/seed rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and resume decisions, then exit.")
    args = parser.parse_args()

    quick = args.quick or not args.full
    cases = args.cases if args.cases else (QUICK_CASES if quick else FULL_CASES)
    seeds = args.seeds if args.seeds else (QUICK_SEEDS if quick else FULL_SEEDS)

    if quick and (len(cases) > 1 or len(seeds) > 1):
        raise SystemExit(
            "Refusing to run a multi-case or multi-seed grid without --full. "
            "Use --full for intentional long seed sweeps."
        )

    case_paths = [_case_path(case) for case in cases]
    missing = [str(path) for path in case_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing case file(s): {', '.join(missing)}")

    out = args.out or _default_out(quick)
    csv_path, json_path = _out_paths(out)
    existing_rows = _load_existing(csv_path, json_path)
    done = {_row_key(row) for row in existing_rows if _is_success(row)}
    done.discard(None)

    planned = [(case, int(seed)) for case in case_paths for seed in seeds]
    runnable = [
        (case, seed)
        for case, seed in planned
        if args.rerun or (_case_suffix(case), int(seed)) not in done
    ]

    if args.dry_run:
        print(f"out_csv={csv_path}")
        print(f"out_json={json_path}")
        print(f"existing_rows={len(existing_rows)}")
        print(f"successful_rows={len(done)}")
        print(f"planned_runs={len(planned)}")
        print(f"runs_to_execute={len(runnable)}")
        for case, seed in runnable:
            print(f"{case},seed={int(seed)}")
        return

    rows = existing_rows
    for case, seed in planned:
        key = (_case_suffix(case), int(seed))
        if not args.rerun and key in done:
            print(f"{case.name},seed={seed},status=skipped", flush=True)
            continue

        row = _evaluate(case, seed)
        _replace_or_append(rows, row)
        _write_outputs(rows, csv_path, json_path)

        print(
            f"{row['case']},seed={row['seed']},status={row['status']},"
            f"hv={row['hv']},base={row['base']},gain={row['gain']},score={row['score']},"
            f"rows={row['rows']},sample_used={row['sample_used']},elapsed={row['elapsed']:.3f},"
            f"returncode={row['returncode']}",
            flush=True,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
