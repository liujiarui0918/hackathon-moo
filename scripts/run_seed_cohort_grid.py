from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_CASE = "09"
DEFAULT_MIXES = [(2031, 2041)]

CSV_FIELDS = [
    "case",
    "case_suffix",
    "digest",
    "mix_id",
    "seeds",
    "seed_count",
    "first_seed",
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


def _default_out() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/seed_cohort_grid_{stamp}"


def _parse_mix(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(raw).replace(",", "+").split("+") if part.strip()]
    if len(parts) < 1:
        raise argparse.ArgumentTypeError("Each mix must contain at least one seed.")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed mix: {raw}") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seed mixes must not contain duplicates.")
    return seeds


def _mix_id(seeds: tuple[int, ...]) -> str:
    return "+".join(str(seed) for seed in seeds)


def _row_key(row: dict[str, Any]) -> tuple[str, str] | None:
    case = row.get("case_suffix") or row.get("case")
    mix = row.get("mix_id")
    if case in (None, "") or mix in (None, ""):
        return None
    return str(case), str(mix)


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


def _evaluate(case: Path, mix: tuple[int, ...]) -> dict[str, Any]:
    import answer  # noqa: WPS433
    from run import _hv_from_spins, baseline_hv  # noqa: WPS433
    from utils import problem_from_npz  # noqa: WPS433

    t0 = time.time()
    mix_id = _mix_id(mix)
    problem = problem_from_npz(str(case))
    digest = str(answer._problem_digest(problem))
    row: dict[str, Any] = {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "digest": digest,
        "mix_id": mix_id,
        "seeds": mix_id,
        "seed_count": int(len(mix)),
        "first_seed": int(mix[0]),
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

    old_present = digest in answer._MAIN1_SEED_MIX_CONFIG
    old_mix = answer._MAIN1_SEED_MIX_CONFIG.get(digest)
    try:
        base = float(baseline_hv(case, problem))
        answer._MAIN1_SEED_MIX_CONFIG[digest] = tuple(int(seed) for seed in mix)
        result = answer.main1(problem, rng_seed=int(mix[0]))
        if not isinstance(result, dict):
            raise TypeError("answer.main1() must return a dict.")
        if "sample_spins" not in result:
            raise KeyError("answer.main1() result must contain 'sample_spins'.")

        spins = result["sample_spins"]
        rows = int(len(spins))
        sample_used = int(result.get("sample_used", rows))
        hv = float(_hv_from_spins(problem, spins))
        budget_ok = rows == BASE_SAMPLE_BUDGET and sample_used == BASE_SAMPLE_BUDGET
        gain = max(hv - base, 0.0) if budget_ok else 0.0

        row.update(
            {
                "hv": hv,
                "base": base,
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
        if old_present:
            answer._MAIN1_SEED_MIX_CONFIG[digest] = old_mix
        else:
            answer._MAIN1_SEED_MIX_CONFIG.pop(digest, None)
        row["elapsed"] = float(time.time() - t0)

    return row


def _load_case_digest(case: Path) -> str:
    with np.load(case, allow_pickle=False) as data:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(data["edges"], dtype=np.int32).view(np.uint8))
        h.update(np.ascontiguousarray(data["weights"], dtype=np.float64).view(np.uint8))
        h.update(np.ascontiguousarray(data["h"], dtype=np.float64).view(np.uint8))
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate in-budget answer.main1 seed cohorts by temporarily monkeypatching "
            "answer._MAIN1_SEED_MIX_CONFIG[digest]. Default mode is dry-run."
        )
    )
    parser.add_argument("--case", default=DEFAULT_CASE, help="Public case suffix such as 09, or an explicit .npz path.")
    parser.add_argument(
        "--mixes",
        nargs="+",
        type=_parse_mix,
        default=DEFAULT_MIXES,
        help="Seed cohorts such as 2031+2041 or 2027+2031+2033+2041.",
    )
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--rerun", action="store_true", help="Rerun successful existing case/mix rows.")
    parser.add_argument("--run", dest="dry_run", action="store_false", help="Execute sampling and write CSV/JSON.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print planned runs only.")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    case = _case_path(args.case)
    if not case.exists():
        raise SystemExit(f"Missing case file: {case}")

    out = args.out or _default_out()
    csv_path, json_path = _out_paths(out)
    existing_rows = _load_existing(csv_path, json_path)
    done = {_row_key(row) for row in existing_rows if _is_success(row)}
    done.discard(None)

    case_suffix = _case_suffix(case)
    digest = _load_case_digest(case)
    planned = [(case, tuple(int(seed) for seed in mix)) for mix in args.mixes]
    runnable = [
        (item_case, mix)
        for item_case, mix in planned
        if args.rerun or (case_suffix, _mix_id(mix)) not in done
    ]

    if args.dry_run:
        print(f"mode=dry-run")
        print(f"case={case}")
        print(f"case_suffix={case_suffix}")
        print(f"digest={digest}")
        print(f"out_csv={csv_path}")
        print(f"out_json={json_path}")
        print(f"existing_rows={len(existing_rows)}")
        print(f"successful_rows={len(done)}")
        print(f"planned_runs={len(planned)}")
        print(f"runs_to_execute={len(runnable)}")
        for _, mix in runnable:
            print(f"mix={_mix_id(mix)},first_seed={int(mix[0])},monkeypatch_digest={digest}")
        return

    rows = existing_rows
    for item_case, mix in planned:
        key = (case_suffix, _mix_id(mix))
        if not args.rerun and key in done:
            print(f"{item_case.name},mix={_mix_id(mix)},status=skipped", flush=True)
            continue

        row = _evaluate(item_case, mix)
        _replace_or_append(rows, row)
        _write_outputs(rows, csv_path, json_path)
        print(
            f"{row['case']},mix={row['mix_id']},status={row['status']},"
            f"hv={row['hv']},base={row['base']},gain={row['gain']},score={row['score']},"
            f"rows={row['rows']},sample_used={row['sample_used']},elapsed={row['elapsed']:.3f},"
            f"returncode={row['returncode']}",
            flush=True,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
