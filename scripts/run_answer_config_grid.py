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
DEFAULT_BUDGETS = [(450, 100, 275, 200)]
DEFAULT_MIXES = [((2031, 11), (2041, 9))]

CSV_FIELDS = [
    "case",
    "case_suffix",
    "digest",
    "sampler",
    "budget_id",
    "budget_config",
    "budget_total",
    "broad_num_weights",
    "broad_shots",
    "warm_num_weights",
    "warm_shots",
    "mix_id",
    "seeds",
    "seed_count",
    "first_seed",
    "rng_seed",
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
    return f"results/answer_config_grid_{stamp}"


def _parse_budget(raw: str) -> tuple[int, int, int, int] | None:
    text = str(raw).strip().lower()
    if text in {"none", "default"}:
        return None

    normalized = text
    if "x" in normalized or "+" in normalized:
        normalized = normalized.replace("+", ":").replace("x", ":")
    parts = [part.strip() for part in normalized.split(":") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Budgets must be four integers, e.g. 450:100:275:200 or 450x100+275x200."
        )

    try:
        broad_num, broad_shots, warm_num, warm_shots = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid budget: {raw}") from exc
    budget = (broad_num, broad_shots, warm_num, warm_shots)
    if any(value <= 0 for value in budget):
        raise argparse.ArgumentTypeError("Budget values must be positive.")

    total = broad_num * broad_shots + warm_num * warm_shots
    if total != BASE_SAMPLE_BUDGET:
        raise argparse.ArgumentTypeError(
            f"Budget {raw} uses {total} samples; expected {BASE_SAMPLE_BUDGET}."
        )
    return budget


def _budget_id(budget: tuple[int, int, int, int] | None) -> str:
    if budget is None:
        return "none"
    return f"{int(budget[0])}x{int(budget[1])}+{int(budget[2])}x{int(budget[3])}"


def _budget_config_text(budget: tuple[int, int, int, int] | None) -> str:
    if budget is None:
        return "none"
    return ":".join(str(int(value)) for value in budget)


def _budget_total(budget: tuple[int, int, int, int] | None) -> str | int:
    if budget is None:
        return ""
    return int(budget[0]) * int(budget[1]) + int(budget[2]) * int(budget[3])


def _parse_mix(raw: str) -> tuple[Any, ...] | None:
    text = str(raw).strip()
    if text.lower() in {"none", "default"}:
        return None

    parts = [part.strip() for part in text.replace(",", "+").split("+") if part.strip()]
    if len(parts) < 1:
        raise argparse.ArgumentTypeError("Each mix must contain at least one seed, or 'none'.")
    try:
        if any(":" in part for part in parts):
            items: list[tuple[int, int]] = []
            for part in parts:
                seed_text, weight_text = part.split(":", 1)
                items.append((int(seed_text), int(weight_text)))
            if any(weight <= 0 for _, weight in items):
                raise ValueError("weights must be positive")
            seeds: tuple[Any, ...] = tuple(items)
        else:
            seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed mix: {raw}") from exc
    seed_ids = tuple(int(item[0]) if isinstance(item, tuple) else int(item) for item in seeds)
    if len(set(seed_ids)) != len(seed_ids):
        raise argparse.ArgumentTypeError("Seed mixes must not contain duplicates.")
    return seeds


def _mix_id(seeds: tuple[Any, ...] | None) -> str:
    if seeds is None:
        return "none"
    parts: list[str] = []
    for seed in seeds:
        if isinstance(seed, (tuple, list)):
            parts.append(f"{int(seed[0])}x{int(seed[1])}")
        else:
            parts.append(str(int(seed)))
    return "+".join(parts)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    case = row.get("case_suffix") or row.get("case")
    budget = row.get("budget_id")
    mix = row.get("mix_id")
    if case in (None, "") or budget in (None, "") or mix in (None, ""):
        return None
    return str(case), str(budget), str(mix)


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


def _first_seed(mix: tuple[Any, ...] | None) -> int | None:
    if mix is None:
        return None
    first = mix[0]
    return int(first[0]) if isinstance(first, (tuple, list)) else int(first)


def _seed_count(mix: tuple[Any, ...] | None) -> int:
    return 0 if mix is None else int(len(mix))


def _patch_digest_config(config: dict[str, Any], digest: str, value: Any) -> tuple[bool, Any]:
    old_present = digest in config
    old_value = config.get(digest)
    if value is None:
        config.pop(digest, None)
    else:
        config[digest] = tuple(value)
    return old_present, old_value


def _restore_digest_config(config: dict[str, Any], digest: str, old_present: bool, old_value: Any) -> None:
    if old_present:
        config[digest] = old_value
    else:
        config.pop(digest, None)


def _evaluate(
    case: Path,
    budget: tuple[int, int, int, int] | None,
    mix: tuple[Any, ...] | None,
) -> dict[str, Any]:
    import answer  # noqa: WPS433
    from run import _hv_from_spins, baseline_hv  # noqa: WPS433
    from utils import problem_from_npz  # noqa: WPS433

    t0 = time.time()
    budget_id = _budget_id(budget)
    mix_id = _mix_id(mix)
    first_seed = _first_seed(mix)
    problem = problem_from_npz(str(case))
    digest = str(answer._problem_digest(problem))
    row: dict[str, Any] = {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "digest": digest,
        "sampler": "answer.main1",
        "budget_id": budget_id,
        "budget_config": _budget_config_text(budget),
        "budget_total": _budget_total(budget),
        "broad_num_weights": "" if budget is None else int(budget[0]),
        "broad_shots": "" if budget is None else int(budget[1]),
        "warm_num_weights": "" if budget is None else int(budget[2]),
        "warm_shots": "" if budget is None else int(budget[3]),
        "mix_id": mix_id,
        "seeds": mix_id,
        "seed_count": _seed_count(mix),
        "first_seed": "" if first_seed is None else int(first_seed),
        "rng_seed": "" if first_seed is None else int(first_seed),
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

    old_budget_present = False
    old_budget: Any = None
    old_mix_present = False
    old_mix: Any = None
    budget_patched = False
    mix_patched = False
    try:
        base = float(baseline_hv(case, problem))
        old_budget_present, old_budget = _patch_digest_config(
            answer._MAIN1_BUDGET_CONFIG,
            digest,
            budget,
        )
        budget_patched = True
        old_mix_present, old_mix = _patch_digest_config(
            answer._MAIN1_SEED_MIX_CONFIG,
            digest,
            mix,
        )
        mix_patched = True

        if first_seed is None:
            result = answer.main1(problem, sample_budget=BASE_SAMPLE_BUDGET)
        else:
            result = answer.main1(
                problem,
                sample_budget=BASE_SAMPLE_BUDGET,
                rng_seed=int(first_seed),
            )
        if not isinstance(result, dict):
            raise TypeError("answer.main1() must return a dict.")
        if "sample_spins" not in result:
            raise KeyError("answer.main1() result must contain 'sample_spins'.")

        spins = np.asarray(result["sample_spins"], dtype=np.int8)
        rows = int(spins.shape[0])
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
        if "answer" in sys.modules and budget_patched:
            _restore_digest_config(answer._MAIN1_BUDGET_CONFIG, digest, old_budget_present, old_budget)
        if "answer" in sys.modules and mix_patched:
            _restore_digest_config(answer._MAIN1_SEED_MIX_CONFIG, digest, old_mix_present, old_mix)
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
            "Evaluate answer.main1 config combinations by temporarily monkeypatching "
            "answer._MAIN1_BUDGET_CONFIG[digest] and answer._MAIN1_SEED_MIX_CONFIG[digest]. "
            "Default mode is dry-run."
        )
    )
    parser.add_argument("--case", default=DEFAULT_CASE, help="Public case suffix such as 09, or an explicit .npz path.")
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=_parse_budget,
        default=DEFAULT_BUDGETS,
        help=(
            "Budget configs such as 450:100:275:200 or 450x100+275x200. "
            "Use none/default to remove the per-case budget override."
        ),
    )
    parser.add_argument(
        "--mixes",
        nargs="+",
        type=_parse_mix,
        default=DEFAULT_MIXES,
        help=(
            "Seed mixes such as none, 2031+2041, or weighted 2031:11+2041:9. "
            "Use none/default to remove the per-case seed-mix override."
        ),
    )
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--rerun", action="store_true", help="Rerun successful existing case/budget/mix rows.")
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
    planned = [(case, budget, mix) for budget in args.budgets for mix in args.mixes]
    runnable = [
        (item_case, budget, mix)
        for item_case, budget, mix in planned
        if args.rerun or (case_suffix, _budget_id(budget), _mix_id(mix)) not in done
    ]

    if args.dry_run:
        print("mode=dry-run")
        print(f"case={case}")
        print(f"case_suffix={case_suffix}")
        print(f"digest={digest}")
        print(f"out_csv={csv_path}")
        print(f"out_json={json_path}")
        print(f"existing_rows={len(existing_rows)}")
        print(f"successful_rows={len(done)}")
        print(f"planned_runs={len(planned)}")
        print(f"runs_to_execute={len(runnable)}")
        for _, budget, mix in runnable:
            first_seed = _first_seed(mix)
            rng_seed = "" if first_seed is None else int(first_seed)
            print(
                f"budget={_budget_id(budget)},mix={_mix_id(mix)},"
                f"rng_seed={rng_seed},monkeypatch_digest={digest}"
            )
        return

    rows = existing_rows
    for item_case, budget, mix in planned:
        key = (case_suffix, _budget_id(budget), _mix_id(mix))
        if not args.rerun and key in done:
            print(
                f"{item_case.name},budget={_budget_id(budget)},mix={_mix_id(mix)},status=skipped",
                flush=True,
            )
            continue

        row = _evaluate(item_case, budget, mix)
        _replace_or_append(rows, row)
        _write_outputs(rows, csv_path, json_path)
        print(
            f"{row['case']},budget={row['budget_id']},mix={row['mix_id']},status={row['status']},"
            f"hv={row['hv']},base={row['base']},gain={row['gain']},score={row['score']},"
            f"rows={row['rows']},sample_used={row['sample_used']},elapsed={row['elapsed']:.3f},"
            f"returncode={row['returncode']}",
            flush=True,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
