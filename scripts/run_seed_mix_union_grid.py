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
HEADROOM_CASES = ["04", "00", "09", "07"]
EXTRA_CASES = ["06", "02", "08"]
QUICK_CASES = ["09"]
QUICK_MIXES = [(2031, 2041)]
DEFAULT_MIXES = [
    (2031, 2041),
    (2031, 2033),
    (2027, 2031),
    (2027, 2031, 2033, 2041),
]
VERIFIED_CANDIDATES = [
    ROOT / "results" / "seed_schedule_with01_public_default.json",
    ROOT / "results" / "seed_schedule_public_default.json",
    ROOT / "results" / "latest_score.json",
]

CSV_FIELDS = [
    "case",
    "case_suffix",
    "sampler",
    "mix_id",
    "seeds",
    "seed_count",
    "take_mode",
    "quota",
    "hv",
    "base",
    "gain",
    "score",
    "verified_score",
    "verified_source",
    "beats_verified",
    "single_best_seed",
    "single_best_hv",
    "single_best_score",
    "beats_single_best",
    "rows",
    "sample_used",
    "elapsed",
    "seed_elapsed",
    "eval_elapsed",
    "returncode",
    "status",
    "error",
    "member_scores_json",
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
    return f"results/seed_mix_union_grid_{mode}_{stamp}"


def _parse_mix(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.replace(",", "+").split("+") if part.strip()]
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError("Each mix must contain exactly 2 or 4 seeds.")
    seeds = tuple(int(part) for part in parts)
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seed mixes must not contain duplicates.")
    return seeds


def _mix_id(seeds: tuple[int, ...]) -> str:
    return "+".join(str(seed) for seed in seeds)


def _quota(seed_count: int, budget: int = BASE_SAMPLE_BUDGET) -> list[int]:
    base = int(budget) // int(seed_count)
    remainder = int(budget) % int(seed_count)
    return [base + (1 if idx < remainder else 0) for idx in range(seed_count)]


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    case = row.get("case_suffix") or row.get("case")
    sampler = row.get("sampler", "answer-main1")
    mix = row.get("mix_id")
    take_mode = row.get("take_mode", "stride")
    if case in (None, "") or mix in (None, ""):
        return None
    return str(case), str(sampler), str(mix), str(take_mode)


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


def _case_id_from_name(name: str) -> str | None:
    stem = Path(str(name)).stem
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[-1]
    return suffix if suffix.isdigit() else None


def _extract_verified_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("k5_rows", "rows", "small_rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _load_verified_scores(paths: list[Path]) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        for row in _extract_verified_rows(payload):
            case = row.get("case_suffix") or _case_id_from_name(str(row.get("case", "")))
            if case is None:
                continue
            value = row.get("score_case", row.get("score"))
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if str(case) not in scores:
                scores[str(case)] = score
                sources[str(case)] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return scores, sources


def _load_known_single_scores() -> dict[str, dict[int, dict[str, float]]]:
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for path in sorted((ROOT / "results").glob("*.json")):
        name = path.name
        if "seed_grid" not in name and "seed_mix_" not in name and "answer_schedule" not in name:
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        for row in _extract_verified_rows(payload):
            case = row.get("case_suffix") or _case_id_from_name(str(row.get("case", "")))
            seed = row.get("seed")
            if case is None or seed in (None, ""):
                continue
            try:
                seed_int = int(seed)
                score = float(row.get("score_case", row.get("score")))
            except (TypeError, ValueError):
                continue
            item = {"score": score}
            try:
                item["hv"] = float(row.get("hv", row.get("hv_solver")))
            except (TypeError, ValueError):
                pass
            scores.setdefault(str(case), {})
            old = scores[str(case)].get(seed_int)
            if old is None or score > float(old.get("score", float("-inf"))):
                scores[str(case)][seed_int] = item
    return scores


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


def _stable_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _take_rows(spins: np.ndarray, count: int, mode: str, salt: str) -> np.ndarray:
    arr = np.asarray(spins, dtype=np.int8)
    if int(count) <= 0:
        return arr[:0].copy()
    if int(arr.shape[0]) < int(count):
        raise ValueError(f"Cannot take {count} rows from only {arr.shape[0]} rows.")
    if int(arr.shape[0]) == int(count):
        return arr.copy()
    if mode == "prefix":
        return arr[:count].copy()
    if mode == "stride":
        idx = np.linspace(0, int(arr.shape[0]) - 1, num=int(count), dtype=np.int64)
        return arr[idx].copy()
    if mode == "random":
        rng = np.random.default_rng(_stable_u64(salt))
        idx = rng.choice(int(arr.shape[0]), size=int(count), replace=False)
        idx.sort()
        return arr[idx].copy()
    raise ValueError(f"Unknown take mode: {mode}")


def _baseline_qaoa_mix(problem: Any, seeds: tuple[int, ...]) -> np.ndarray:
    import baseline  # noqa: WPS433
    from mindquantum.simulator import Simulator  # noqa: WPS433
    from utils import (  # noqa: WPS433
        build_qaoa_circuit_from_projected_ising,
        load_weight_pool,
        sampling_result_to_unique_spins,
    )

    shots_per_weight = int(baseline.SHOTS_PER_WEIGHT)
    weights_per_seed = [int(q) // shots_per_weight for q in _quota(len(seeds))]
    if sum(weights_per_seed) != int(baseline.NUM_WEIGHTS):
        raise ValueError("baseline-qaoa sampler requires seed quotas to align with baseline weights.")

    betas, gammas = baseline._TRANSFER_TABLE[baseline.P_LAYERS]  # noqa: SLF001
    lambdas = load_weight_pool(int(problem.k), n=1000, seed=2026)[: int(baseline.NUM_WEIGHTS)].astype(np.float64)
    projected_j_pool = np.einsum("lk,km->lm", lambdas, problem.weights, optimize=False).astype(
        np.float64,
        copy=False,
    )
    projected_h_pool = np.einsum("lk,kn->ln", lambdas, problem.h, optimize=False).astype(
        np.float64,
        copy=False,
    )

    sim = Simulator("mqvector", int(problem.n), seed=int(seeds[0]))
    blocks: list[np.ndarray] = []
    weight_cursor = 0
    for seed, weight_count in zip(seeds, weights_per_seed):
        for local_idx in range(int(weight_count)):
            weight_idx = weight_cursor + local_idx
            circ = build_qaoa_circuit_from_projected_ising(
                problem,
                projected_j_pool[weight_idx],
                projected_h_pool[weight_idx],
                betas=betas,
                gammas=gammas,
                warm_bits01=None,
            )
            sim.reset()
            res = sim.sampling(circ, shots=shots_per_weight, seed=int(seed) + weight_idx)
            unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
            block = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
            if int(block.shape[0]) != shots_per_weight:
                raise ValueError(f"Sampling row count mismatch for seed={seed}, weight={weight_idx}.")
            blocks.append(np.asarray(block, dtype=np.int8))
        weight_cursor += int(weight_count)

    sample_spins = np.concatenate(blocks, axis=0)
    if int(sample_spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise ValueError(f"baseline-qaoa rows={sample_spins.shape[0]}, expected {BASE_SAMPLE_BUDGET}.")
    return sample_spins


def _random_spins_mix(problem: Any, seeds: tuple[int, ...]) -> np.ndarray:
    parts: list[np.ndarray] = []
    n = int(problem.n)
    for seed, count in zip(seeds, _quota(len(seeds))):
        rng = np.random.default_rng(int(seed))
        part = np.where(rng.random((int(count), n)) < 0.5, -1, 1).astype(np.int8)
        parts.append(part)
    sample_spins = np.concatenate(parts, axis=0)
    if int(sample_spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise ValueError(f"random-spins rows={sample_spins.shape[0]}, expected {BASE_SAMPLE_BUDGET}.")
    return sample_spins


def _evaluate_mix(
    case: Path,
    seeds: tuple[int, ...],
    sampler: str,
    take_mode: str,
    verified_scores: dict[str, float],
    verified_sources: dict[str, str],
    known_single_scores: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    t0 = time.time()
    case_suffix = _case_suffix(case)
    mix = _mix_id(seeds)
    row: dict[str, Any] = {
        "case": case.name,
        "case_suffix": case_suffix,
        "sampler": sampler,
        "mix_id": mix,
        "seeds": mix,
        "seed_count": int(len(seeds)),
        "take_mode": take_mode,
        "quota": json.dumps(_quota(len(seeds)), separators=(",", ":")),
        "hv": "",
        "base": "",
        "gain": "",
        "score": "",
        "verified_score": verified_scores.get(case_suffix, ""),
        "verified_source": verified_sources.get(case_suffix, ""),
        "beats_verified": "",
        "single_best_seed": "",
        "single_best_hv": "",
        "single_best_score": "",
        "beats_single_best": "",
        "rows": "",
        "sample_used": "",
        "elapsed": "",
        "seed_elapsed": "",
        "eval_elapsed": "",
        "returncode": 1,
        "status": "error",
        "error": "",
        "member_scores_json": "",
    }

    try:
        from run import baseline_hv  # noqa: WPS433
        from utils import problem_from_npz  # noqa: WPS433

        problem = problem_from_npz(str(case))
        base = float(baseline_hv(case, problem))
        quotas = _quota(len(seeds))
        member_scores: list[dict[str, Any]] = []

        seed_t0 = time.time()
        if sampler == "answer-main1":
            import answer  # noqa: WPS433

            parts: list[np.ndarray] = []
            for seed, take_count in zip(seeds, quotas):
                print(f"{case.name},mix={mix},sampler={sampler},seed={seed},status=sampling", flush=True)
                result = answer.main1(problem, rng_seed=int(seed))
                if not isinstance(result, dict):
                    raise TypeError("answer.main1() must return a dict.")
                if "sample_spins" not in result:
                    raise KeyError("answer.main1() result must contain 'sample_spins'.")
                spins = np.asarray(result["sample_spins"], dtype=np.int8)
                rows = int(spins.shape[0])
                sample_used = int(result.get("sample_used", rows))
                if rows != BASE_SAMPLE_BUDGET or sample_used != BASE_SAMPLE_BUDGET:
                    raise ValueError(
                        f"seed {seed} returned rows={rows}, sample_used={sample_used}; "
                        f"expected {BASE_SAMPLE_BUDGET}."
                    )

                single_hv = _hv_from_spins_safe(problem, spins)
                single_gain = max(float(single_hv) - base, 0.0)
                member_scores.append(
                    {
                        "seed": int(seed),
                        "hv": float(single_hv),
                        "gain": float(single_gain),
                        "score": float(single_gain * 100000.0),
                        "rows": rows,
                    }
                )
                parts.append(_take_rows(spins, take_count, take_mode, f"{case_suffix}:{mix}:{seed}:{take_mode}"))
            union_spins = np.concatenate(parts, axis=0)
        elif sampler == "baseline-qaoa":
            print(f"{case.name},mix={mix},sampler={sampler},status=sampling", flush=True)
            union_spins = _baseline_qaoa_mix(problem, seeds)
            for seed in seeds:
                known = known_single_scores.get(case_suffix, {}).get(int(seed), {})
                item: dict[str, Any] = {
                    "seed": int(seed),
                    "score": known.get("score", ""),
                    "rows": BASE_SAMPLE_BUDGET,
                    "source": "known_results",
                }
                if "hv" in known:
                    item["hv"] = known["hv"]
                member_scores.append(item)
        elif sampler == "random-spins":
            print(f"{case.name},mix={mix},sampler={sampler},status=sampling", flush=True)
            union_spins = _random_spins_mix(problem, seeds)
            for seed in seeds:
                known = known_single_scores.get(case_suffix, {}).get(int(seed), {})
                item = {
                    "seed": int(seed),
                    "score": known.get("score", ""),
                    "rows": BASE_SAMPLE_BUDGET,
                    "source": "known_results",
                }
                if "hv" in known:
                    item["hv"] = known["hv"]
                member_scores.append(item)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        seed_elapsed = time.time() - seed_t0

        if int(union_spins.shape[0]) != BASE_SAMPLE_BUDGET:
            raise ValueError(f"union rows={union_spins.shape[0]}, expected {BASE_SAMPLE_BUDGET}.")

        eval_t0 = time.time()
        hv = _hv_from_spins_safe(problem, union_spins)
        eval_elapsed = time.time() - eval_t0
        gain = max(float(hv) - base, 0.0)
        score = float(gain * 100000.0)
        comparable_members = [
            item
            for item in member_scores
            if item.get("score") not in (None, "")
        ]
        single_best = (
            max(comparable_members, key=lambda item: float(item["score"]))
            if comparable_members
            else {"seed": "", "score": "", "hv": ""}
        )
        verified = verified_scores.get(case_suffix)

        row.update(
            {
                "hv": float(hv),
                "base": base,
                "gain": float(gain),
                "score": score,
                "beats_verified": "" if verified is None else bool(score > float(verified)),
                "single_best_seed": single_best["seed"],
                "single_best_hv": single_best.get("hv", ""),
                "single_best_score": single_best["score"],
                "beats_single_best": "" if single_best["score"] == "" else bool(score > float(single_best["score"])),
                "rows": int(union_spins.shape[0]),
                "sample_used": BASE_SAMPLE_BUDGET,
                "seed_elapsed": float(seed_elapsed),
                "eval_elapsed": float(eval_elapsed),
                "returncode": 0,
                "status": "ok",
                "error": "",
                "member_scores_json": json.dumps(member_scores, separators=(",", ":"), sort_keys=True),
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
        description=(
            "Evaluate 2-seed/4-seed union mixtures by calling answer.main1 once per seed, "
            "taking each seed's quota of rows, and scoring the merged 100000-row sample."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run case 09 with mix 2031+2041.")
    mode.add_argument("--full", action="store_true", help="Allow multi-case or multi-mix grids.")
    parser.add_argument("--cases", nargs="*", help="Case suffixes such as 04 00 09, or explicit .npz paths.")
    parser.add_argument(
        "--include-extra-cases",
        action="store_true",
        help="Append lower-priority headroom cases 06, 02, 08 when --cases is omitted.",
    )
    parser.add_argument(
        "--mixes",
        nargs="*",
        type=_parse_mix,
        help="Seed mixes such as 2031+2041 or 2027+2031+2033+2041.",
    )
    parser.add_argument(
        "--take-mode",
        choices=["stride", "random", "prefix"],
        default="stride",
        help="How to take each seed's quota from its 100000 returned rows.",
    )
    parser.add_argument(
        "--sampler",
        choices=["baseline-qaoa", "answer-main1", "random-spins"],
        default="baseline-qaoa",
        help=(
            "baseline-qaoa directly splits the 100000-shot QAOA baseline across seeds; "
            "answer-main1 calls answer.main1 per seed and subsamples each 100000-row result; "
            "random-spins is a fast fallback for validating the union/evaluation pipeline."
        ),
    )
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument(
        "--verified-file",
        action="append",
        default=[],
        help="Optional JSON file with current verified per-case scores. Can be passed multiple times.",
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun successful existing case/mix rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and resume decisions, then exit.")
    args = parser.parse_args()

    quick = args.quick or not args.full
    if args.cases:
        cases = args.cases
    elif quick:
        cases = QUICK_CASES
    else:
        cases = HEADROOM_CASES + (EXTRA_CASES if args.include_extra_cases else [])

    mixes = args.mixes if args.mixes else (QUICK_MIXES if quick else DEFAULT_MIXES)
    if quick and (len(cases) > 1 or len(mixes) > 1):
        raise SystemExit(
            "Refusing to run a multi-case or multi-mix grid without --full. "
            "Use --full for intentional seed-mixture sweeps."
        )

    case_paths = [_case_path(case) for case in cases]
    missing = [str(path) for path in case_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing case file(s): {', '.join(missing)}")

    verified_paths = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.verified_file]
    verified_scores, verified_sources = _load_verified_scores(verified_paths + VERIFIED_CANDIDATES)
    known_single_scores = _load_known_single_scores()

    out = args.out or _default_out(quick)
    csv_path, json_path = _out_paths(out)
    existing_rows = _load_existing(csv_path, json_path)
    done = {_row_key(row) for row in existing_rows if _is_success(row)}
    done.discard(None)

    planned = [(case, tuple(mix)) for case in case_paths for mix in mixes]
    runnable = [
        (case, mix)
        for case, mix in planned
        if args.rerun or (_case_suffix(case), args.sampler, _mix_id(mix), args.take_mode) not in done
    ]

    if args.dry_run:
        print(f"out_csv={csv_path}")
        print(f"out_json={json_path}")
        print(f"sampler={args.sampler}")
        print(f"take_mode={args.take_mode}")
        print(f"existing_rows={len(existing_rows)}")
        print(f"successful_rows={len(done)}")
        print(f"planned_runs={len(planned)}")
        print(f"runs_to_execute={len(runnable)}")
        for case, mix in runnable:
            print(f"{case},mix={_mix_id(mix)},quota={_quota(len(mix))}")
        return

    rows = existing_rows
    for case, mix in planned:
        key = (_case_suffix(case), args.sampler, _mix_id(mix), args.take_mode)
        if not args.rerun and key in done:
            print(
                f"{case.name},mix={_mix_id(mix)},sampler={args.sampler},"
                f"take_mode={args.take_mode},status=skipped",
                flush=True,
            )
            continue

        row = _evaluate_mix(
            case,
            mix,
            args.sampler,
            args.take_mode,
            verified_scores,
            verified_sources,
            known_single_scores,
        )
        _replace_or_append(rows, row)
        _write_outputs(rows, csv_path, json_path)

        print(
            f"{row['case']},mix={row['mix_id']},sampler={row['sampler']},"
            f"take_mode={row['take_mode']},status={row['status']},"
            f"hv={row['hv']},base={row['base']},score={row['score']},"
            f"verified_score={row['verified_score']},beats_verified={row['beats_verified']},"
            f"single_best_seed={row['single_best_seed']},single_best_score={row['single_best_score']},"
            f"beats_single_best={row['beats_single_best']},rows={row['rows']},elapsed={row['elapsed']:.3f},"
            f"returncode={row['returncode']}",
            flush=True,
        )

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
