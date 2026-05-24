from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_repo_imports() -> None:
    """Load heavy repo dependencies only after argparse/help and path checks."""
    global HV_REF
    global hypervolume_pygmo
    global lexsort_rows
    global load_weight_pool
    global merge_non_dominated_pool
    global normalize_energies
    global objective_extrema
    global pg_non_dominated_indices
    global problem_from_npz

    if "problem_from_npz" in globals():
        return

    from utils import (  # noqa: WPS433
        HV_REF,
        hypervolume_pygmo,
        lexsort_rows,
        load_weight_pool,
        merge_non_dominated_pool,
        normalize_energies,
        objective_extrema,
        pg_non_dominated_indices,
        problem_from_npz,
    )


DEFAULT_RANK_LIMIT = 80
DEFAULT_REGION_LIMIT = 20
HV_EXACT_EVAL_LIMIT = 2500


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _resolve_case(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _case_suffix(case_path: Path) -> str:
    return case_path.stem.rsplit("_", 1)[-1]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_exact_frontier(path: Path) -> dict[str, np.ndarray | float | None]:
    if not path.exists():
        raise FileNotFoundError(f"Exact frontier NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "objectives_norm" not in data:
            raise KeyError(f"{path} does not contain required field 'objectives_norm'")
        objectives = np.asarray(data["objectives_norm"], dtype=np.float64)
        state_indices = np.asarray(data["state_indices"], dtype=np.int64) if "state_indices" in data else None
        energies_raw = np.asarray(data["energies_raw"], dtype=np.float64) if "energies_raw" in data else None
        hv_exact = float(np.asarray(data["hv_exact"]).reshape(-1)[0]) if "hv_exact" in data else None
        lower_bounds = np.asarray(data["lower_bounds"], dtype=np.float64) if "lower_bounds" in data else None
        upper_bounds = np.asarray(data["upper_bounds"], dtype=np.float64) if "upper_bounds" in data else None

    if objectives.ndim != 2:
        raise ValueError(f"objectives_norm must be 2D, got shape {objectives.shape}")
    exact_nd_idx = pg_non_dominated_indices(objectives)
    objectives = np.asarray(objectives[exact_nd_idx], dtype=np.float64)
    if state_indices is not None and int(state_indices.shape[0]) == int(exact_nd_idx.shape[0]):
        state_indices = np.asarray(state_indices, dtype=np.int64)
    elif state_indices is not None and int(state_indices.shape[0]) >= int(np.max(exact_nd_idx, initial=-1)) + 1:
        state_indices = np.asarray(state_indices[exact_nd_idx], dtype=np.int64)
    else:
        state_indices = None

    _, unique_idx = np.unique(objectives, axis=0, return_index=True)
    unique_idx = np.asarray(unique_idx, dtype=np.int64)
    objectives = objectives[unique_idx]
    if state_indices is not None:
        state_indices = state_indices[unique_idx]

    if int(objectives.shape[0]) > 1:
        order = np.lexsort(objectives[:, ::-1].T)
        objectives = objectives[order]
        if state_indices is not None:
            state_indices = state_indices[order]
    return {
        "objectives_norm": objectives,
        "state_indices": state_indices,
        "energies_raw": energies_raw,
        "hv_exact": hv_exact,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }


def _validate_spins(problem: Any, spins: np.ndarray) -> np.ndarray:
    arr = np.asarray(spins, dtype=np.int8)
    if arr.ndim != 2 or int(arr.shape[1]) != int(problem.n):
        raise ValueError(f"sample_spins must have shape [n_samples, {int(problem.n)}], got {arr.shape}")
    if arr.size and not np.all(np.isin(np.unique(arr), np.array([-1, 1], dtype=np.int8))):
        raise ValueError("sample_spins must contain only -1/+1 values")
    return arr


def _objectives_from_spins(problem: Any, spins: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    spins = _validate_spins(problem, spins)
    lower, upper = objective_extrema(problem)
    out = np.empty((int(spins.shape[0]), int(problem.k)), dtype=np.float64)
    for start in range(0, int(spins.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(spins.shape[0]))
        energies = _energy_batch_safe(spins[start:end], problem.edges, problem.weights, problem.h)
        out[start:end] = normalize_energies(energies, lower, upper)
    return out


def _sampled_nd_from_objectives(objs: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_pool = np.zeros((0, int(arr.shape[1])), dtype=np.float64)
    for start in range(0, int(arr.shape[0]), int(chunk_size)):
        block = arr[start : start + int(chunk_size)]
        local = block[pg_non_dominated_indices(block)]
        nd_pool = local if nd_pool.size == 0 else merge_non_dominated_pool(nd_pool, local)
    return np.asarray(lexsort_rows(nd_pool), dtype=np.float64)


def _load_sample_cache(problem: Any, path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "frontier_objectives_norm" in data:
                return np.asarray(data["frontier_objectives_norm"], dtype=np.float64), "frontier_objectives_norm"
            if "objectives_norm" in data:
                return _sampled_nd_from_objectives(np.asarray(data["objectives_norm"], dtype=np.float64)), "objectives_norm"
            if "sample_spins" in data:
                objs = _objectives_from_spins(problem, np.asarray(data["sample_spins"], dtype=np.int8))
                return _sampled_nd_from_objectives(objs), "sample_spins"
        raise KeyError(f"{path} must contain one of sample_spins/objectives_norm/frontier_objectives_norm")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "frontier_objectives_norm" in raw:
            return np.asarray(raw["frontier_objectives_norm"], dtype=np.float64), "frontier_objectives_norm"
        if "objectives_norm" in raw:
            return _sampled_nd_from_objectives(np.asarray(raw["objectives_norm"], dtype=np.float64)), "objectives_norm"
        if "sample_spins" in raw:
            objs = _objectives_from_spins(problem, np.asarray(raw["sample_spins"], dtype=np.int8))
            return _sampled_nd_from_objectives(objs), "sample_spins"
    raise KeyError(f"{path} must contain one of sample_spins/objectives_norm/frontier_objectives_norm")


def _run_or_load_samples(
    problem: Any,
    *,
    seed: int,
    sample_cache: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if sample_cache is not None and sample_cache.exists():
        nd, field = _load_sample_cache(problem, sample_cache)
        return nd, {
            "source": "sample_cache",
            "sample_cache": str(sample_cache),
            "sample_cache_field": field,
            "sample_rows": None,
            "elapsed_s": 0.0,
        }

    import answer  # noqa: WPS433

    t0 = time.time()
    result = answer.main1(problem, rng_seed=int(seed))
    elapsed = time.time() - t0
    if not isinstance(result, dict) or "sample_spins" not in result:
        raise KeyError("answer.main1() must return a dict containing sample_spins")
    spins = _validate_spins(problem, np.asarray(result["sample_spins"], dtype=np.int8))
    objs = _objectives_from_spins(problem, spins)
    sampled_nd = _sampled_nd_from_objectives(objs)

    if sample_cache is not None:
        sample_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_cache,
            sample_spins=spins,
            objectives_norm=objs,
            frontier_objectives_norm=sampled_nd,
            seed=np.asarray([int(seed)], dtype=np.int64),
        )

    return sampled_nd, {
        "source": "answer.main1",
        "sample_cache": str(sample_cache) if sample_cache is not None else None,
        "sample_cache_field": None,
        "sample_rows": int(spins.shape[0]),
        "sample_used": int(result.get("sample_used", int(spins.shape[0]))),
        "elapsed_s": float(elapsed),
    }


def _nearest_distances(
    exact_objs: np.ndarray,
    sampled_nd: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_exact = int(exact_objs.shape[0])
    if sampled_nd.size == 0:
        inf = np.full((n_exact,), np.inf, dtype=np.float64)
        return inf.copy(), inf.copy(), np.full((n_exact,), -1, dtype=np.int64)

    nearest_l2 = np.full((n_exact,), np.inf, dtype=np.float64)
    nearest_linf = np.full((n_exact,), np.inf, dtype=np.float64)
    nearest_idx = np.full((n_exact,), -1, dtype=np.int64)
    sampled = np.asarray(sampled_nd, dtype=np.float64)

    for start in range(0, n_exact, int(chunk_size)):
        block = exact_objs[start : start + int(chunk_size)]
        diff = np.abs(block[:, None, :] - sampled[None, :, :])
        linf = diff.max(axis=2)
        l2 = np.sqrt(np.sum(diff * diff, axis=2))
        idx = np.argmin(l2, axis=1)
        row = np.arange(int(block.shape[0]))
        nearest_l2[start : start + int(block.shape[0])] = l2[row, idx]
        nearest_linf[start : start + int(block.shape[0])] = linf[row, idx]
        nearest_idx[start : start + int(block.shape[0])] = idx.astype(np.int64)

    return nearest_l2, nearest_linf, nearest_idx


def _distance_summary(vals: np.ndarray) -> dict[str, float]:
    finite = np.asarray(vals[np.isfinite(vals)], dtype=np.float64)
    if finite.size == 0:
        return {"min": float("inf"), "p50": float("inf"), "p90": float("inf"), "p95": float("inf"), "max": float("inf")}
    return {
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _single_point_hv_scores(objs: np.ndarray, ref: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if np.isscalar(ref):
        ref_vec = np.full((int(arr.shape[1]),), float(ref), dtype=np.float64)
    else:
        ref_vec = np.asarray(ref, dtype=np.float64).reshape(-1)
    return np.prod(np.maximum(ref_vec[None, :] - arr, 0.0), axis=1)


def _crowding_priority(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    n, k = int(arr.shape[0]), int(arr.shape[1])
    if n <= 2:
        return np.ones((n,), dtype=np.float64)
    mins = arr.min(axis=0)
    span = np.maximum(arr.max(axis=0) - mins, 1e-12)
    scaled = (arr - mins[None, :]) / span[None, :]
    score = np.zeros((n,), dtype=np.float64)
    for d in range(k):
        order = np.argsort(scaled[:, d])
        score[order[0]] += 1.0
        score[order[-1]] += 1.0
        if n > 2:
            score[order[1:-1]] += scaled[order[2:], d] - scaled[order[:-2], d]
    return score


def _lambda_targets(exact_objs: np.ndarray, *, k: int, lambda_count: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(load_weight_pool(int(k), n=int(lambda_count)), dtype=np.float64)
    if weights.ndim != 2 or int(weights.shape[1]) != int(k):
        raise ValueError(f"lambda pool shape mismatch: got {weights.shape}, expected [*, {int(k)}]")
    objs = np.asarray(exact_objs, dtype=np.float64)
    scores = np.empty((int(objs.shape[0]), int(weights.shape[0])), dtype=np.float64)
    for start in range(0, int(objs.shape[0]), 512):
        block = objs[start : start + 512]
        scores[start : start + int(block.shape[0])] = np.einsum(
            "ik,jk->ij",
            block,
            weights,
            optimize=False,
        )
    ids = np.argmin(scores, axis=1).astype(np.int64)
    return ids, weights


def _marginal_hv_gains(
    exact_objs: np.ndarray,
    sampled_nd: np.ndarray,
    nearest_l2: np.ndarray,
    *,
    ref: float,
) -> np.ndarray:
    out = np.zeros((int(exact_objs.shape[0]),), dtype=np.float64)
    sampled_hv = float(hypervolume_pygmo(sampled_nd, ref=ref))
    if exact_objs.size == 0:
        return out

    # Exact one-point marginal HV is useful but expensive. Evaluate all small frontiers,
    # otherwise prioritize distant anchors and leave low-priority gains at zero.
    if int(exact_objs.shape[0]) <= HV_EXACT_EVAL_LIMIT:
        candidate_idx = np.arange(int(exact_objs.shape[0]), dtype=np.int64)
    else:
        candidate_idx = np.argsort(-nearest_l2)[:HV_EXACT_EVAL_LIMIT].astype(np.int64)

    for idx in candidate_idx:
        point = exact_objs[int(idx) : int(idx) + 1]
        merged = point if sampled_nd.size == 0 else np.vstack([sampled_nd, point])
        out[int(idx)] = max(float(hypervolume_pygmo(merged, ref=ref)) - sampled_hv, 0.0)
    return out


def _rank_missing(
    exact_objs: np.ndarray,
    exact_state_indices: np.ndarray | None,
    nearest_l2: np.ndarray,
    nearest_linf: np.ndarray,
    nearest_idx: np.ndarray,
    lambda_ids: np.ndarray,
    hv_gains: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    scale = np.maximum(nearest_l2, 1e-12)
    score = hv_gains * 1.0e3 + scale + 0.25 * nearest_linf
    order = np.argsort(-score)
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[: max(1, int(limit))], start=1):
        i = int(idx)
        row = {
            "rank": int(rank),
            "exact_index": i,
            "state_index": int(exact_state_indices[i]) if exact_state_indices is not None and i < len(exact_state_indices) else None,
            "objective_norm": [float(x) for x in exact_objs[i]],
            "nearest_sampled_nd_index": int(nearest_idx[i]),
            "nearest_l2": float(nearest_l2[i]),
            "nearest_linf": float(nearest_linf[i]),
            "approx_marginal_hv_gain": float(hv_gains[i]),
            "score": float(score[i]),
            "lambda_id": int(lambda_ids[i]),
        }
        rows.append(row)
    return rows


def _regions_from_missing(
    ranked_missing: list[dict[str, Any]],
    *,
    region_limit: int,
) -> list[dict[str, Any]]:
    by_lambda: dict[int, list[dict[str, Any]]] = {}
    for row in ranked_missing:
        by_lambda.setdefault(int(row["lambda_id"]), []).append(row)

    regions: list[dict[str, Any]] = []
    for lambda_id, rows in by_lambda.items():
        objs = np.asarray([r["objective_norm"] for r in rows], dtype=np.float64)
        scores = np.asarray([float(r["score"]) for r in rows], dtype=np.float64)
        gains = np.asarray([float(r["approx_marginal_hv_gain"]) for r in rows], dtype=np.float64)
        l2 = np.asarray([float(r["nearest_l2"]) for r in rows], dtype=np.float64)
        regions.append(
            {
                "lambda_id": int(lambda_id),
                "anchor_count": int(len(rows)),
                "best_rank": int(min(int(r["rank"]) for r in rows)),
                "score_sum": float(np.sum(scores)),
                "approx_marginal_hv_gain_sum": float(np.sum(gains)),
                "max_nearest_l2": float(np.max(l2)),
                "centroid_objective_norm": [float(x) for x in objs.mean(axis=0)],
                "representative_exact_indices": [int(r["exact_index"]) for r in rows[:5]],
            }
        )

    regions.sort(key=lambda r: (-float(r["score_sum"]), int(r["best_rank"])))
    return regions[: max(1, int(region_limit))]


def build_guidance(args: argparse.Namespace) -> dict[str, Any]:
    case_path = _resolve_case(args.case)
    if not case_path.exists():
        raise FileNotFoundError(f"Case NPZ not found: {case_path}")
    exact_path = _resolve_path(args.exact_frontier)
    if not exact_path.exists():
        raise FileNotFoundError(f"Exact frontier NPZ not found: {exact_path}")
    sample_cache = _resolve_path(args.sample_cache) if args.sample_cache else None

    _ensure_repo_imports()
    problem = problem_from_npz(str(case_path))
    exact = _load_exact_frontier(exact_path)
    exact_objs = np.asarray(exact["objectives_norm"], dtype=np.float64)
    if int(exact_objs.shape[1]) != int(problem.k):
        raise ValueError(f"Exact objective dimension {exact_objs.shape[1]} does not match problem.k={int(problem.k)}")

    exact_hv = float(exact["hv_exact"]) if exact["hv_exact"] is not None else float(hypervolume_pygmo(exact_objs, ref=HV_REF))
    lambda_ids, weights = _lambda_targets(exact_objs, k=int(problem.k), lambda_count=int(args.lambda_count))
    if bool(args.exact_only):
        sampled_nd = np.zeros((0, int(problem.k)), dtype=np.float64)
        sample_meta = {
            "source": "exact_only",
            "sample_cache": str(sample_cache) if sample_cache is not None else None,
            "sample_cache_field": None,
            "sample_rows": 0,
            "elapsed_s": 0.0,
            "note": "No answer.main1 sampling was run; lambda recommendations are based on exact-frontier HV/crowding anchors.",
        }
        sampled_hv = None
        nearest_l2 = _crowding_priority(exact_objs)
        nearest_linf = nearest_l2.copy()
        nearest_idx = np.full((int(exact_objs.shape[0]),), -1, dtype=np.int64)
        hv_gains = _single_point_hv_scores(exact_objs, ref=HV_REF)
    else:
        sampled_nd, sample_meta = _run_or_load_samples(problem, seed=int(args.seed), sample_cache=sample_cache)
        sampled_nd = np.asarray(sampled_nd, dtype=np.float64)
        if sampled_nd.ndim != 2 or (sampled_nd.size and int(sampled_nd.shape[1]) != int(problem.k)):
            raise ValueError(f"Sampled ND objective shape mismatch: {sampled_nd.shape}")
        sampled_hv = float(hypervolume_pygmo(sampled_nd, ref=HV_REF))
        nearest_l2, nearest_linf, nearest_idx = _nearest_distances(exact_objs, sampled_nd)
        hv_gains = _marginal_hv_gains(exact_objs, sampled_nd, nearest_l2, ref=HV_REF)

    ranked_missing = _rank_missing(
        exact_objs,
        exact["state_indices"] if isinstance(exact["state_indices"], np.ndarray) else None,
        nearest_l2,
        nearest_linf,
        nearest_idx,
        lambda_ids,
        hv_gains,
        limit=int(args.rank_limit),
    )
    regions = _regions_from_missing(ranked_missing, region_limit=int(args.region_limit))

    recommended_lambda_ids: list[int] = []
    seen: set[int] = set()
    for region in regions:
        lam = int(region["lambda_id"])
        if lam not in seen:
            recommended_lambda_ids.append(lam)
            seen.add(lam)
    if len(recommended_lambda_ids) < int(args.recommend_count):
        counts: dict[int, float] = {}
        for row in ranked_missing:
            counts[int(row["lambda_id"])] = counts.get(int(row["lambda_id"]), 0.0) + float(row["score"])
        for lam, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if lam not in seen:
                recommended_lambda_ids.append(int(lam))
                seen.add(int(lam))
            if len(recommended_lambda_ids) >= int(args.recommend_count):
                break

    return {
        "case": case_path.name,
        "case_suffix": _case_suffix(case_path),
        "case_path": str(case_path),
        "exact_frontier": str(exact_path),
        "seed": int(args.seed),
        "note": "Diagnostic only: exact frontier states are not returned or used as final samples.",
        "counts": {
            "exact_nd": int(exact_objs.shape[0]),
            "sampled_nd": int(sampled_nd.shape[0]),
            "lambda_count": int(weights.shape[0]),
        },
        "hypervolume": {
            "exact": float(exact_hv),
            "sampled": None if sampled_hv is None else float(sampled_hv),
            "gap": None if sampled_hv is None else float(max(exact_hv - sampled_hv, 0.0)),
            "gap_score_units": None if sampled_hv is None else float(max(exact_hv - sampled_hv, 0.0) * 100000.0),
        },
        "nearest_distance": {
            "l2": _distance_summary(nearest_l2),
            "linf": _distance_summary(nearest_linf),
        },
        "sample_source": sample_meta,
        "ranked_missing": ranked_missing,
        "missing_regions": regions,
        "recommended_lambda_ids": recommended_lambda_ids[: max(1, int(args.recommend_count))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact frontier coverage against current answer.main1 samples and recommend lambda IDs."
    )
    parser.add_argument("--case", required=True, help="Case suffix such as 08, or explicit case NPZ path.")
    parser.add_argument("--exact-frontier", required=True, help="Worker A exact frontier NPZ path.")
    parser.add_argument("--seed", type=int, default=2026, help="rng_seed for answer.main1 when no cache exists.")
    parser.add_argument("--out", default=None, help="Output JSON path. Defaults to results/exact_guidance_<case>.json.")
    parser.add_argument("--lambda-count", type=int, default=1000, help="Number of lambda vectors to load from weight pool.")
    parser.add_argument("--sample-cache", default=None, help="Optional NPZ/JSON sample cache. Existing cache is loaded; missing NPZ is written after sampling.")
    parser.add_argument("--exact-only", action="store_true", help="Skip answer.main1 sampling and rank exact frontier anchors by HV/crowding only.")
    parser.add_argument("--rank-limit", type=int, default=DEFAULT_RANK_LIMIT, help="Number of missing anchors to emit.")
    parser.add_argument("--region-limit", type=int, default=DEFAULT_REGION_LIMIT, help="Number of grouped missing regions to emit.")
    parser.add_argument("--recommend-count", type=int, default=25, help="Number of lambda IDs to recommend.")
    args = parser.parse_args()

    payload = build_guidance(args)
    out_path = _resolve_path(args.out) if args.out else ROOT / "results" / f"exact_guidance_{payload['case_suffix']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_to_jsonable),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "case": payload["case"],
                "exact_nd": payload["counts"]["exact_nd"],
                "sampled_nd": payload["counts"]["sampled_nd"],
                "hv_gap_score_units": payload["hypervolume"]["gap_score_units"],
                "recommended_lambda_ids": payload["recommended_lambda_ids"][:10],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
