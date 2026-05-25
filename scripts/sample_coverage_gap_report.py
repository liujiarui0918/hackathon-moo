from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CLUSTER_RADIUS = 0.055
DEFAULT_HV_EVAL_LIMIT = 600
DEFAULT_REGION_LIMIT = 16
DEFAULT_TOP_POINTS = 160


def _ensure_repo_imports() -> None:
    global HV_REF
    global hypervolume_pygmo
    global lexsort_rows
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
        merge_non_dominated_pool,
        normalize_energies,
        objective_extrema,
        pg_non_dominated_indices,
        problem_from_npz,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _resolve_case(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        return _resolve_path(raw)
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _default_exact_frontier(case_path: Path) -> Path:
    return ROOT / "results" / "exact_frontiers" / case_path.name


def _default_sample_cache(case_suffix: str, seed: int | None) -> Path:
    seed_part = "answer_default" if seed is None else f"seed{int(seed)}"
    return ROOT / "results" / "coverage_samples" / f"coverage_sample_{case_suffix}_{seed_part}.npz"


def _default_out(case_suffix: str) -> Path:
    return ROOT / "results" / f"coverage_gap_{case_suffix}.json"


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Energy helper that avoids BLAS-backed matmul paths on this Windows env."""
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _validate_spins(problem: Any, spins: np.ndarray) -> np.ndarray:
    arr = np.asarray(spins, dtype=np.int8)
    if arr.ndim != 2 or int(arr.shape[1]) != int(problem.n):
        raise ValueError(f"sample_spins must have shape [n_samples, {int(problem.n)}], got {arr.shape}")
    if arr.size and not np.all(np.isin(np.unique(arr), np.array([-1, 1], dtype=np.int8))):
        raise ValueError("sample_spins must contain only -1/+1 values")
    return arr


def _objectives_from_spins(problem: Any, spins: np.ndarray, *, chunk_size: int) -> np.ndarray:
    arr = _validate_spins(problem, spins)
    lower, upper = objective_extrema(problem)
    out = np.empty((int(arr.shape[0]), int(problem.k)), dtype=np.float64)
    for start in range(0, int(arr.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(arr.shape[0]))
        energies = _energy_batch_safe(arr[start:end], problem.edges, problem.weights, problem.h)
        out[start:end] = normalize_energies(energies, lower, upper)
    return out


def _sampled_nd_from_objectives(objs: np.ndarray, *, chunk_size: int) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"objectives must be 2D, got {arr.shape}")

    nd_pool = np.zeros((0, int(arr.shape[1])), dtype=np.float64)
    for start in range(0, int(arr.shape[0]), int(chunk_size)):
        block = arr[start : start + int(chunk_size)]
        local = block[pg_non_dominated_indices(block)]
        nd_pool = local if nd_pool.size == 0 else merge_non_dominated_pool(nd_pool, local)
    return np.asarray(lexsort_rows(nd_pool), dtype=np.float64)


def _load_exact_frontier(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Exact frontier NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "objectives_norm" not in data:
            raise KeyError(f"{path} does not contain objectives_norm")
        raw_objs = np.asarray(data["objectives_norm"], dtype=np.float64)
        raw_states = np.asarray(data["state_indices"], dtype=np.int64) if "state_indices" in data else None
        energies_raw = np.asarray(data["energies_raw"], dtype=np.float64) if "energies_raw" in data else None
        hv_exact = float(np.asarray(data["hv_exact"]).reshape(-1)[0]) if "hv_exact" in data else None
        lower = np.asarray(data["lower_bounds"], dtype=np.float64) if "lower_bounds" in data else None
        upper = np.asarray(data["upper_bounds"], dtype=np.float64) if "upper_bounds" in data else None

    if raw_objs.ndim != 2:
        raise ValueError(f"objectives_norm must be 2D, got {raw_objs.shape}")

    nd_idx = pg_non_dominated_indices(raw_objs)
    objs = np.asarray(raw_objs[nd_idx], dtype=np.float64)
    states = None
    if raw_states is not None:
        if int(raw_states.shape[0]) == int(raw_objs.shape[0]):
            states = np.asarray(raw_states[nd_idx], dtype=np.int64)
        elif int(raw_states.shape[0]) == int(objs.shape[0]):
            states = np.asarray(raw_states, dtype=np.int64)

    _, unique_idx = np.unique(objs, axis=0, return_index=True)
    unique_idx = np.asarray(unique_idx, dtype=np.int64)
    objs = objs[unique_idx]
    if states is not None:
        states = states[unique_idx]

    if int(objs.shape[0]) > 1:
        order = np.lexsort(objs[:, ::-1].T)
        objs = objs[order]
        if states is not None:
            states = states[order]

    return {
        "objectives_norm": objs,
        "state_indices": states,
        "energies_raw": energies_raw,
        "hv_exact": hv_exact,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }


def _load_sample_cache(problem: Any, path: Path, *, chunk_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    if path.suffix.lower() != ".npz":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Sample cache JSON must be an object: {path}")
        if "frontier_objectives_norm" in raw:
            nd = np.asarray(raw["frontier_objectives_norm"], dtype=np.float64)
            field = "frontier_objectives_norm"
        elif "objectives_norm" in raw:
            nd = _sampled_nd_from_objectives(np.asarray(raw["objectives_norm"], dtype=np.float64), chunk_size=chunk_size)
            field = "objectives_norm"
        elif "sample_spins" in raw:
            objs = _objectives_from_spins(problem, np.asarray(raw["sample_spins"], dtype=np.int8), chunk_size=chunk_size)
            nd = _sampled_nd_from_objectives(objs, chunk_size=chunk_size)
            field = "sample_spins"
        else:
            raise KeyError(f"{path} must contain sample_spins/objectives_norm/frontier_objectives_norm")
        return nd, {"source": "sample_cache", "sample_cache": str(path), "sample_cache_field": field}

    with np.load(path, allow_pickle=False) as data:
        if "frontier_objectives_norm" in data:
            nd = np.asarray(data["frontier_objectives_norm"], dtype=np.float64)
            field = "frontier_objectives_norm"
        elif "objectives_norm" in data:
            nd = _sampled_nd_from_objectives(np.asarray(data["objectives_norm"], dtype=np.float64), chunk_size=chunk_size)
            field = "objectives_norm"
        elif "sample_spins" in data:
            objs = _objectives_from_spins(problem, np.asarray(data["sample_spins"], dtype=np.int8), chunk_size=chunk_size)
            nd = _sampled_nd_from_objectives(objs, chunk_size=chunk_size)
            field = "sample_spins"
        else:
            raise KeyError(f"{path} must contain sample_spins/objectives_norm/frontier_objectives_norm")

        meta = {
            "source": "sample_cache",
            "sample_cache": str(path),
            "sample_cache_field": field,
            "seed": int(np.asarray(data["seed"]).reshape(-1)[0]) if "seed" in data else None,
            "sample_rows": int(np.asarray(data["sample_rows"]).reshape(-1)[0]) if "sample_rows" in data else None,
            "sample_used": int(np.asarray(data["sample_used"]).reshape(-1)[0]) if "sample_used" in data else None,
            "answer_mtime": float(np.asarray(data["answer_mtime"]).reshape(-1)[0]) if "answer_mtime" in data else None,
            "elapsed_s": float(np.asarray(data["elapsed_s"]).reshape(-1)[0]) if "elapsed_s" in data else 0.0,
        }
    return nd, meta


def _run_or_load_samples(
    problem: Any,
    *,
    seed: int | None,
    sample_cache: Path,
    refresh_samples: bool,
    allow_stale_cache: bool,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    answer_path = ROOT / "answer.py"
    answer_mtime = answer_path.stat().st_mtime if answer_path.exists() else None
    if sample_cache.exists() and not refresh_samples:
        nd, meta = _load_sample_cache(problem, sample_cache, chunk_size=chunk_size)
        cached_mtime = meta.get("answer_mtime")
        stale = (
            answer_mtime is not None
            and cached_mtime is not None
            and abs(float(cached_mtime) - float(answer_mtime)) > 1.0e-6
        )
        if not stale or allow_stale_cache:
            meta["stale_answer_cache"] = bool(stale)
            return nd, meta

    import answer  # noqa: WPS433

    t0 = time.time()
    result = answer.main1(problem, rng_seed=seed) if seed is not None else answer.main1(problem)
    elapsed = time.time() - t0
    if not isinstance(result, dict) or "sample_spins" not in result:
        raise KeyError("answer.main1() must return a dict containing sample_spins")

    spins = _validate_spins(problem, np.asarray(result["sample_spins"], dtype=np.int8))
    objs = _objectives_from_spins(problem, spins, chunk_size=chunk_size)
    sampled_nd = _sampled_nd_from_objectives(objs, chunk_size=chunk_size)

    sample_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_cache,
        sample_spins=spins,
        objectives_norm=objs,
        frontier_objectives_norm=sampled_nd,
        seed=np.asarray([-1 if seed is None else int(seed)], dtype=np.int64),
        sample_rows=np.asarray([int(spins.shape[0])], dtype=np.int64),
        sample_used=np.asarray([int(result.get("sample_used", int(spins.shape[0])))], dtype=np.int64),
        answer_mtime=np.asarray([0.0 if answer_mtime is None else float(answer_mtime)], dtype=np.float64),
        elapsed_s=np.asarray([float(elapsed)], dtype=np.float64),
    )
    return sampled_nd, {
        "source": "answer.main1",
        "sample_cache": str(sample_cache),
        "sample_cache_field": None,
        "seed": seed,
        "sample_rows": int(spins.shape[0]),
        "sample_used": int(result.get("sample_used", int(spins.shape[0]))),
        "answer_mtime": answer_mtime,
        "elapsed_s": float(elapsed),
        "stale_answer_cache": False,
    }


def _nearest_distances(
    exact_objs: np.ndarray,
    sampled_nd: np.ndarray,
    *,
    chunk_size: int,
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
        rows = np.arange(int(block.shape[0]))
        nearest_l2[start : start + int(block.shape[0])] = l2[rows, idx]
        nearest_linf[start : start + int(block.shape[0])] = linf[rows, idx]
        nearest_idx[start : start + int(block.shape[0])] = idx.astype(np.int64)

    return nearest_l2, nearest_linf, nearest_idx


def _summary(vals: np.ndarray) -> dict[str, float]:
    finite = np.asarray(vals[np.isfinite(vals)], dtype=np.float64)
    if finite.size == 0:
        return {"min": float("inf"), "p50": float("inf"), "p75": float("inf"), "p90": float("inf"), "p95": float("inf"), "max": float("inf")}
    return {
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p75": float(np.quantile(finite, 0.75)),
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


def _marginal_hv_gains(
    exact_objs: np.ndarray,
    sampled_nd: np.ndarray,
    nearest_l2: np.ndarray,
    *,
    ref: float,
    eval_limit: int,
) -> tuple[np.ndarray, list[int]]:
    gains = np.zeros((int(exact_objs.shape[0]),), dtype=np.float64)
    if exact_objs.size == 0:
        return gains, []

    single_scores = _single_point_hv_scores(exact_objs, ref=ref)
    by_dist = np.argsort(-nearest_l2)[: max(1, int(eval_limit) // 2)]
    by_single = np.argsort(-single_scores)[: max(1, int(eval_limit))]
    candidate_idx = np.unique(np.concatenate([by_dist, by_single]))[: max(1, int(eval_limit))]

    sampled_hv = float(hypervolume_pygmo(sampled_nd, ref=ref)) if sampled_nd.size else 0.0
    for idx in candidate_idx:
        point = exact_objs[int(idx) : int(idx) + 1]
        merged = point if sampled_nd.size == 0 else np.vstack([sampled_nd, point])
        gains[int(idx)] = max(float(hypervolume_pygmo(merged, ref=ref)) - sampled_hv, 0.0)
    return gains, [int(x) for x in candidate_idx]


def _normalized_score(nearest_l2: np.ndarray, nearest_linf: np.ndarray, hv_gains: np.ndarray) -> np.ndarray:
    def norm(vals: np.ndarray) -> np.ndarray:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return np.zeros_like(vals, dtype=np.float64)
        hi = float(np.max(finite))
        if hi <= 1.0e-15:
            return np.zeros_like(vals, dtype=np.float64)
        return np.nan_to_num(vals / hi, nan=0.0, posinf=1.0, neginf=0.0)

    return 0.55 * norm(nearest_l2) + 0.25 * norm(nearest_linf) + 0.20 * norm(hv_gains)


def _missing_candidate_ids(
    score: np.ndarray,
    nearest_l2: np.ndarray,
    hv_gains: np.ndarray,
    *,
    min_l2: float,
    quantile: float,
    top_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    finite_l2 = nearest_l2[np.isfinite(nearest_l2)]
    auto_threshold = float(np.quantile(finite_l2, float(quantile))) if finite_l2.size else float("inf")
    threshold = max(float(min_l2), auto_threshold)
    hv_positive = hv_gains > 0.0
    mask = (nearest_l2 >= threshold) | hv_positive
    ids = np.flatnonzero(mask)
    order = np.argsort(-score)
    if ids.size < max(1, int(top_points)):
        ids = np.unique(np.concatenate([ids, order[: max(1, int(top_points))]]))
    ids = ids[np.argsort(-score[ids])]
    return ids[: max(1, int(top_points))], {
        "min_l2": float(min_l2),
        "quantile": float(quantile),
        "auto_l2_threshold": auto_threshold,
        "used_l2_threshold": threshold,
        "candidate_count_before_cap": int(np.count_nonzero(mask)),
        "candidate_count": int(min(int(ids.size), max(1, int(top_points)))),
    }


def _action_for_region(region: dict[str, Any]) -> tuple[str, str]:
    centroid = np.asarray(region["centroid_objective_norm"], dtype=np.float64)
    span = np.asarray(region["objective_span"], dtype=np.float64)
    mean_l2 = float(region["mean_nearest_l2"])
    max_l2 = float(region["max_nearest_l2"])
    hv_gain = float(region["approx_marginal_hv_gain_sum"])
    compact = float(np.mean(span)) <= 0.070
    broad = float(np.mean(span)) >= 0.145 or int(region["anchor_count"]) >= 18
    extreme_low = np.flatnonzero(centroid <= 0.14).astype(int).tolist()
    extreme_high = np.flatnonzero(centroid >= 0.78).astype(int).tolist()

    if mean_l2 <= 0.018 and hv_gain <= 1.0e-8:
        return (
            "seed_mix",
            "Gap is close to the sampled frontier; run a seed mix and keep the best sampled ND union.",
        )
    if compact and max_l2 >= 0.030 and not broad:
        return (
            "legal_local_warm",
            "Compact uncovered pocket; expand legal local-search warm starts around this objective zone without injecting exact states.",
        )
    if extreme_low or extreme_high:
        axes = sorted(set(extreme_low + extreme_high))
        return (
            "broad_shots",
            f"Region sits on objective-axis extremes {axes}; shift budget toward broader scalarization shots/directions.",
        )
    if broad:
        return (
            "parameter_portfolio",
            "Gap is spread across a wide tradeoff patch; try a small beta/gamma/warm_c portfolio instead of one schedule.",
        )
    return (
        "parameter_portfolio",
        "Mid-front hole with moderate distance; use parameter portfolio coverage before exact-frontier-specific tactics.",
    )


def _rank_regions(
    exact_objs: np.ndarray,
    exact_states: np.ndarray | None,
    nearest_l2: np.ndarray,
    nearest_linf: np.ndarray,
    nearest_idx: np.ndarray,
    hv_gains: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    region_limit: int,
    cluster_radius: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score = _normalized_score(nearest_l2, nearest_linf, hv_gains)
    assigned: set[int] = set()
    regions: list[dict[str, Any]] = []
    ranked_points: list[dict[str, Any]] = []
    candidate_set = np.asarray([int(x) for x in candidate_ids], dtype=np.int64)

    for idx in candidate_set[np.argsort(-score[candidate_set])]:
        i = int(idx)
        ranked_points.append(
            {
                "rank": int(len(ranked_points) + 1),
                "exact_index": i,
                "state_index": int(exact_states[i]) if exact_states is not None and i < len(exact_states) else None,
                "objective_norm": [float(x) for x in exact_objs[i]],
                "nearest_sampled_nd_index": int(nearest_idx[i]),
                "nearest_l2": float(nearest_l2[i]),
                "nearest_linf": float(nearest_linf[i]),
                "approx_marginal_hv_gain": float(hv_gains[i]),
                "coverage_score": float(score[i]),
            }
        )
        if len(ranked_points) >= max(1, int(candidate_ids.size)):
            break

    for anchor in candidate_set[np.argsort(-score[candidate_set])]:
        a = int(anchor)
        if a in assigned:
            continue
        remaining = np.asarray([int(x) for x in candidate_set if int(x) not in assigned], dtype=np.int64)
        if remaining.size == 0:
            break
        d = np.sqrt(np.sum(np.square(exact_objs[remaining] - exact_objs[a][None, :]), axis=1))
        members = remaining[d <= float(cluster_radius)]
        if members.size == 0:
            members = np.asarray([a], dtype=np.int64)
        for m in members:
            assigned.add(int(m))

        objs = exact_objs[members]
        member_scores = score[members]
        member_hv = hv_gains[members]
        member_l2 = nearest_l2[members]
        member_linf = nearest_linf[members]
        order = members[np.argsort(-score[members])]
        centroid = objs.mean(axis=0)
        mn = objs.min(axis=0)
        mx = objs.max(axis=0)
        region = {
            "rank": int(len(regions) + 1),
            "anchor_exact_index": a,
            "anchor_state_index": int(exact_states[a]) if exact_states is not None and a < len(exact_states) else None,
            "anchor_objective_norm": [float(x) for x in exact_objs[a]],
            "anchor_count": int(members.size),
            "coverage_score_sum": float(np.sum(member_scores)),
            "approx_marginal_hv_gain_sum": float(np.sum(member_hv)),
            "mean_nearest_l2": float(np.mean(member_l2)),
            "max_nearest_l2": float(np.max(member_l2)),
            "mean_nearest_linf": float(np.mean(member_linf)),
            "max_nearest_linf": float(np.max(member_linf)),
            "centroid_objective_norm": [float(x) for x in centroid],
            "objective_min": [float(x) for x in mn],
            "objective_max": [float(x) for x in mx],
            "objective_span": [float(x) for x in mx - mn],
            "representative_exact_indices": [int(x) for x in order[:8]],
        }
        action, reason = _action_for_region(region)
        region["action"] = action
        region["action_reason"] = reason
        regions.append(region)
        if len(regions) >= max(1, int(region_limit)):
            break

    regions.sort(key=lambda r: (-float(r["coverage_score_sum"]), -float(r["approx_marginal_hv_gain_sum"])))
    for rank, region in enumerate(regions, start=1):
        region["rank"] = int(rank)
    return regions, ranked_points


def _action_summary(regions: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, float] = {
        "seed_mix": 0.0,
        "parameter_portfolio": 0.0,
        "legal_local_warm": 0.0,
        "broad_shots": 0.0,
    }
    counts = {key: 0 for key in scores}
    for region in regions:
        action = str(region["action"])
        scores[action] = scores.get(action, 0.0) + float(region["coverage_score_sum"])
        counts[action] = counts.get(action, 0) + int(region["anchor_count"])

    priority = max(scores, key=lambda key: (scores[key], counts.get(key, 0))) if regions else "seed_mix"
    details = {
        "seed_mix": "Run several current answer.main1 seeds, then HV-greedy/select from the legal sampled union.",
        "parameter_portfolio": "Test a small parameter portfolio for beta/gamma/warm_c/selector settings; evaluate by sampled ND HV.",
        "legal_local_warm": "Increase legal multi-objective local-search warm-start coverage in compact holes; exact states remain diagnostic only.",
        "broad_shots": "Move more budget to broad scalarization directions or shots when uncovered regions are extreme/sparse.",
    }
    return {
        "priority_action": priority,
        "score_by_action": {key: float(value) for key, value in sorted(scores.items())},
        "anchor_count_by_action": {key: int(value) for key, value in sorted(counts.items())},
        "recommendations": details,
        "avoid": "Do not use the prior exact-only lambda strategy; exact frontier points are only diagnostics for where legal samples are missing.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_repo_imports()
    case_path = _resolve_case(args.case)
    if not case_path.exists():
        raise FileNotFoundError(f"Case NPZ not found: {case_path}")
    case_suffix = _case_suffix(case_path)
    exact_path = _resolve_path(args.exact_frontier) if args.exact_frontier else _default_exact_frontier(case_path)
    sample_cache = _resolve_path(args.sample_cache) if args.sample_cache else _default_sample_cache(case_suffix, args.seed)

    problem = problem_from_npz(str(case_path))
    exact = _load_exact_frontier(exact_path)
    exact_objs = np.asarray(exact["objectives_norm"], dtype=np.float64)
    if exact_objs.ndim != 2 or int(exact_objs.shape[1]) != int(problem.k):
        raise ValueError(f"Exact frontier shape {exact_objs.shape} does not match problem.k={int(problem.k)}")

    sampled_nd, sample_meta = _run_or_load_samples(
        problem,
        seed=args.seed,
        sample_cache=sample_cache,
        refresh_samples=bool(args.refresh_samples),
        allow_stale_cache=bool(args.allow_stale_cache),
        chunk_size=int(args.chunk_size),
    )
    sampled_nd = np.asarray(sampled_nd, dtype=np.float64)
    if sampled_nd.ndim != 2 or (sampled_nd.size and int(sampled_nd.shape[1]) != int(problem.k)):
        raise ValueError(f"Sampled ND shape {sampled_nd.shape} does not match problem.k={int(problem.k)}")

    exact_hv = float(exact["hv_exact"]) if exact["hv_exact"] is not None else float(hypervolume_pygmo(exact_objs, ref=HV_REF))
    sampled_hv = float(hypervolume_pygmo(sampled_nd, ref=HV_REF)) if sampled_nd.size else 0.0
    nearest_l2, nearest_linf, nearest_idx = _nearest_distances(
        exact_objs,
        sampled_nd,
        chunk_size=int(args.distance_chunk_size),
    )
    hv_gains, hv_eval_indices = _marginal_hv_gains(
        exact_objs,
        sampled_nd,
        nearest_l2,
        ref=HV_REF,
        eval_limit=int(args.hv_eval_limit),
    )
    score = _normalized_score(nearest_l2, nearest_linf, hv_gains)
    candidate_ids, missing_meta = _missing_candidate_ids(
        score,
        nearest_l2,
        hv_gains,
        min_l2=float(args.min_l2),
        quantile=float(args.missing_quantile),
        top_points=int(args.top_points),
    )
    regions, ranked_points = _rank_regions(
        exact_objs,
        exact["state_indices"] if isinstance(exact["state_indices"], np.ndarray) else None,
        nearest_l2,
        nearest_linf,
        nearest_idx,
        hv_gains,
        candidate_ids,
        region_limit=int(args.region_limit),
        cluster_radius=float(args.cluster_radius),
    )

    exact_matched = int(np.count_nonzero(nearest_l2 <= float(args.covered_l2)))
    return {
        "case": case_path.name,
        "case_suffix": case_suffix,
        "case_path": _project_path(case_path),
        "exact_frontier": _project_path(exact_path),
        "created_at_unix": float(time.time()),
        "note": "Coverage diagnostic only. Exact frontier states are not recommended as returned samples.",
        "counts": {
            "exact_nd": int(exact_objs.shape[0]),
            "sampled_nd": int(sampled_nd.shape[0]),
            "exact_points_within_covered_l2": exact_matched,
            "exact_covered_fraction": float(exact_matched / max(1, int(exact_objs.shape[0]))),
            "hv_gain_evaluated_points": int(len(hv_eval_indices)),
        },
        "hypervolume": {
            "exact": float(exact_hv),
            "sampled": float(sampled_hv),
            "gap": float(max(exact_hv - sampled_hv, 0.0)),
            "gap_score_units": float(max(exact_hv - sampled_hv, 0.0) * 100000.0),
        },
        "nearest_distance": {
            "l2": _summary(nearest_l2),
            "linf": _summary(nearest_linf),
            "covered_l2": float(args.covered_l2),
        },
        "missing_selection": missing_meta,
        "sample_source": sample_meta,
        "action_recommendation": _action_summary(regions),
        "missing_regions": regions,
        "ranked_missing_points": ranked_points[: max(1, int(args.top_points))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact frontier coverage against current answer.main1 sampled ND and rank missing regions."
    )
    parser.add_argument("--case", required=True, help="Case suffix such as 09, or explicit case NPZ path.")
    parser.add_argument("--exact-frontier", default=None, help="Exact frontier NPZ. Defaults to results/exact_frontiers/<case>.npz.")
    parser.add_argument("--seed", type=int, default=2026, help="rng_seed for answer.main1. Use a cached sample unless refreshed.")
    parser.add_argument("--sample-cache", default=None, help="NPZ/JSON sample cache. Defaults to results/coverage_samples/.")
    parser.add_argument("--refresh-samples", action="store_true", help="Ignore sample cache and rerun answer.main1.")
    parser.add_argument("--allow-stale-cache", action="store_true", help="Allow cache generated from an older answer.py mtime.")
    parser.add_argument("--out", default=None, help="Output JSON path. Defaults to results/coverage_gap_<case>.json.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Energy/objective chunk size.")
    parser.add_argument("--distance-chunk-size", type=int, default=512, help="Nearest-distance exact-frontier chunk size.")
    parser.add_argument("--hv-eval-limit", type=int, default=DEFAULT_HV_EVAL_LIMIT, help="Max exact points for marginal HV probes.")
    parser.add_argument("--top-points", type=int, default=DEFAULT_TOP_POINTS, help="Number of missing exact anchors to keep.")
    parser.add_argument("--region-limit", type=int, default=DEFAULT_REGION_LIMIT, help="Number of missing regions to emit.")
    parser.add_argument("--cluster-radius", type=float, default=DEFAULT_CLUSTER_RADIUS, help="Greedy objective-space region radius.")
    parser.add_argument("--missing-quantile", type=float, default=0.80, help="Nearest-L2 quantile used for missing candidate threshold.")
    parser.add_argument("--min-l2", type=float, default=0.0125, help="Minimum nearest-L2 threshold for a missing candidate.")
    parser.add_argument("--covered-l2", type=float, default=1.0e-9, help="L2 tolerance for exact objective rows already hit by samples.")
    args = parser.parse_args()

    payload = build_report(args)
    out_path = _resolve_path(args.out) if args.out else _default_out(str(payload["case_suffix"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable),
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
                "priority_action": payload["action_recommendation"]["priority_action"],
                "top_region_action": payload["missing_regions"][0]["action"] if payload["missing_regions"] else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
