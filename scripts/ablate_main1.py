from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import baseline_hv
from utils import (
    build_qaoa_circuit_from_projected_ising,
    exact_frontier_from_lambda_unique_batches,
    load_transfer_params_csv,
    load_weight_pool,
    objective_extrema,
    problem_from_npz,
    sampling_result_to_unique_spins,
    sampling_result_to_spins,
    normalize_energies,
    objective_extrema,
    merge_non_dominated_pool,
    hypervolume_pygmo,
    pg_non_dominated_indices,
)


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _hv_from_spins_safe(problem, spins: np.ndarray) -> float:
    if spins.size == 0:
        return 0.0
    lower, upper = objective_extrema(problem)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)
    arr = np.asarray(spins, dtype=np.int8)
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


def _sample_round(
    problem,
    lambda_ids: np.ndarray,
    shots: int,
    proj_j: np.ndarray,
    proj_h: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    seed: int,
    warm_bits: list[np.ndarray | None] | None = None,
    warm_c: float = 0.4,
):
    sim = Simulator("mqvector", int(problem.n), seed=int(seed))
    dense_parts = []
    unique_blocks = []
    count_blocks = []
    for pos, lam_id in enumerate(np.asarray(lambda_ids, dtype=np.int64)):
        wb = None if warm_bits is None else warm_bits[pos]
        circ = build_qaoa_circuit_from_projected_ising(
            problem,
            proj_j[int(lam_id)],
            proj_h[int(lam_id)],
            betas=betas,
            gammas=gammas,
            warm_bits01=wb,
            warm_c=warm_c,
        )
        sim.reset()
        res = sim.sampling(circ, shots=int(shots), seed=int(seed + pos))
        unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
        dense_parts.append(np.repeat(unique_spins, counts.astype(np.int32), axis=0))
        unique_blocks.append(unique_spins)
        count_blocks.append(counts)
    return (
        np.vstack(dense_parts).astype(np.int8),
        unique_blocks,
        count_blocks,
        np.asarray(lambda_ids, dtype=np.int64),
    )


def _sample_param_portfolio(
    problem,
    lambda_ids: np.ndarray,
    shots: int,
    proj_j: np.ndarray,
    proj_h: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    seed: int,
    gamma_scales: list[float],
    beta_scales: list[float] | None = None,
):
    if beta_scales is None:
        beta_scales = [1.0] * len(gamma_scales)
    dense_parts = []
    pos = 0
    for gamma_scale, beta_scale in zip(gamma_scales, beta_scales):
        scaled_betas = np.asarray(betas, dtype=np.float64) * float(beta_scale)
        scaled_gammas = np.asarray(gammas, dtype=np.float64) * float(gamma_scale)
        s, *_ = _sample_round(
            problem,
            lambda_ids,
            shots,
            proj_j,
            proj_h,
            scaled_betas,
            scaled_gammas,
            seed=int(seed + pos * 100000),
        )
        dense_parts.append(s)
        pos += 1
    return np.vstack(dense_parts).astype(np.int8)


def _frontier_warm_bits(problem, unique_blocks, count_blocks, lambda_ids):
    lower, upper = objective_extrema(problem)
    objs, spins, lams, counts = exact_frontier_from_lambda_unique_batches(
        unique_blocks,
        count_blocks,
        list(np.asarray(lambda_ids, dtype=np.int64)),
        edges=problem.edges,
        weights=problem.weights,
        h=problem.h,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    if spins.size == 0:
        return objs, spins, lams, counts, []
    # One strong incumbent per requested lambda: choose sampled frontier point
    # with smallest scalarized normalized objective under that lambda.
    return objs, spins, lams, counts, None


def _make_warm_for_ids(frontier_objs, frontier_spins, next_ids, pool, n_qubits: int):
    if frontier_spins.size == 0:
        return [None] * int(len(next_ids))
    warm = []
    for lam_id in np.asarray(next_ids, dtype=np.int64):
        scalar = np.einsum(
            "ij,j->i",
            np.asarray(frontier_objs, dtype=np.float64),
            np.asarray(pool[int(lam_id)], dtype=np.float64),
            optimize=False,
        )
        idx = int(np.argmin(scalar))
        warm.append(np.where(frontier_spins[idx] > 0, 0, 1).astype(np.int8))
    return warm


def _frontier_gap_ids(frontier_objs, pool, *, count: int):
    objs = np.asarray(frontier_objs, dtype=np.float64)
    if objs.size == 0:
        return np.arange(int(count), dtype=np.int64)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    scaled = (objs - mins[None, :]) / span[None, :]
    order = np.lexsort(scaled[:, ::-1].T)
    sorted_scaled = scaled[order]
    targets = []
    k = int(objs.shape[1])
    targets.extend(np.eye(k, dtype=np.float64))
    if sorted_scaled.shape[0] > 1:
        diffs = np.diff(sorted_scaled, axis=0)
        gaps = np.linalg.norm(diffs, axis=1)
        for idx in np.argsort(-gaps):
            mid = 0.5 * (sorted_scaled[int(idx)] + sorted_scaled[int(idx) + 1])
            s = float(mid.sum())
            if s > 1e-12:
                targets.append(mid / s)
            if len(targets) >= int(count) * 4:
                break
    # Also use regret normals over sparse frontier points.
    cd = np.zeros((scaled.shape[0],), dtype=np.float64)
    for d in range(k):
        od = np.argsort(scaled[:, d])
        cd[od[0]] = np.inf
        cd[od[-1]] = np.inf
        if len(od) > 2:
            cd[od[1:-1]] += scaled[od[2:], d] - scaled[od[:-2], d]
    for idx in np.argsort(-cd):
        row = scaled[int(idx)]
        s = float(row.sum())
        if s > 1e-12:
            targets.append(row / s)
        if len(targets) >= int(count) * 8:
            break
    if not targets:
        return np.arange(int(count), dtype=np.int64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    out = []
    seen = set()
    for target in targets_arr:
        d = pool - target[None, :]
        order2 = np.argsort(np.einsum("ij,ij->i", d, d, optimize=False))
        for oid in order2[:16]:
            val = int(oid)
            if val not in seen:
                out.append(val)
                seen.add(val)
                break
        if len(out) >= int(count):
            break
    if len(out) < int(count):
        for val in range(int(pool.shape[0])):
            if val not in seen:
                out.append(val)
                seen.add(val)
            if len(out) >= int(count):
                break
    return np.asarray(out[: int(count)], dtype=np.int64)


def _greedy_ids_from_sampled_blocks(problem, unique_blocks, candidate_ids, *, count: int):
    lower, upper = objective_extrema(problem)
    per_objs = []
    for spins in unique_blocks:
        e = _energy_batch_safe(spins, problem.edges, problem.weights, problem.h)
        objs = normalize_energies(e, lower, upper)
        per_objs.append(objs[pg_non_dominated_indices(objs)])
    selected = []
    selected_set = set()
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)
    remaining = list(range(len(per_objs)))
    for _ in range(min(int(count), len(remaining))):
        best_pos = None
        best_hv = -1.0
        best_pool = None
        # Evaluate a deterministic candidate subset each step to keep this cheap.
        cand_positions = remaining if len(remaining) <= 96 else remaining[:32] + remaining[len(remaining)//3:len(remaining)//3+32] + remaining[-32:]
        for pos in cand_positions:
            merged = merge_non_dominated_pool(nd_pool, per_objs[pos])
            hv = hypervolume_pygmo(merged)
            if hv > best_hv:
                best_hv = hv
                best_pos = pos
                best_pool = merged
        if best_pos is None:
            break
        selected.append(int(candidate_ids[best_pos]))
        selected_set.add(best_pos)
        nd_pool = np.asarray(best_pool, dtype=np.float64)
        remaining = [x for x in remaining if x != best_pos]
    if len(selected) < int(count):
        for lam_id in candidate_ids:
            val = int(lam_id)
            if val not in selected:
                selected.append(val)
            if len(selected) >= int(count):
                break
    return np.asarray(selected[: int(count)], dtype=np.int64)


def _scalar_local_descent_spin(problem, j_raw, h_raw, *, seed: int, restarts: int = 8):
    rng = np.random.default_rng(int(seed))
    n = int(problem.n)
    edges = np.asarray(problem.edges, dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]
    j = np.asarray(j_raw, dtype=np.float64)
    h = np.asarray(h_raw, dtype=np.float64)

    starts = [
        np.ones((n,), dtype=np.int8),
        -np.ones((n,), dtype=np.int8),
        np.where(h <= 0.0, 1, -1).astype(np.int8),
        np.where(h >= 0.0, 1, -1).astype(np.int8),
    ]
    for _ in range(max(0, int(restarts) - len(starts))):
        starts.append(np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8))

    best = None
    best_e = np.inf
    for start in starts[: int(restarts)]:
        z = np.asarray(start, dtype=np.int8).copy()
        improved = True
        sweeps = 0
        while improved and sweeps < 32:
            improved = False
            sweeps += 1
            order = rng.permutation(n)
            for q in order:
                q = int(q)
                field = float(h[q])
                mask_u = u == q
                if np.any(mask_u):
                    field += float(np.dot(j[mask_u], z[v[mask_u]]))
                mask_v = v == q
                if np.any(mask_v):
                    field += float(np.dot(j[mask_v], z[u[mask_v]]))
                delta = -2.0 * float(z[q]) * field
                if delta < -1e-12:
                    z[q] = np.int8(-z[q])
                    improved = True
        e = float(np.dot(h, z) + np.dot(j, z[u] * z[v]))
        if e < best_e:
            best_e = e
            best = z.copy()
    return np.asarray(best, dtype=np.int8)


def _local_descent_warm_for_ids(problem, ids, proj_j, proj_h, *, seed: int, restarts: int = 8):
    out = []
    for pos, lam_id in enumerate(np.asarray(ids, dtype=np.int64)):
        z = _scalar_local_descent_spin(
            problem,
            proj_j[int(lam_id)],
            proj_h[int(lam_id)],
            seed=int(seed + pos * 7919),
            restarts=int(restarts),
        )
        out.append(np.where(z > 0, 0, 1).astype(np.int8))
    return out


def _multiobjective_local_candidates(problem, pool, proj_j, proj_h, *, seed: int, restarts: int = 6):
    spins = []
    lam_ids = []
    for lam_id in range(int(pool.shape[0])):
        for r in range(int(restarts)):
            z = _scalar_local_descent_spin(
                problem,
                proj_j[lam_id],
                proj_h[lam_id],
                seed=int(seed + lam_id * 1009 + r * 9176),
                restarts=1,
            )
            spins.append(z)
            lam_ids.append(lam_id)
    spins_arr = np.unique(np.asarray(spins, dtype=np.int8), axis=0)
    lower, upper = objective_extrema(problem)
    objs = normalize_energies(
        _energy_batch_safe(spins_arr, problem.edges, problem.weights, problem.h),
        lower,
        upper,
    )
    nd = pg_non_dominated_indices(objs)
    nd_spins = spins_arr[nd]
    nd_objs = objs[nd]
    if nd_spins.shape[0] == 0:
        return nd_spins, nd_objs, np.zeros((0,), dtype=np.int64)
    # Assign each ND state to the lambda under which it is best scalarized.
    scalar = np.einsum("ik,jk->ij", nd_objs, pool, optimize=False)
    nd_lams = np.argmin(scalar, axis=1).astype(np.int64)
    return nd_spins, nd_objs, nd_lams


def _select_diverse_warm_states(spins, objs, lams, *, count: int):
    spins = np.asarray(spins, dtype=np.int8)
    objs = np.asarray(objs, dtype=np.float64)
    lams = np.asarray(lams, dtype=np.int64)
    if spins.size == 0:
        return spins, lams
    m = int(objs.shape[0])
    k = int(objs.shape[1])
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    scaled = (objs - mins[None, :]) / np.maximum(maxs - mins, 1e-12)
    cd = np.zeros((m,), dtype=np.float64)
    anchors = []
    for d in range(k):
        order = np.argsort(scaled[:, d])
        anchors.append(int(order[0]))
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        if m > 2:
            cd[order[1:-1]] += scaled[order[2:], d] - scaled[order[:-2], d]
    selected = []
    seen = set()
    for idx in anchors:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    for idx in np.argsort(-cd):
        val = int(idx)
        if val in seen:
            continue
        selected.append(val)
        seen.add(val)
        if len(selected) >= int(count):
            break
    while len(selected) < int(count):
        selected.append(selected[len(selected) % max(1, len(selected))])
    sel = np.asarray(selected[: int(count)], dtype=np.int64)
    return spins[sel], lams[sel]


def _crowding_priority(objs, counts=None):
    objs = np.asarray(objs, dtype=np.float64)
    m = int(objs.shape[0])
    if m == 0:
        return (
            np.zeros_like(objs, dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    k = int(objs.shape[1])
    counts_arr = (
        np.ones((m,), dtype=np.int64)
        if counts is None
        else np.asarray(counts, dtype=np.int64).reshape(-1)
    )
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    scaled = (objs - mins[None, :]) / np.maximum(maxs - mins, 1e-12)
    cd = np.zeros((m,), dtype=np.float64)
    anchors = []
    if m >= 2:
        for d in range(k):
            order = np.argsort(scaled[:, d])
            anchors.append(int(order[0]))
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if m > 2:
                cd[order[1:-1]] += scaled[order[2:], d] - scaled[order[:-2], d]
    else:
        anchors.append(0)
        cd[:] = np.inf
    anchors = np.asarray(list(dict.fromkeys(anchors)), dtype=np.int64)
    anchor_mask = np.zeros((m,), dtype=bool)
    anchor_mask[anchors] = True
    inf_mask = np.isinf(cd).astype(np.int8)
    cd_key = np.where(np.isfinite(cd), cd, 0.0)
    rest = np.lexsort(
        (
            np.arange(m, dtype=np.int64),
            -counts_arr.astype(np.int64, copy=False),
            -cd_key,
            -inf_mask,
        )
    )
    order = np.concatenate([anchors, rest[~anchor_mask[rest]]])
    return scaled, cd, anchors, order


def _select_frontier_cap_warm_states(
    spins,
    objs,
    lams,
    *,
    count: int,
    dist_thr: float,
    lambda_cap: int,
    counts=None,
):
    spins = np.asarray(spins, dtype=np.int8)
    objs = np.asarray(objs, dtype=np.float64)
    lams = np.asarray(lams, dtype=np.int64)
    if spins.size == 0 or objs.size == 0:
        return spins, lams

    count = int(count)
    lambda_cap = max(1, int(lambda_cap))
    m = int(objs.shape[0])
    scaled, _, _, order = _crowding_priority(objs, counts=counts)
    selected = []
    seen = np.zeros((m,), dtype=bool)
    lam_counts = np.zeros((max(int(lams.max()) + 1, 1),), dtype=np.int16)
    min_d2 = np.full((m,), np.inf, dtype=np.float64)

    def can_use(i: int) -> bool:
        return int(lam_counts[int(lams[i])]) < lambda_cap

    def add(i: int) -> None:
        ii = int(i)
        selected.append(ii)
        seen[ii] = True
        lam_counts[int(lams[ii])] += 1
        d = scaled - scaled[ii]
        d2 = np.einsum("ij,ij->i", d, d, optimize=False)
        min_d2[:] = np.minimum(min_d2, d2)
        min_d2[ii] = 0.0

    thr0 = max(float(dist_thr), 0.0)
    for thr2 in [thr0 * thr0, (thr0 * 0.3) ** 2, (thr0 * 0.1) ** 2, 0.0]:
        for idx in order:
            if len(selected) >= count:
                break
            ii = int(idx)
            if seen[ii] or not can_use(ii):
                continue
            if thr2 <= 0.0 or float(min_d2[ii]) >= thr2:
                add(ii)
        if len(selected) >= count:
            break

    if len(selected) < count:
        for idx in order:
            if len(selected) >= count:
                break
            ii = int(idx)
            if not seen[ii] and can_use(ii):
                add(ii)

    if not selected:
        selected = [0]
    while len(selected) < count:
        selected.append(selected[len(selected) % len(selected)])

    sel = np.asarray(selected[:count], dtype=np.int64)
    return spins[sel], lams[sel]


def _prefilter_selector_indices(objs, *, count: int, prefilter: int, counts=None):
    objs = np.asarray(objs, dtype=np.float64)
    m = int(objs.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=np.int64)
    limit = int(prefilter)
    if limit <= 0 or m <= limit:
        return np.arange(m, dtype=np.int64)
    limit = max(limit, int(count))
    _, _, anchors, order = _crowding_priority(objs, counts=counts)
    selected = []
    seen = set()
    for idx in anchors:
        val = int(idx)
        if val not in seen:
            selected.append(val)
            seen.add(val)
    k = int(objs.shape[1])
    for d in range(k):
        val = int(np.argmin(objs[:, d]))
        if val not in seen:
            selected.append(val)
            seen.add(val)
    for idx in order:
        val = int(idx)
        if val in seen:
            continue
        selected.append(val)
        seen.add(val)
        if len(selected) >= limit:
            break
    return np.asarray(selected[:limit], dtype=np.int64)


def _select_hv_greedy_warm_states(
    spins,
    objs,
    lams,
    *,
    count: int,
    dist_thr: float,
    lambda_cap: int,
    prefilter: int,
    counts=None,
):
    spins = np.asarray(spins, dtype=np.int8)
    objs = np.asarray(objs, dtype=np.float64)
    lams = np.asarray(lams, dtype=np.int64)
    if spins.size == 0 or objs.size == 0:
        return spins, lams

    count = int(count)
    lambda_cap = max(1, int(lambda_cap))
    keep = _prefilter_selector_indices(objs, count=count, prefilter=int(prefilter), counts=counts)
    cand_spins = spins[keep]
    cand_objs = objs[keep]
    cand_lams = lams[keep]
    m = int(cand_objs.shape[0])
    scaled, cd, anchors, _ = _crowding_priority(cand_objs, counts=None)
    selected = []
    seen = np.zeros((m,), dtype=bool)
    lam_counts = np.zeros((max(int(cand_lams.max()) + 1, 1),), dtype=np.int16)
    min_d2 = np.full((m,), np.inf, dtype=np.float64)
    hv_pool = np.zeros((0, int(cand_objs.shape[1])), dtype=np.float64)
    current_hv = 0.0

    def merge_nd_safe(pool, point):
        a = np.asarray(pool, dtype=np.float64)
        b = np.asarray(point, dtype=np.float64)
        merged = b if a.size == 0 else np.vstack([a, b])
        if int(merged.shape[0]) <= 1:
            return merged
        return merge_non_dominated_pool(a, b)

    def hv_safe(points):
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if int(arr.shape[0]) == 1:
            ref = np.full((int(arr.shape[1]),), 1.01, dtype=np.float64)
            if not np.all(arr[0] <= ref):
                return 0.0
            return float(np.prod(ref - arr[0]))
        return float(hypervolume_pygmo(arr))

    def can_use(i: int) -> bool:
        return int(lam_counts[int(cand_lams[i])]) < lambda_cap

    def add(i: int) -> None:
        nonlocal hv_pool, current_hv
        ii = int(i)
        selected.append(ii)
        seen[ii] = True
        lam_counts[int(cand_lams[ii])] += 1
        d = scaled - scaled[ii]
        d2 = np.einsum("ij,ij->i", d, d, optimize=False)
        min_d2[:] = np.minimum(min_d2, d2)
        min_d2[ii] = 0.0
        hv_pool = merge_nd_safe(hv_pool, cand_objs[ii : ii + 1])
        current_hv = hv_safe(hv_pool)

    for idx in anchors:
        if len(selected) >= count:
            break
        ii = int(idx)
        if not seen[ii] and can_use(ii):
            add(ii)

    thr0 = max(float(dist_thr), 0.0)
    for thr2 in [thr0 * thr0, (thr0 * 0.3) ** 2, (thr0 * 0.1) ** 2, 0.0]:
        while len(selected) < count:
            best_idx = None
            best_gain = -np.inf
            best_cd = -np.inf
            for idx in range(m):
                if seen[idx] or not can_use(idx):
                    continue
                if thr2 > 0.0 and float(min_d2[idx]) < thr2:
                    continue
                merged = merge_nd_safe(hv_pool, cand_objs[idx : idx + 1])
                hv = hv_safe(merged)
                gain = float(hv - current_hv)
                tie_cd = float(cd[idx]) if np.isfinite(cd[idx]) else np.inf
                if gain > best_gain + 1e-18 or (
                    abs(gain - best_gain) <= 1e-18 and tie_cd > best_cd
                ):
                    best_idx = int(idx)
                    best_gain = gain
                    best_cd = tie_cd
            if best_idx is None:
                break
            add(best_idx)
        if len(selected) >= count:
            break

    if len(selected) < count:
        _, _, _, order = _crowding_priority(cand_objs, counts=None)
        for idx in order:
            if len(selected) >= count:
                break
            ii = int(idx)
            if not seen[ii] and can_use(ii):
                add(ii)

    if not selected:
        selected = [0]
    while len(selected) < count:
        selected.append(selected[len(selected) % len(selected)])

    sel = np.asarray(selected[:count], dtype=np.int64)
    return cand_spins[sel], cand_lams[sel]


def _warm_bits_from_spins(warm_spins):
    return [np.where(z > 0, 0, 1).astype(np.int8) for z in np.asarray(warm_spins, dtype=np.int8)]


def _materialize_warm_selection(warm_spins, warm_lams, *, count: int):
    warm_lams = np.asarray(warm_lams, dtype=np.int64)
    if warm_lams.size == 0:
        return [None] * int(count), np.arange(int(count), dtype=np.int64)
    return _warm_bits_from_spins(warm_spins), warm_lams


def _lambda_support_counts(lams):
    lams = np.asarray(lams, dtype=np.int64)
    if lams.size == 0:
        return np.zeros((0,), dtype=np.int64)
    counts = np.bincount(lams, minlength=int(lams.max()) + 1)
    return counts[lams].astype(np.int64, copy=False)


def _walk_json_values(obj, key: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk_json_values(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json_values(item, key)


def _coerce_lambda_ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, float):
        return [int(value)] if float(value).is_integer() else []
    if isinstance(value, str):
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, dict):
        out = []
        for key in ("lambda_id", "best_lambda_id", "id"):
            out.extend(_coerce_lambda_ids(value.get(key)))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_coerce_lambda_ids(item))
        return out
    return []


def _load_guidance_lambda_ids(guidance_json: str | Path | None, *, pool_size: int) -> np.ndarray:
    if guidance_json is None or str(guidance_json).strip() == "":
        raise ValueError(
            "--guidance-json is required for exact_guided_* strategies "
            "(expected recommended_lambda_ids or ranked_missing[*].best_lambda_id)."
        )
    path = Path(guidance_json)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"guidance JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid guidance JSON {path}: {exc}") from exc

    raw_ids: list[int] = []
    for recommended in _walk_json_values(data, "recommended_lambda_ids"):
        raw_ids.extend(_coerce_lambda_ids(recommended))
    for ranked in _walk_json_values(data, "ranked_missing"):
        if isinstance(ranked, list):
            for row in ranked:
                if isinstance(row, dict):
                    raw_ids.extend(_coerce_lambda_ids(row.get("best_lambda_id")))
    for best in _walk_json_values(data, "best_lambda_id"):
        raw_ids.extend(_coerce_lambda_ids(best))

    ids = []
    seen = set()
    for lam_id in raw_ids:
        val = int(lam_id)
        if 0 <= val < int(pool_size) and val not in seen:
            ids.append(val)
            seen.add(val)
    if not ids:
        raise ValueError(
            f"guidance JSON {path} did not contain any valid lambda ids in "
            "recommended_lambda_ids or ranked_missing[*].best_lambda_id"
        )
    return np.asarray(ids, dtype=np.int64)


def _fill_lambda_ids(preferred_ids, *, count: int, pool_size: int) -> np.ndarray:
    out = []
    seen = set()
    for lam_id in np.asarray(preferred_ids, dtype=np.int64):
        val = int(lam_id)
        if 0 <= val < int(pool_size) and val not in seen:
            out.append(val)
            seen.add(val)
        if len(out) >= int(count):
            break
    for val in range(int(pool_size)):
        if len(out) >= int(count):
            break
        if val not in seen:
            out.append(val)
            seen.add(val)
    return np.asarray(out[: int(count)], dtype=np.int64)


def _select_guided_warm_states(spins, objs, lams, guidance_ids, *, count: int):
    spins = np.asarray(spins, dtype=np.int8)
    objs = np.asarray(objs, dtype=np.float64)
    lams = np.asarray(lams, dtype=np.int64)
    if spins.size == 0 or objs.size == 0:
        return spins, lams

    rank = {int(lam_id): pos for pos, lam_id in enumerate(np.asarray(guidance_ids, dtype=np.int64))}
    _, _, anchors, order = _crowding_priority(objs)
    selected = []
    seen = set()

    def add(idx: int) -> None:
        val = int(idx)
        if val not in seen:
            selected.append(val)
            seen.add(val)

    for idx in anchors:
        if len(selected) >= int(count):
            break
        add(int(idx))

    guided_positions = [
        int(idx)
        for idx in order
        if int(lams[int(idx)]) in rank and int(idx) not in seen
    ]
    guided_positions.sort(key=lambda idx: (rank[int(lams[idx])], idx))
    for idx in guided_positions:
        if len(selected) >= int(count):
            break
        add(idx)

    for idx in order:
        if len(selected) >= int(count):
            break
        add(int(idx))

    if not selected:
        selected = [0]
    while len(selected) < int(count):
        selected.append(selected[len(selected) % len(selected)])

    sel = np.asarray(selected[: int(count)], dtype=np.int64)
    return spins[sel], lams[sel]


def _run_strategy(
    case: Path,
    strategy: str,
    seed: int,
    q_target: int,
    p_layers: int,
    warm_c: float,
    selector_dist_thr: float,
    selector_lambda_cap: int,
    selector_prefilter: int,
    guidance_json: str | None,
) -> float:
    if strategy.startswith("exact_guided_") and (guidance_json is None or str(guidance_json).strip() == ""):
        raise ValueError(f"{strategy} requires --guidance-json")
    problem = problem_from_npz(str(case))
    pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)
    table = load_transfer_params_csv(str(ROOT / "transfer_data.csv"), q_target=int(q_target), p_list=(int(p_layers),))
    betas, gammas = table[int(p_layers)]
    proj_j = np.einsum("lk,km->lm", pool, problem.weights, optimize=False).astype(np.float64, copy=False)
    proj_h = np.einsum("lk,kn->ln", pool, problem.h, optimize=False).astype(np.float64, copy=False)
    base = baseline_hv(case, problem)
    t0 = time.time()

    if strategy == "first100_1000":
        spins, *_ = _sample_round(problem, np.arange(100), 1000, proj_j, proj_h, betas, gammas, seed=seed)
    elif strategy == "uniform_100k":
        circ = Circuit()
        for q in range(int(problem.n)):
            circ += H.on(q)
        if hasattr(circ, "measure_all"):
            circ.measure_all()
        sim = Simulator("mqvector", int(problem.n), seed=int(seed))
        res = sim.sampling(circ, shots=100000, seed=int(seed))
        spins = sampling_result_to_spins(res, int(problem.n), 100000)
    elif strategy == "pool1000_100":
        spins, *_ = _sample_round(problem, np.arange(1000), 100, proj_j, proj_h, betas, gammas, seed=seed)
    elif strategy == "pool500_200":
        spins, *_ = _sample_round(problem, np.arange(500), 200, proj_j, proj_h, betas, gammas, seed=seed)
    elif strategy == "exact_guided_lambda_500":
        guidance_ids = _load_guidance_lambda_ids(guidance_json, pool_size=int(pool.shape[0]))
        ids = _fill_lambda_ids(guidance_ids, count=500, pool_size=int(pool.shape[0]))
        spins, *_ = _sample_round(problem, ids, 200, proj_j, proj_h, betas, gammas, seed=seed)
    elif strategy == "pool250_400":
        spins, *_ = _sample_round(problem, np.arange(250), 400, proj_j, proj_h, betas, gammas, seed=seed)
    elif strategy == "baseline50_pool500_100":
        s0, *_ = _sample_round(problem, np.arange(100), 500, proj_j, proj_h, betas, gammas, seed=seed)
        s1, *_ = _sample_round(problem, np.arange(500), 100, proj_j, proj_h, betas, gammas, seed=seed + 10000)
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "baseline50_pool500_warm100":
        s0, u0, c0, l0 = _sample_round(problem, np.arange(100), 500, proj_j, proj_h, betas, gammas, seed=seed)
        ids = np.arange(500)
        s1, u1, c1, l1 = _sample_round(problem, ids, 50, proj_j, proj_h, betas, gammas, seed=seed + 10000)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0 + u1, c0 + c1, list(l0) + list(l1))
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s2, *_ = _sample_round(
            problem,
            ids,
            50,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 20000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1, s2]).astype(np.int8)
    elif strategy == "two_round_500_100":
        ids = np.arange(500)
        s0, u0, c0, l0 = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s1, *_ = _sample_round(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "two_round_500_then_500":
        ids0 = np.arange(500)
        ids1 = np.arange(500, 1000)
        s0, u0, c0, l0 = _sample_round(problem, ids0, 100, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        warm = _make_warm_for_ids(objs, fspins, ids1, pool, int(problem.n))
        s1, *_ = _sample_round(
            problem,
            ids1,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "frontier_gap_500":
        ids0 = np.arange(500)
        s0, u0, c0, l0 = _sample_round(problem, ids0, 100, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        ids1 = _frontier_gap_ids(objs, pool, count=500)
        warm = _make_warm_for_ids(objs, fspins, ids1, pool, int(problem.n))
        s1, *_ = _sample_round(
            problem,
            ids1,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "greedy_select_500":
        scout_ids = np.arange(1000)
        s0, u0, c0, l0 = _sample_round(problem, scout_ids, 20, proj_j, proj_h, betas, gammas, seed=seed)
        ids = _greedy_ids_from_sampled_blocks(problem, u0, scout_ids, count=400)
        s1, u1, c1, l1 = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed + 10000)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0 + u1, c0 + c1, list(l0) + list(l1))
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s2, *_ = _sample_round(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 20000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        # 20k scout + 40k focused + 40k focused warm = 100k.
        spins = np.vstack([s0, s1, s2]).astype(np.int8)
    elif strategy == "two_round_1000_split_no_warm":
        ids0 = np.arange(500)
        ids1 = np.arange(500, 1000)
        s0, *_ = _sample_round(problem, ids0, 100, proj_j, proj_h, betas, gammas, seed=seed)
        s1, *_ = _sample_round(problem, ids1, 100, proj_j, proj_h, betas, gammas, seed=seed + 10000)
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "local_ws_500_100":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        warm = _local_descent_warm_for_ids(problem, ids, proj_j, proj_h, seed=seed + 50000, restarts=8)
        s1, *_ = _sample_round(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "local_ws_1000_50":
        ids = np.arange(1000)
        s0, *_ = _sample_round(problem, ids, 50, proj_j, proj_h, betas, gammas, seed=seed)
        warm = _local_descent_warm_for_ids(problem, ids, proj_j, proj_h, seed=seed + 50000, restarts=8)
        s1, *_ = _sample_round(
            problem,
            ids,
            50,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "local_only_500_200":
        ids = np.arange(500)
        warm = _local_descent_warm_for_ids(problem, ids, proj_j, proj_h, seed=seed + 50000, restarts=8)
        spins, *_ = _sample_round(
            problem,
            ids,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            warm_bits=warm,
            warm_c=warm_c,
        )
    elif strategy == "local_only_1000_100":
        ids = np.arange(1000)
        warm = _local_descent_warm_for_ids(problem, ids, proj_j, proj_h, seed=seed + 50000, restarts=8)
        spins, *_ = _sample_round(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            warm_bits=warm,
            warm_c=warm_c,
        )
    elif strategy == "mo_local_warm_500_200":
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=500)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        spins, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            warm_bits=warm,
            warm_c=warm_c,
        )
    elif strategy == "hybrid_broad_molocal":
        ids = np.arange(500)
        # 50k broad QAOA coverage.
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        # 50k multi-objective local-search warm-start quantum sampling.
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=250)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "exact_guided_broad_warm":
        guidance_ids = _load_guidance_lambda_ids(guidance_json, pool_size=int(pool.shape[0]))
        broad_ids = np.arange(500)
        s0, *_ = _sample_round(problem, broad_ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_guided_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            guidance_ids,
            count=250,
        )
        warm, warm_lams = _materialize_warm_selection(warm_spins, warm_lams, count=250)
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad_molocal_frontier_cap":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        cand_counts = _lambda_support_counts(cand_lams)
        warm_spins, warm_lams = _select_frontier_cap_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=250,
            dist_thr=selector_dist_thr,
            lambda_cap=selector_lambda_cap,
            counts=cand_counts,
        )
        warm, warm_lams = _materialize_warm_selection(warm_spins, warm_lams, count=250)
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad_molocal_frontier_cap500_100":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        cand_counts = _lambda_support_counts(cand_lams)
        warm_spins, warm_lams = _select_frontier_cap_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=500,
            dist_thr=selector_dist_thr,
            lambda_cap=selector_lambda_cap,
            counts=cand_counts,
        )
        warm, warm_lams = _materialize_warm_selection(warm_spins, warm_lams, count=500)
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad_molocal_hvgreedy":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        cand_counts = _lambda_support_counts(cand_lams)
        warm_spins, warm_lams = _select_hv_greedy_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=250,
            dist_thr=selector_dist_thr,
            lambda_cap=selector_lambda_cap,
            prefilter=selector_prefilter,
            counts=cand_counts,
        )
        warm, warm_lams = _materialize_warm_selection(warm_spins, warm_lams, count=250)
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad_molocal_hvgreedy500_100":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        cand_counts = _lambda_support_counts(cand_lams)
        warm_spins, warm_lams = _select_hv_greedy_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=500,
            dist_thr=selector_dist_thr,
            lambda_cap=selector_lambda_cap,
            prefilter=selector_prefilter,
            counts=cand_counts,
        )
        warm, warm_lams = _materialize_warm_selection(warm_spins, warm_lams, count=500)
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad_molocal500_100":
        ids = np.arange(500)
        # 50k broad QAOA coverage + 50k warm-start coverage across more
        # distinct local-frontier states.
        s0, *_ = _sample_round(problem, ids, 100, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=500)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad70_molocal30":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 140, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=150)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad1000_molocal50":
        ids = np.arange(1000)
        s0, *_ = _sample_round(problem, ids, 50, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=250)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "hybrid_broad60_molocal40":
        ids = np.arange(500)
        s0, *_ = _sample_round(problem, ids, 120, proj_j, proj_h, betas, gammas, seed=seed)
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=200)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        s1, *_ = _sample_round(
            problem,
            warm_lams,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "mo_local_warm_1000_100":
        cand_spins, cand_objs, cand_lams = _multiobjective_local_candidates(
            problem, pool, proj_j, proj_h, seed=seed + 70000, restarts=6
        )
        warm_spins, warm_lams = _select_diverse_warm_states(cand_spins, cand_objs, cand_lams, count=1000)
        warm = [np.where(z > 0, 0, 1).astype(np.int8) for z in warm_spins]
        spins, *_ = _sample_round(
            problem,
            warm_lams,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            warm_bits=warm,
            warm_c=warm_c,
        )
    elif strategy == "param_500_4x50":
        ids = np.arange(500)
        spins = _sample_param_portfolio(
            problem,
            ids,
            50,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            gamma_scales=[0.7, 0.9, 1.1, 1.3],
        )
    elif strategy == "param_1000_2x50":
        ids = np.arange(1000)
        spins = _sample_param_portfolio(
            problem,
            ids,
            50,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            gamma_scales=[0.85, 1.15],
        )
    elif strategy == "param_250_4x100":
        ids = np.arange(250)
        spins = _sample_param_portfolio(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed,
            gamma_scales=[0.7, 0.9, 1.1, 1.3],
        )
    elif strategy == "two_round_1000_50":
        ids = np.arange(1000)
        s0, u0, c0, l0 = _sample_round(problem, ids, 50, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s1, *_ = _sample_round(
            problem,
            ids,
            50,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "two_round_250_200":
        ids = np.arange(250)
        s0, u0, c0, l0 = _sample_round(problem, ids, 200, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s1, *_ = _sample_round(
            problem,
            ids,
            200,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1]).astype(np.int8)
    elif strategy == "three_round_250_150_100_150":
        ids = np.arange(250)
        s0, u0, c0, l0 = _sample_round(problem, ids, 150, proj_j, proj_h, betas, gammas, seed=seed)
        objs, fspins, _, _, _ = _frontier_warm_bits(problem, u0, c0, l0)
        warm = _make_warm_for_ids(objs, fspins, ids, pool, int(problem.n))
        s1, u1, c1, l1 = _sample_round(
            problem,
            ids,
            100,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 10000,
            warm_bits=warm,
            warm_c=warm_c,
        )
        objs2, fspins2, _, _, _ = _frontier_warm_bits(problem, u0 + u1, c0 + c1, list(l0) + list(l1))
        warm2 = _make_warm_for_ids(objs2, fspins2, ids, pool, int(problem.n))
        s2, *_ = _sample_round(
            problem,
            ids,
            150,
            proj_j,
            proj_h,
            betas,
            gammas,
            seed=seed + 20000,
            warm_bits=warm2,
            warm_c=warm_c,
        )
        spins = np.vstack([s0, s1, s2]).astype(np.int8)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    if int(spins.shape[0]) != 100000:
        raise RuntimeError(f"{strategy} produced {spins.shape[0]} rows, expected 100000")

    hv = _hv_from_spins_safe(problem, spins)
    gain = max(float(hv - base), 0.0)
    score = float(gain * 100000)
    print(
        f"{case.name},{strategy},q={q_target},p={p_layers},warm_c={warm_c},rows={spins.shape[0]},hv={hv:.12f},"
        f"base={base:.12f},gain={gain:.12f},score={score:.6f},"
        f"elapsed={time.time() - t0:.3f}",
        flush=True,
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="data/public/k5_grid4x5_00.npz")
    parser.add_argument("--glob", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--strategy", action="append", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--q-target", type=int, default=2)
    parser.add_argument("--p-layers", type=int, default=3)
    parser.add_argument("--warm-c", type=float, default=0.1)
    parser.add_argument("--selector-dist-thr", type=float, default=1e-4)
    parser.add_argument("--selector-lambda-cap", type=int, default=2)
    parser.add_argument("--selector-prefilter", type=int, default=800)
    parser.add_argument("--guidance-json", default="")
    args = parser.parse_args()
    if args.glob:
        cases = sorted(ROOT.glob(args.glob))
        if args.max_cases > 0:
            cases = cases[: args.max_cases]
    else:
        cases = [ROOT / args.case]
    scores: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for strategy in args.strategy:
            score = _run_strategy(
                case,
                strategy,
                args.seed,
                args.q_target,
                args.p_layers,
                args.warm_c,
                args.selector_dist_thr,
                args.selector_lambda_cap,
                args.selector_prefilter,
                args.guidance_json,
            )
            scores[strategy].append(float(score))
    if len(cases) > 1:
        print("strategy,mean_score,n_cases")
        for strategy in args.strategy:
            vals = scores[strategy]
            print(f"{strategy},{float(np.mean(vals)):.6f},{len(vals)}", flush=True)


if __name__ == "__main__":
    main()
