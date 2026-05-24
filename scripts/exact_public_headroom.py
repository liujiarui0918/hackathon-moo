from __future__ import annotations

import argparse
import csv
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

from utils import (  # noqa: E402
    HV_REF,
    hypervolume_pygmo,
    normalize_energies,
    problem_from_npz,
    _state_index_block_to_spins,
)


SUMMARY_FIELDS = [
    "case",
    "hv_base",
    "hv_solver_old",
    "hv_exact",
    "score_old",
    "score_max_case",
    "remaining_case_score",
    "captured_fraction",
    "exact_nd_count",
    "frontier_npz",
    "elapsed_s",
    "local_candidate_count",
    "chunk_size",
]


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)


def _safe_non_dominated_indices(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if int(arr.shape[0]) <= 1:
        return np.zeros((1,), dtype=np.int64)

    return _blocked_non_dominated_indices(arr)


def _incremental_front_indices(arr: np.ndarray) -> np.ndarray:
    order = np.lexsort(arr[:, ::-1].T)
    capacity = min(max(16, int(arr.shape[0])), 4096)
    front_indices = np.empty((capacity,), dtype=np.int64)
    front_objs = np.empty((capacity, arr.shape[1]), dtype=np.float64)
    front_n = 0

    for idx in order:
        x = arr[int(idx)]
        active = front_objs[:front_n]
        if front_n:
            dominated_by_front = np.all(active <= x[None, :], axis=1) & np.any(
                active < x[None, :], axis=1
            )
            if bool(np.any(dominated_by_front)):
                continue

            duplicate = np.all(active == x[None, :], axis=1)
            if bool(np.any(duplicate)):
                continue

            dominated_by_x = np.all(x[None, :] <= active, axis=1) & np.any(
                x[None, :] < active, axis=1
            )
            if bool(np.any(dominated_by_x)):
                keep = ~dominated_by_x
                kept_n = int(np.count_nonzero(keep))
                front_indices[:kept_n] = front_indices[:front_n][keep]
                front_objs[:kept_n] = active[keep]
                front_n = kept_n

        if front_n >= int(front_indices.shape[0]):
            new_capacity = max(front_n + 1, int(front_indices.shape[0]) * 2)
            new_indices = np.empty((new_capacity,), dtype=np.int64)
            new_objs = np.empty((new_capacity, arr.shape[1]), dtype=np.float64)
            new_indices[:front_n] = front_indices[:front_n]
            new_objs[:front_n] = front_objs[:front_n]
            front_indices = new_indices
            front_objs = new_objs

        front_indices[front_n] = int(idx)
        front_objs[front_n] = x
        front_n += 1

    return front_indices[:front_n].copy()


def _dominated_by_pool_mask(
    points: np.ndarray,
    pool: np.ndarray,
    *,
    point_block_size: int = 512,
    pool_block_size: int = 256,
) -> np.ndarray:
    if points.size == 0 or pool.size == 0:
        return np.zeros((int(points.shape[0]),), dtype=bool)
    dominated = np.zeros((int(points.shape[0]),), dtype=bool)
    for point_start in range(0, int(points.shape[0]), int(point_block_size)):
        point_slice = slice(point_start, point_start + int(point_block_size))
        point_chunk = points[point_slice]
        point_dominated = np.zeros((int(point_chunk.shape[0]),), dtype=bool)
        for pool_start in range(0, int(pool.shape[0]), int(pool_block_size)):
            pool_chunk = pool[pool_start : pool_start + int(pool_block_size)]
            le = np.all(pool_chunk[:, None, :] <= point_chunk[None, :, :], axis=2)
            lt = np.any(pool_chunk[:, None, :] < point_chunk[None, :, :], axis=2)
            point_dominated |= np.any(le & lt, axis=0)
            if bool(np.all(point_dominated)):
                break
        dominated[point_slice] = point_dominated
    return dominated


def _blocked_non_dominated_indices(arr: np.ndarray, *, block_size: int = 4096) -> np.ndarray:
    # pygmo's sorter is fast when it works, but this Windows contest env can
    # abort on real case arrays. This blocked first-front filter is exact while
    # keeping Python loops over block/front candidates instead of every point.
    order = np.lexsort(arr[:, ::-1].T)
    front_indices = np.zeros((0,), dtype=np.int64)
    front_objs = np.zeros((0, arr.shape[1]), dtype=np.float64)

    for start in range(0, int(order.shape[0]), int(block_size)):
        block_indices = order[start : start + int(block_size)]
        block_objs = arr[block_indices]

        dominated = _dominated_by_pool_mask(block_objs, front_objs)
        if bool(np.all(dominated)):
            continue

        survivor_indices = block_indices[~dominated]
        survivor_objs = block_objs[~dominated]
        local_idx = _incremental_front_indices(survivor_objs)
        if local_idx.size == 0:
            continue

        local_indices = survivor_indices[local_idx]
        local_objs = survivor_objs[local_idx]
        if front_objs.size == 0:
            front_indices = local_indices.astype(np.int64, copy=True)
            front_objs = local_objs.astype(np.float64, copy=True)
            continue

        merged_indices = np.concatenate([front_indices, local_indices.astype(np.int64, copy=False)])
        merged_objs = np.vstack([front_objs, local_objs])
        merged_front = _incremental_front_indices(merged_objs)
        front_indices = merged_indices[merged_front]
        front_objs = merged_objs[merged_front]

    return front_indices


def _safe_hypervolume(objs: np.ndarray, ref: float | np.ndarray = HV_REF) -> float:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if np.isscalar(ref):
        ref_vec = np.full(arr.shape[1], float(ref), dtype=np.float64)
    else:
        ref_vec = np.asarray(ref, dtype=np.float64).reshape(-1)
        if ref_vec.shape[0] != arr.shape[1]:
            raise ValueError(f"ref dimension mismatch: got {ref_vec.shape[0]}, expect {arr.shape[1]}")

    mask = np.all(arr <= ref_vec[None, :], axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return 0.0
    arr = np.unique(arr, axis=0)
    nd = arr[_safe_non_dominated_indices(arr)]
    if nd.size == 0:
        return 0.0
    if int(nd.shape[0]) == 1:
        width = np.maximum(ref_vec - nd[0], 0.0)
        return float(np.prod(width))
    return float(hypervolume_pygmo(nd, ref=ref_vec))


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _objective_extrema_safe(problem: Any, *, chunk_size: int, case_name: str) -> tuple[np.ndarray, np.ndarray]:
    n = int(problem.n)
    k = int(problem.k)
    total = 1 << n
    step = max(1, int(chunk_size))
    lower = np.full((k,), np.inf, dtype=np.float64)
    upper = np.full((k,), -np.inf, dtype=np.float64)
    t0 = time.time()

    for chunk_id, start in enumerate(range(0, total, step), start=1):
        size = min(step, total - start)
        spins = _state_index_block_to_spins(start, size, n)
        energies = _energy_batch_safe(spins, problem.edges, problem.weights, problem.h)
        lower = np.minimum(lower, energies.min(axis=0))
        upper = np.maximum(upper, energies.max(axis=0))
        _print_json(
            {
                "event": "extrema_chunk_done",
                "case": case_name,
                "chunk": chunk_id,
                "done_states": int(start + size),
                "total_states": int(total),
                "elapsed_s": round(float(time.time() - t0), 3),
            }
        )
    return lower, upper


def _unique_objective_rows(
    state_indices: np.ndarray,
    objectives_norm: np.ndarray,
    energies_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if objectives_norm.size == 0:
        return state_indices, objectives_norm, energies_raw
    unique_objs, first_idx = np.unique(objectives_norm, axis=0, return_index=True)
    return (
        np.asarray(state_indices[first_idx], dtype=np.uint64),
        np.asarray(unique_objs, dtype=np.float64),
        np.asarray(energies_raw[first_idx], dtype=np.float64),
    )


def _case_path_from_arg(raw: str) -> Path:
    text = str(raw)
    if text.endswith(".npz") or "/" in text or "\\" in text:
        path = Path(text)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(text):02d}.npz"


def _resolve_case_paths(cases: list[str] | None) -> list[Path]:
    if cases:
        return [_case_path_from_arg(raw) for raw in cases]
    return sorted((ROOT / "data" / "public").glob("k5_grid4x5_*.npz"))


def _load_score_rows(score_json: str | os.PathLike[str] | None) -> dict[str, dict[str, Any]]:
    if not score_json:
        return {}
    score_path = Path(score_json)
    if not score_path.is_absolute():
        score_path = ROOT / score_path
    if not score_path.exists():
        _print_json({"event": "score_json_missing", "path": _project_path(score_path)})
        return {}

    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("k5_rows", payload.get("rows", []))
    else:
        rows = []
    return {str(row["case"]): row for row in rows if isinstance(row, dict) and "case" in row}


def _csv_path_from_out(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_suffix(".csv")
    return Path(str(out_path) + ".csv")


def exact_frontier_for_case(
    case_path: Path,
    *,
    chunk_size: int,
    local_block_size: int,
    frontier_dir: Path,
) -> dict[str, Any]:
    problem = problem_from_npz(str(case_path))
    n = int(problem.n)
    k = int(problem.k)
    if n > 24:
        raise ValueError(f"Refusing exact enumeration for n={n}")

    step = max(1, int(chunk_size))
    local_step = max(1, min(int(local_block_size), step))
    total = 1 << n
    front_states = np.zeros((0,), dtype=np.uint64)
    front_objs = np.zeros((0, k), dtype=np.float64)
    front_energies = np.zeros((0, k), dtype=np.float64)
    local_candidate_count = 0
    t0 = time.time()

    _print_json(
        {
            "event": "case_start",
            "case": case_path.name,
            "n": n,
            "k": k,
            "total_states": total,
            "chunk_size": step,
            "local_block_size": local_step,
        }
    )

    lower, upper = _objective_extrema_safe(problem, chunk_size=step, case_name=case_path.name)

    for chunk_id, start in enumerate(range(0, total, step), start=1):
        size = min(step, total - start)
        spins = _state_index_block_to_spins(start, size, n)
        energies = _energy_batch_safe(spins, problem.edges, problem.weights, problem.h)
        objs = normalize_energies(energies, lower, upper)
        states = np.arange(start, start + size, dtype=np.uint64)
        chunk_added = 0

        for off in range(0, size, local_step):
            end = min(off + local_step, size)
            block_objs = np.asarray(objs[off:end], dtype=np.float64)
            block_states = np.asarray(states[off:end], dtype=np.uint64)
            block_energies = np.asarray(energies[off:end], dtype=np.float64)

            if front_objs.size:
                keep = ~_dominated_by_pool_mask(block_objs, front_objs)
                if not bool(np.any(keep)):
                    continue
                block_objs = block_objs[keep]
                block_states = block_states[keep]
                block_energies = block_energies[keep]

            local_idx = _safe_non_dominated_indices(block_objs)
            if local_idx.size == 0:
                continue

            cand_objs = np.asarray(block_objs[local_idx], dtype=np.float64)
            cand_states = np.asarray(block_states[local_idx], dtype=np.uint64)
            cand_energies = np.asarray(block_energies[local_idx], dtype=np.float64)

            if front_objs.size:
                keep_front = ~_dominated_by_pool_mask(front_objs, cand_objs)
                front_objs = front_objs[keep_front]
                front_states = front_states[keep_front]
                front_energies = front_energies[keep_front]

            front_objs = cand_objs if front_objs.size == 0 else np.vstack([front_objs, cand_objs])
            front_states = (
                cand_states if front_states.size == 0 else np.concatenate([front_states, cand_states])
            )
            front_energies = (
                cand_energies if front_energies.size == 0 else np.vstack([front_energies, cand_energies])
            )
            chunk_added += int(cand_objs.shape[0])
            local_candidate_count += int(cand_objs.shape[0])

        front_states, front_objs, front_energies = _unique_objective_rows(
            front_states,
            front_objs,
            front_energies,
        )

        _print_json(
            {
                "event": "chunk_done",
                "case": case_path.name,
                "chunk": chunk_id,
                "done_states": int(start + size),
                "total_states": int(total),
                "chunk_added": int(chunk_added),
                "front_count": int(front_objs.shape[0]),
                "candidate_count": int(local_candidate_count),
                "elapsed_s": round(float(time.time() - t0), 3),
            }
        )

    final_idx = _safe_non_dominated_indices(front_objs)
    nd_states = front_states[final_idx]
    nd_objs = front_objs[final_idx]
    nd_energies = front_energies[final_idx]
    nd_states, nd_objs, nd_energies = _unique_objective_rows(nd_states, nd_objs, nd_energies)

    exact_hv = _safe_hypervolume(nd_objs, ref=HV_REF)
    elapsed_s = float(time.time() - t0)

    frontier_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = frontier_dir / case_path.name
    np.savez_compressed(
        frontier_path,
        state_indices=nd_states,
        objectives_norm=nd_objs,
        energies_raw=nd_energies,
        hv_exact=np.asarray(exact_hv, dtype=np.float64),
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
    )

    return {
        "case": case_path.name,
        "hv_exact": exact_hv,
        "exact_nd_count": int(nd_objs.shape[0]),
        "frontier_npz": _project_path(frontier_path),
        "elapsed_s": elapsed_s,
        "local_candidate_count": int(local_candidate_count),
        "chunk_size": int(step),
        "local_block_size": int(local_step),
    }


def add_score_headroom(row: dict[str, Any], old: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(row)
    row.update(
        {
            "hv_base": None,
            "hv_solver_old": None,
            "score_old": None,
            "score_max_case": None,
            "remaining_case_score": None,
            "captured_fraction": None,
        }
    )
    if old is None:
        return row

    hv_base = float(old["hv_base"])
    hv_solver = float(old["hv_solver"])
    score_old = float(old.get("score_case", max(hv_solver - hv_base, 0.0) * 100000.0))
    score_max = max(float(row["hv_exact"]) - hv_base, 0.0) * 100000.0
    remaining = max(float(row["hv_exact"]) - hv_solver, 0.0) * 100000.0
    captured = (score_old / score_max) if score_max > 0 else None
    row.update(
        {
            "hv_base": hv_base,
            "hv_solver_old": hv_solver,
            "score_old": score_old,
            "score_max_case": score_max,
            "remaining_case_score": remaining,
            "captured_fraction": captured,
        }
    )
    return row


def write_outputs(rows: list[dict[str, Any]], out_path: Path) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    csv_path = _csv_path_from_out(out_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({key for row in rows for key in row} - set(SUMMARY_FIELDS))
    fieldnames = SUMMARY_FIELDS + extra_fields
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast exact public-case HV/headroom by enumerating all states.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case suffixes such as 00 08, or explicit paths.")
    parser.add_argument("--chunk-size", type=int, default=131072)
    parser.add_argument("--local-block-size", type=int, default=4096)
    parser.add_argument("--out", default="results/exact_public_headroom.json")
    parser.add_argument("--frontier-dir", default="results/exact_frontiers")
    parser.add_argument("--score-json", default="results/seed_schedule_with01_public_default.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    frontier_dir = Path(args.frontier_dir)
    if not frontier_dir.is_absolute():
        frontier_dir = ROOT / frontier_dir

    score_rows = _load_score_rows(args.score_json)
    case_paths = _resolve_case_paths(args.cases)

    rows: list[dict[str, Any]] = []
    for case_path in case_paths:
        exact_row = exact_frontier_for_case(
            case_path,
            chunk_size=int(args.chunk_size),
            local_block_size=int(args.local_block_size),
            frontier_dir=frontier_dir,
        )
        row = add_score_headroom(exact_row, score_rows.get(case_path.name))
        rows.append(row)
        _print_json(row)

    json_path, csv_path = write_outputs(rows, out_path)
    _print_json({"event": "wrote_summary", "json": _project_path(json_path), "csv": _project_path(csv_path)})


if __name__ == "__main__":
    main()
