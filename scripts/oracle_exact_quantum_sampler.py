from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
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

HV_REF = 1.01


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


@lru_cache(maxsize=1)
def _load_mindquantum() -> tuple[Any, Any, Any]:
    try:
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import X
        from mindquantum.simulator import Simulator
    except Exception as exc:  # pragma: no cover - depends on local contest env
        raise RuntimeError(
            "mindquantum is required for oracle sampling, but it is not importable "
            f"in this Python environment: {exc}"
        ) from exc
    return Circuit, X, Simulator


def _problem_from_npz(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "name": str(data["name"]) if "name" in data else path.stem,
        "a": int(data["a"]),
        "b": int(data["b"]),
        "k": int(data["k"]),
        "edges": np.asarray(data["edges"], dtype=np.int32),
        "weights": np.asarray(data["weights"], dtype=np.float64),
        "h": np.asarray(data["h"], dtype=np.float64),
    }


def _state_indices_to_spins(state_indices: np.ndarray, n: int) -> np.ndarray:
    idx = np.asarray(state_indices, dtype=np.uint64).reshape(-1, 1)
    shifts = np.arange(int(n), dtype=np.uint64).reshape(1, -1)
    bits = (idx >> shifts) & np.uint64(1)
    return np.where(bits == 0, 1, -1).astype(np.int8, copy=False)


def _energy_batch_safe(
    spins: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _normalize_energies(
    energies: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    lo = np.asarray(lower_bounds, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(upper_bounds, dtype=np.float64).reshape(1, -1)
    denom = np.maximum(hi - lo, 1e-12)
    return (np.asarray(energies, dtype=np.float64) - lo) / denom


def _sampling_result_to_unique_spins(res: object, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(res, dict):
        counts = res
    else:
        counts = getattr(res, "data", None)
    if not isinstance(counts, dict):
        raise TypeError(f"Unsupported sampling result type: {type(res)}")

    rows: list[np.ndarray] = []
    reps: list[int] = []
    n_qubits = int(n_qubits)
    for bitstring, cnt in counts.items():
        rep = int(cnt)
        if rep <= 0:
            continue
        b = str(bitstring).strip()
        if b.startswith("0b"):
            b = b[2:]
        b = b.zfill(n_qubits)[-n_qubits:]
        row = np.fromiter((1 if c == "0" else -1 for c in reversed(b)), dtype=np.int8)
        rows.append(row)
        reps.append(rep)

    if not rows:
        raise ValueError("MindQuantum sampling produced zero valid states.")
    return np.asarray(np.vstack(rows), dtype=np.int8), np.asarray(reps, dtype=np.int64)


def _ensure_measure_all(circ: Any, n_qubits: int) -> Any:
    from mindquantum.core.gates import Measure

    # `Circuit.measure_all()` is version-sensitive for an otherwise empty
    # basis-preparation circuit, so add measurements explicitly.
    for q in range(int(n_qubits)):
        circ += Measure().on(q)
    return circ


def _basis_state_circuit(spin: np.ndarray, *, qubit_order: str) -> Any:
    Circuit, X, _Simulator = _load_mindquantum()
    n = int(spin.shape[0])
    circ = Circuit()
    for spin_idx, value in enumerate(np.asarray(spin, dtype=np.int8)):
        if int(value) >= 0:
            continue
        if qubit_order == "little":
            qubit = int(spin_idx)
        elif qubit_order == "reversed":
            qubit = int(n - 1 - spin_idx)
        else:
            raise ValueError(f"Unsupported qubit_order: {qubit_order}")
        circ += X.on(qubit)
    return _ensure_measure_all(circ, n)


def _sample_basis_state(
    sim: Any,
    spin: np.ndarray,
    *,
    shots: int,
    seed: int,
    qubit_order: str,
) -> np.ndarray:
    if int(shots) <= 0:
        return np.zeros((0, int(spin.shape[0])), dtype=np.int8)
    circ = _basis_state_circuit(spin, qubit_order=qubit_order)
    sim.reset()
    res = sim.sampling(circ, shots=int(shots), seed=int(seed))
    unique_spins, counts = _sampling_result_to_unique_spins(res, int(spin.shape[0]))
    out = np.repeat(unique_spins, counts.astype(np.int32), axis=0)
    if int(out.shape[0]) != int(shots):
        raise ValueError(f"Sampling row count mismatch: got {out.shape[0]}, expect {shots}")
    return np.asarray(out, dtype=np.int8)


def _choose_qubit_order(
    sim: Any,
    frontier_spins: np.ndarray,
    *,
    seed: int,
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    probe = np.asarray(frontier_spins[0], dtype=np.int8)
    mixed = np.where((frontier_spins == -1).any(axis=1) & (frontier_spins == 1).any(axis=1))[0]
    if mixed.size:
        probe = np.asarray(frontier_spins[int(mixed[0])], dtype=np.int8)

    for order in ("little", "reversed"):
        rows = _sample_basis_state(sim, probe, shots=8, seed=int(seed), qubit_order=order)
        if bool(np.all(rows == probe[None, :])):
            return order
    raise RuntimeError("Could not calibrate MindQuantum qubit order against the target basis state.")


def _sample_frontier_basis_states(
    frontier_spins: np.ndarray,
    *,
    sample_budget: int,
    seed: int,
    qubit_order: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(sample_budget) <= 0:
        raise ValueError("sample_budget must be positive.")
    if int(frontier_spins.shape[0]) == 0:
        raise ValueError("Exact frontier has no states.")
    if int(frontier_spins.shape[0]) > int(sample_budget):
        raise ValueError(
            "Exact frontier is larger than sample_budget; cannot include every frontier state once."
        )

    _Circuit, _X, Simulator = _load_mindquantum()
    sim = Simulator("mqvector", int(frontier_spins.shape[1]), seed=int(seed) % (2**23))
    actual_order = _choose_qubit_order(sim, frontier_spins, seed=int(seed), requested=qubit_order)

    n_front = int(frontier_spins.shape[0])
    base, rem = divmod(int(sample_budget), n_front)
    sample_spins = np.empty((int(sample_budget), int(frontier_spins.shape[1])), dtype=np.int8)
    cursor = 0
    mismatch_count = 0
    for i, spin in enumerate(frontier_spins):
        shots = int(base + (1 if i < rem else 0))
        rows = _sample_basis_state(
            sim,
            np.asarray(spin, dtype=np.int8),
            shots=shots,
            seed=int(seed) + 1009 * (i + 1),
            qubit_order=actual_order,
        )
        if rows.size and not bool(np.all(rows == spin[None, :])):
            mismatch_count += 1
        sample_spins[cursor : cursor + shots] = rows
        cursor += shots

    if cursor != int(sample_budget):
        raise RuntimeError(f"Internal fill error: cursor={cursor}, sample_budget={sample_budget}")
    if mismatch_count:
        raise RuntimeError(f"{mismatch_count} prepared basis states did not sample back as requested.")

    return sample_spins, {
        "qubit_order": actual_order,
        "base_repeats_per_frontier_state": int(base),
        "extra_one_shot_states": int(rem),
    }


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


def _safe_non_dominated_indices(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if int(arr.shape[0]) <= 1:
        return np.arange(int(arr.shape[0]), dtype=np.int64)
    return _blocked_non_dominated_indices(arr)


def _lexsort_rows(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 2 or int(x.shape[0]) <= 1:
        return x
    order = np.lexsort(x[:, ::-1].T)
    return x[order]


def _objectives_match(a: np.ndarray, b: np.ndarray, *, atol: float) -> bool:
    aa = _lexsort_rows(np.unique(np.asarray(a, dtype=np.float64), axis=0))
    bb = _lexsort_rows(np.unique(np.asarray(b, dtype=np.float64), axis=0))
    return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=float(atol), rtol=0.0))


def _safe_hypervolume(
    nd_objs: np.ndarray,
    *,
    ref: float,
    certificate_objs: np.ndarray,
    certificate_hv: float,
    atol: float,
) -> tuple[float, str]:
    arr = np.asarray(nd_objs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, "empty"
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    ref_vec = np.full(int(arr.shape[1]), float(ref), dtype=np.float64)
    arr = arr[np.all(arr <= ref_vec[None, :], axis=1)]
    arr = np.unique(arr, axis=0)
    arr = arr[_safe_non_dominated_indices(arr)]
    if arr.size == 0:
        return 0.0, "empty_after_ref_filter"

    try:
        import pygmo as pg

        hv = float(pg.hypervolume(arr).compute(ref_vec))
        return hv, "pygmo"
    except Exception as exc:
        if _objectives_match(arr, certificate_objs, atol=atol):
            return float(certificate_hv), f"frontier_certificate_match_no_pygmo:{type(exc).__name__}"
        raise RuntimeError(
            "pygmo is not available for direct HV computation, and sampled objectives "
            "do not exactly match the frontier certificate."
        ) from exc


def _evaluate_sample_spins(
    sample_spins: np.ndarray,
    problem: dict[str, Any],
    frontier: dict[str, np.ndarray],
    *,
    atol: float,
) -> dict[str, Any]:
    unique_spins = np.unique(np.asarray(sample_spins, dtype=np.int8), axis=0)
    energies = _energy_batch_safe(unique_spins, problem["edges"], problem["weights"], problem["h"])
    objectives = _normalize_energies(
        energies,
        np.asarray(frontier["lower_bounds"], dtype=np.float64),
        np.asarray(frontier["upper_bounds"], dtype=np.float64),
    )
    unique_objs = np.unique(np.asarray(objectives, dtype=np.float64), axis=0)
    nd_objs = unique_objs[_safe_non_dominated_indices(unique_objs)]
    hv, hv_method = _safe_hypervolume(
        nd_objs,
        ref=HV_REF,
        certificate_objs=np.asarray(frontier["objectives_norm"], dtype=np.float64),
        certificate_hv=float(np.asarray(frontier["hv_exact"])),
        atol=float(atol),
    )
    return {
        "sample_unique_spins": int(unique_spins.shape[0]),
        "sample_unique_objectives": int(unique_objs.shape[0]),
        "sample_nd_count": int(nd_objs.shape[0]),
        "sample_frontier_objectives_match_exact_npz": _objectives_match(
            nd_objs, np.asarray(frontier["objectives_norm"], dtype=np.float64), atol=float(atol)
        ),
        "hv_sample": float(hv),
        "hv_method": hv_method,
    }


def _case_path_from_arg(raw: str) -> Path:
    text = str(raw)
    if text.endswith(".npz") or "/" in text or "\\" in text:
        path = Path(text)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(text):02d}.npz"


def _frontier_path_for_case(case_path: Path, frontier_dir: Path) -> Path:
    return frontier_dir / case_path.name


def run_oracle_case(
    case_path: Path,
    *,
    frontier_dir: Path,
    sample_budget: int,
    seed: int,
    qubit_order: str,
    out: Path | None,
    sample_out: Path | None,
    atol: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    case_path = case_path.resolve()
    frontier_path = _frontier_path_for_case(case_path, frontier_dir.resolve())
    if not case_path.exists():
        raise FileNotFoundError(case_path)
    if not frontier_path.exists():
        raise FileNotFoundError(frontier_path)

    problem = _problem_from_npz(case_path)
    frontier_npz = np.load(frontier_path)
    frontier = {key: np.asarray(frontier_npz[key]) for key in frontier_npz.files}
    n = int(problem["a"] * problem["b"])
    frontier_spins = _state_indices_to_spins(frontier["state_indices"], n)

    sample_spins, sampling_meta = _sample_frontier_basis_states(
        frontier_spins,
        sample_budget=int(sample_budget),
        seed=int(seed),
        qubit_order=str(qubit_order),
    )

    eval_row = _evaluate_sample_spins(sample_spins, problem, frontier, atol=float(atol))
    hv_exact = float(np.asarray(frontier["hv_exact"]))
    hv_sample = float(eval_row["hv_sample"])
    hv_abs_diff = abs(hv_sample - hv_exact)

    sample_out_path: Path | None = None
    if sample_out is not None:
        sample_out_path = sample_out if sample_out.is_absolute() else ROOT / sample_out
        sample_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_out_path,
            sample_spins=sample_spins,
            case=str(case_path.name),
            frontier_npz=str(_project_path(frontier_path)),
        )

    summary: dict[str, Any] = {
        "case": case_path.name,
        "case_npz": _project_path(case_path),
        "frontier_npz": _project_path(frontier_path),
        "oracle_only": True,
        "contest_safe": False,
        "note": (
            "Oracle diagnostic only: it prepares exact-frontier computational basis states "
            "and samples them. Do not use this as main1 contest logic."
        ),
        "n": n,
        "k": int(problem["k"]),
        "sample_budget": int(sample_budget),
        "sample_used": int(sample_spins.shape[0]),
        "frontier_state_count": int(frontier_spins.shape[0]),
        "hv_exact": hv_exact,
        "hv_sample": hv_sample,
        "hv_abs_diff": float(hv_abs_diff),
        "hv_within_tolerance": bool(hv_abs_diff <= float(atol)),
        "full_score_oracle_proof": bool(
            hv_abs_diff <= float(atol)
            and eval_row["sample_frontier_objectives_match_exact_npz"]
            and int(sample_spins.shape[0]) == int(sample_budget)
        ),
        "elapsed_s": float(time.perf_counter() - t0),
        **sampling_meta,
        **eval_row,
    }
    if sample_out_path is not None:
        summary["sample_npz"] = _project_path(sample_out_path)

    if out is not None:
        out_path = out if out.is_absolute() else ROOT / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        summary["summary_json"] = _project_path(out_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle-only exact-frontier MindQuantum basis sampler for public cases."
    )
    parser.add_argument("--case", default="09", help="Case suffix such as 09, or explicit case npz path.")
    parser.add_argument("--frontier-dir", default="results/exact_frontiers")
    parser.add_argument("--sample-budget", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qubit-order", choices=["auto", "little", "reversed"], default="auto")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--out", default=None, help="JSON summary path. Defaults to results/oracle_exact_<case>.json.")
    parser.add_argument("--sample-out", default=None, help="Optional npz path for dense sampled spins.")
    args = parser.parse_args()

    case_path = _case_path_from_arg(str(args.case))
    frontier_dir = Path(args.frontier_dir)
    if not frontier_dir.is_absolute():
        frontier_dir = ROOT / frontier_dir

    out = args.out
    out_path = Path(out) if out else ROOT / "results" / f"oracle_exact_quantum_sampler_{case_path.stem}.json"
    sample_out = Path(args.sample_out) if args.sample_out else None

    try:
        summary = run_oracle_case(
            case_path,
            frontier_dir=frontier_dir,
            sample_budget=int(args.sample_budget),
            seed=int(args.seed),
            qubit_order=str(args.qubit_order),
            out=out_path,
            sample_out=sample_out,
            atol=float(args.atol),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    except Exception as exc:
        error_summary = {
            "case": case_path.name,
            "case_npz": _project_path(case_path),
            "frontier_npz": _project_path(_frontier_path_for_case(case_path, frontier_dir)),
            "oracle_only": True,
            "contest_safe": False,
            "full_score_oracle_proof": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(error_summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(json.dumps(error_summary, ensure_ascii=False, indent=2, default=_json_default))
        raise


if __name__ == "__main__":
    main()
