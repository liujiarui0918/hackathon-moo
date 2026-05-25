from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from mindquantum.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import answer  # noqa: E402
from run import baseline_hv  # noqa: E402
from utils import (  # noqa: E402
    build_qaoa_circuit_from_projected_ising,
    hypervolume_pygmo,
    load_transfer_params_csv,
    load_weight_pool,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    pg_non_dominated_indices,
    problem_from_npz,
    sampling_result_to_unique_spins,
)

BASE_SAMPLE_BUDGET = 100000
DEFAULT_CASE = "data/public/k5_grid4x5_09.npz"
DEFAULT_QUICK_COMBOS = ["broad_gamma3_p3"]
FULL_COMBOS = [
    "broad_p3",
    "broad_gamma3_p3",
    "beta_gamma5_p3",
    "warmc4_local_p3",
    "p23_split",
    "p23_gamma3",
]

CSV_FIELDS = [
    "case",
    "case_suffix",
    "kind",
    "combo",
    "seed",
    "rows",
    "hv",
    "base",
    "gain",
    "score",
    "elapsed",
    "p_layers",
    "gamma_scales",
    "beta_scales",
    "warm_cs",
    "allocation",
    "error",
]


@dataclass(frozen=True)
class Arm:
    name: str
    rows: int
    lambda_count: int
    p_layers: int
    gamma_scale: float = 1.0
    beta_scale: float = 1.0
    warm_c: float | None = None
    warm_source: str = "none"


def _case_path(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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
    return f"results/param_portfolio_grid_{stamp}"


def _write_outputs(rows: list[dict[str, Any]], out: str) -> tuple[Path, Path]:
    csv_path, json_path = _out_paths(out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
    return csv_path, json_path


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _hv_from_spins_safe(problem: Any, spins: np.ndarray) -> float:
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
    return float(hypervolume_pygmo(nd_pool))


def _repeat_unique(unique_spins: np.ndarray, counts: np.ndarray, shots: int) -> np.ndarray:
    spins = np.repeat(unique_spins, np.asarray(counts, dtype=np.int32), axis=0)
    if int(spins.shape[0]) != int(shots):
        raise ValueError(f"sampling row count mismatch: got {spins.shape[0]}, expected {shots}")
    return np.asarray(spins, dtype=np.int8)


def _local_descent_spin(problem: Any, j_raw: np.ndarray, h_raw: np.ndarray, *, seed: int, restarts: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(problem.n)
    edges = np.asarray(problem.edges, dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]
    j = np.asarray(j_raw, dtype=np.float64)
    h = np.asarray(h_raw, dtype=np.float64)
    starts = [
        np.where(h <= 0.0, 1, -1).astype(np.int8),
        np.where(h >= 0.0, 1, -1).astype(np.int8),
    ]
    for _ in range(max(0, int(restarts) - len(starts))):
        starts.append(np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8))

    best = np.asarray(starts[0], dtype=np.int8)
    best_e = np.inf
    for start in starts[: int(restarts)]:
        z = np.asarray(start, dtype=np.int8).copy()
        for _ in range(32):
            improved = False
            for q in rng.permutation(n):
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
            if not improved:
                break
        e = float(np.dot(h, z) + np.dot(j, z[u] * z[v]))
        if e < best_e:
            best_e = e
            best = z.copy()
    return np.asarray(best, dtype=np.int8)


def _local_warm_bits(
    problem: Any,
    lambda_ids: np.ndarray,
    proj_j: np.ndarray,
    proj_h: np.ndarray,
    *,
    seed: int,
    restarts: int,
) -> list[np.ndarray]:
    warm: list[np.ndarray] = []
    for pos, lam_id in enumerate(np.asarray(lambda_ids, dtype=np.int64)):
        spin = _local_descent_spin(
            problem,
            proj_j[int(lam_id)],
            proj_h[int(lam_id)],
            seed=int(seed + pos * 7919),
            restarts=int(restarts),
        )
        warm.append(np.where(spin > 0, 0, 1).astype(np.int8))
    return warm


def _sample_arm(
    problem: Any,
    arm: Arm,
    table: dict[int, tuple[np.ndarray, np.ndarray]],
    proj_j: np.ndarray,
    proj_h: np.ndarray,
    *,
    seed: int,
    warm_cache: dict[tuple[str, int], list[np.ndarray]],
    warm_restarts: int,
) -> np.ndarray:
    if int(arm.rows) % int(arm.lambda_count) != 0:
        raise ValueError(f"{arm.name}: rows must be divisible by lambda_count")
    if int(arm.p_layers) not in table:
        raise ValueError(f"{arm.name}: missing transfer params for p={arm.p_layers}")
    if int(arm.lambda_count) > int(proj_j.shape[0]):
        raise ValueError(f"{arm.name}: lambda_count exceeds pool size")

    shots = int(arm.rows) // int(arm.lambda_count)
    lambda_ids = np.arange(int(arm.lambda_count), dtype=np.int64)
    betas0, gammas0 = table[int(arm.p_layers)]
    betas = np.asarray(betas0, dtype=np.float64) * float(arm.beta_scale)
    gammas = np.asarray(gammas0, dtype=np.float64) * float(arm.gamma_scale)

    warm_bits: list[np.ndarray | None]
    if arm.warm_source == "none":
        warm_bits = [None] * int(arm.lambda_count)
    elif arm.warm_source == "local":
        key = (arm.warm_source, int(arm.lambda_count))
        if key not in warm_cache:
            warm_cache[key] = _local_warm_bits(
                problem,
                lambda_ids,
                proj_j,
                proj_h,
                seed=int(seed + 50000),
                restarts=int(warm_restarts),
            )
        warm_bits = list(warm_cache[key])
    else:
        raise ValueError(f"{arm.name}: unknown warm_source={arm.warm_source}")

    sim = Simulator("mqvector", int(problem.n), seed=int(seed) % (2**23))
    out = np.empty((int(arm.rows), int(problem.n)), dtype=np.int8)
    cursor = 0
    for pos, lam_id in enumerate(lambda_ids):
        circ = build_qaoa_circuit_from_projected_ising(
            problem,
            proj_j[int(lam_id)],
            proj_h[int(lam_id)],
            betas=betas,
            gammas=gammas,
            warm_bits01=warm_bits[pos],
            warm_c=0.5 if arm.warm_c is None else float(arm.warm_c),
        )
        sim.reset()
        res = sim.sampling(circ, shots=shots, seed=int(seed + pos))
        unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
        block = _repeat_unique(unique_spins, counts, shots)
        out[cursor : cursor + shots] = block
        cursor += shots
    if cursor != int(arm.rows):
        raise RuntimeError(f"{arm.name}: wrote {cursor} rows, expected {arm.rows}")
    return out


def _combo_arms(name: str) -> list[Arm]:
    if name == "broad_p3":
        return [Arm("p3_g1.0_b1.0_nowarm_100k", 100000, 1000, 3)]
    if name == "broad_gamma3_p3":
        return [
            Arm("p3_g0.8_b1.0_nowarm_30k", 30000, 1000, 3, gamma_scale=0.8),
            Arm("p3_g1.0_b1.0_nowarm_40k", 40000, 1000, 3, gamma_scale=1.0),
            Arm("p3_g1.2_b1.0_nowarm_30k", 30000, 1000, 3, gamma_scale=1.2),
        ]
    if name == "beta_gamma5_p3":
        return [
            Arm("p3_g1.0_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=1.0, beta_scale=1.0),
            Arm("p3_g0.8_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=0.8, beta_scale=1.0),
            Arm("p3_g1.2_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=1.2, beta_scale=1.0),
            Arm("p3_g1.0_b0.9_nowarm_20k", 20000, 1000, 3, gamma_scale=1.0, beta_scale=0.9),
            Arm("p3_g1.0_b1.1_nowarm_20k", 20000, 1000, 3, gamma_scale=1.0, beta_scale=1.1),
        ]
    if name == "warmc4_local_p3":
        return [
            Arm("p3_g1.0_b1.0_local_c0.03_25k", 25000, 500, 3, warm_c=0.03, warm_source="local"),
            Arm("p3_g1.0_b1.0_local_c0.05_25k", 25000, 500, 3, warm_c=0.05, warm_source="local"),
            Arm("p3_g1.0_b1.0_local_c0.10_25k", 25000, 500, 3, warm_c=0.10, warm_source="local"),
            Arm("p3_g1.0_b1.0_local_c0.15_25k", 25000, 500, 3, warm_c=0.15, warm_source="local"),
        ]
    if name == "p23_split":
        return [
            Arm("p2_g1.0_b1.0_nowarm_50k", 50000, 1000, 2),
            Arm("p3_g1.0_b1.0_nowarm_50k", 50000, 1000, 3),
        ]
    if name == "p23_gamma3":
        return [
            Arm("p2_g1.0_b1.0_nowarm_40k", 40000, 1000, 2),
            Arm("p3_g0.8_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=0.8),
            Arm("p3_g1.0_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=1.0),
            Arm("p3_g1.2_b1.0_nowarm_20k", 20000, 1000, 3, gamma_scale=1.2),
        ]
    raise ValueError(f"unknown combo: {name}")


def _allocation_summary(arms: list[Arm]) -> str:
    return json.dumps([arm.__dict__ for arm in arms], sort_keys=True, separators=(",", ":"))


def _scales_summary(arms: list[Arm], attr: str) -> str:
    vals = []
    for arm in arms:
        value = getattr(arm, attr)
        if value is not None and value not in vals:
            vals.append(value)
    return json.dumps(vals, separators=(",", ":"))


def _run_current_answer(case: Path, problem: Any, base: float) -> dict[str, Any]:
    t0 = time.time()
    result = answer.main1(problem_input=problem)
    spins = np.asarray(result["sample_spins"], dtype=np.int8)
    hv = _hv_from_spins_safe(problem, spins)
    gain = max(float(hv - base), 0.0)
    return {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "kind": "current_answer",
        "combo": "answer.main1",
        "seed": "answer_default",
        "rows": int(spins.shape[0]),
        "hv": float(hv),
        "base": float(base),
        "gain": float(gain),
        "score": float(gain * 100000.0),
        "elapsed": float(time.time() - t0),
        "p_layers": "",
        "gamma_scales": "",
        "beta_scales": "",
        "warm_cs": "",
        "allocation": "",
        "error": "",
    }


def _run_combo(case: Path, problem: Any, combo: str, seed: int, warm_restarts: int, base: float) -> dict[str, Any]:
    arms = _combo_arms(combo)
    total_rows = sum(int(arm.rows) for arm in arms)
    if total_rows != BASE_SAMPLE_BUDGET:
        raise ValueError(f"{combo}: total rows {total_rows}, expected {BASE_SAMPLE_BUDGET}")

    needed_p = sorted({int(arm.p_layers) for arm in arms})
    table = load_transfer_params_csv(str(ROOT / "transfer_data.csv"), q_target=2, p_list=needed_p)
    missing = [p for p in needed_p if p not in table]
    if missing:
        raise ValueError(f"{combo}: missing transfer_data.csv params for p={missing}")

    pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)
    proj_j = np.einsum("lk,km->lm", pool, problem.weights, optimize=False).astype(np.float64, copy=False)
    proj_h = np.einsum("lk,kn->ln", pool, problem.h, optimize=False).astype(np.float64, copy=False)

    t0 = time.time()
    parts = []
    warm_cache: dict[tuple[str, int], list[np.ndarray]] = {}
    seed_stride = 100000
    for pos, arm in enumerate(arms):
        parts.append(
            _sample_arm(
                problem,
                arm,
                table,
                proj_j,
                proj_h,
                seed=int(seed + pos * seed_stride),
                warm_cache=warm_cache,
                warm_restarts=int(warm_restarts),
            )
        )
    spins = np.vstack(parts).astype(np.int8)
    if int(spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise RuntimeError(f"{combo}: produced {spins.shape[0]} rows")

    hv = _hv_from_spins_safe(problem, spins)
    gain = max(float(hv - base), 0.0)
    return {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "kind": "portfolio",
        "combo": combo,
        "seed": int(seed),
        "rows": int(spins.shape[0]),
        "hv": float(hv),
        "base": float(base),
        "gain": float(gain),
        "score": float(gain * 100000.0),
        "elapsed": float(time.time() - t0),
        "p_layers": _scales_summary(arms, "p_layers"),
        "gamma_scales": _scales_summary(arms, "gamma_scale"),
        "beta_scales": _scales_summary(arms, "beta_scale"),
        "warm_cs": _scales_summary(arms, "warm_c"),
        "allocation": _allocation_summary(arms),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 100000-row parameter portfolio ablations for main1-style QAOA sampling."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the guarded smoke grid.")
    mode.add_argument("--full", action="store_true", help="Run all built-in portfolio combos.")
    parser.add_argument("--case", action="append", dest="cases", help="Case suffix, path, or repeated cases.")
    parser.add_argument("--combo", action="append", choices=FULL_COMBOS, help="Portfolio combo to run; repeatable.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warm-restarts", type=int, default=4)
    parser.add_argument("--include-current", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    quick = args.quick or not args.full
    cases = [_case_path(c) for c in (args.cases or [DEFAULT_CASE])]
    combos = args.combo or (DEFAULT_QUICK_COMBOS if quick else FULL_COMBOS)
    out = args.out or _default_out()

    missing_cases = [str(path) for path in cases if not path.exists()]
    if missing_cases:
        raise SystemExit(f"Missing case file(s): {', '.join(missing_cases)}")
    unknown = [combo for combo in combos if combo not in FULL_COMBOS]
    if unknown:
        raise SystemExit(f"Unknown combo(s): {', '.join(unknown)}")

    if args.dry_run:
        for case in cases:
            if args.include_current:
                print(f"{case.name}: current answer.main1")
            for combo in combos:
                print(f"{case.name}: {combo}: {_allocation_summary(_combo_arms(combo))}")
        return

    rows: list[dict[str, Any]] = []
    for case in cases:
        problem = problem_from_npz(str(case))
        base = baseline_hv(case, problem)
        if args.include_current:
            try:
                row = _run_current_answer(case, problem, base)
            except Exception as exc:
                row = {
                    "case": case.name,
                    "case_suffix": _case_suffix(case),
                    "kind": "current_answer",
                    "combo": "answer.main1",
                    "seed": "answer_default",
                    "rows": "",
                    "hv": "",
                    "base": float(base),
                    "gain": "",
                    "score": "",
                    "elapsed": "",
                    "p_layers": "",
                    "gamma_scales": "",
                    "beta_scales": "",
                    "warm_cs": "",
                    "allocation": "",
                    "error": repr(exc),
                }
            rows.append(row)
            _write_outputs(rows, out)
            print(
                f"{row['case']},{row['kind']},{row['combo']},rows={row['rows']},"
                f"score={row['score']},elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

        for combo in combos:
            try:
                row = _run_combo(case, problem, combo, int(args.seed), int(args.warm_restarts), base)
            except Exception as exc:
                row = {
                    "case": case.name,
                    "case_suffix": _case_suffix(case),
                    "kind": "portfolio",
                    "combo": combo,
                    "seed": int(args.seed),
                    "rows": "",
                    "hv": "",
                    "base": float(base),
                    "gain": "",
                    "score": "",
                    "elapsed": "",
                    "p_layers": "",
                    "gamma_scales": "",
                    "beta_scales": "",
                    "warm_cs": "",
                    "allocation": _allocation_summary(_combo_arms(combo)),
                    "error": repr(exc),
                }
            rows.append(row)
            _write_outputs(rows, out)
            print(
                f"{row['case']},{row['kind']},{row['combo']},rows={row['rows']},"
                f"score={row['score']},elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [r for r in rows if r.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda r: float(r["score"]))
        print(f"best={best['case']},{best['kind']},{best['combo']},score={float(best['score']):.6f}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
