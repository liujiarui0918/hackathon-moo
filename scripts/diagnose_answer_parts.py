from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from mindquantum.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import answer  # noqa: E402
from run import baseline_hv, _hv_from_spins  # noqa: E402
from utils import (  # noqa: E402
    build_qaoa_circuit_from_projected_ising,
    energy_batch_fast,
    hypervolume_pygmo,
    load_weight_pool,
    normalize_energies,
    objective_extrema,
    pg_non_dominated_indices,
    problem_from_npz,
    sampling_result_to_unique_spins,
)


def _sample_blocks(problem, lam_ids, shots, proj_j, proj_h, betas, gammas, *, seed, warm_bits=None, warm_c=0.1):
    sim = Simulator("mqvector", int(problem.n), seed=int(seed))
    parts = []
    for pos, lam_id in enumerate(np.asarray(lam_ids, dtype=np.int64)):
        wb = None if warm_bits is None else warm_bits[pos]
        circ = build_qaoa_circuit_from_projected_ising(
            problem,
            proj_j[int(lam_id)],
            proj_h[int(lam_id)],
            betas=betas,
            gammas=gammas,
            warm_bits01=wb,
            warm_c=float(warm_c),
        )
        sim.reset()
        res = sim.sampling(circ, shots=int(shots), seed=int(seed + pos))
        unique_spins, counts = sampling_result_to_unique_spins(res, int(problem.n))
        parts.append(np.repeat(unique_spins, counts.astype(np.int32), axis=0))
    return np.vstack(parts).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--warm-c", type=float, default=answer.WARM_C_FIXED)
    parser.add_argument("--local-count", type=int, default=answer.LOCAL_WARM_NUM_WEIGHTS)
    parser.add_argument("--local-shots", type=int, default=answer.LOCAL_WARM_SHOTS_PER_WEIGHT)
    parser.add_argument("--restarts", type=int, default=6)
    args = parser.parse_args()

    problem = problem_from_npz(str(ROOT / args.case))
    pool = load_weight_pool(int(problem.k), n=answer.LAMBDA_POOL_SIZE, seed=2026).astype(np.float64)
    proj_j = np.asarray(pool @ problem.weights, dtype=np.float64)
    proj_h = np.asarray(pool @ problem.h, dtype=np.float64)
    betas, gammas = answer._TRANSFER_TABLE[answer.P_LAYERS]
    base = baseline_hv(ROOT / args.case, problem)
    seed = 2026

    t0 = time.time()
    broad = _sample_blocks(
        problem,
        np.arange(answer.BROAD_NUM_WEIGHTS, dtype=np.int64),
        answer.BROAD_SHOTS_PER_WEIGHT,
        proj_j,
        proj_h,
        betas,
        gammas,
        seed=seed,
    )
    print("broad_score", max(_hv_from_spins(problem, broad) - base, 0.0) * 100000, "t", time.time() - t0)

    t0 = time.time()
    local_spins, local_objs = answer._multiobjective_local_frontier(
        problem,
        pool,
        proj_j,
        proj_h,
        seed=seed + 70000,
        restarts=int(args.restarts),
    )
    local_hv = hypervolume_pygmo(local_objs[pg_non_dominated_indices(local_objs)]) if local_objs.size else 0.0
    print(
        "local_frontier_score",
        max(float(local_hv) - base, 0.0) * 100000,
        "local_nd",
        int(local_spins.shape[0]),
        "t",
        time.time() - t0,
    )

    warm_bits, warm_lams = answer._select_diverse_warm_states(
        local_spins,
        local_objs,
        pool,
        count=int(args.local_count),
    )
    t0 = time.time()
    warm = _sample_blocks(
        problem,
        warm_lams,
        int(args.local_shots),
        proj_j,
        proj_h,
        betas,
        gammas,
        seed=seed + 10000,
        warm_bits=warm_bits,
        warm_c=float(args.warm_c),
    )
    print("warm_score", max(_hv_from_spins(problem, warm) - base, 0.0) * 100000, "t", time.time() - t0)
    combo = np.vstack([broad, warm]).astype(np.int8)
    print("combo_score", max(_hv_from_spins(problem, combo) - base, 0.0) * 100000)


if __name__ == "__main__":
    main()
