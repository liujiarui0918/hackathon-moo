from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    HV_REF,
    energy_batch_fast,
    hypervolume_pygmo,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    pg_non_dominated_indices,
    problem_from_npz,
)


def main() -> None:
    case = Path("data/large/large_k5_grid40x50_00.npz")
    problem = problem_from_npz(str(case))
    shots = 200000
    chunk_size = 4096
    rng = np.random.default_rng(101)
    lower, upper = objective_extrema(problem)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)

    t_random = 0.0
    t_energy = 0.0
    t_nd_local = 0.0
    t_merge = 0.0
    t_norm = 0.0
    remaining = shots
    while remaining > 0:
        bs = min(chunk_size, remaining)
        t0 = time.perf_counter()
        spins = np.where(rng.random((bs, int(problem.n))) < 0.5, 1, -1).astype(np.int8)
        t_random += time.perf_counter() - t0

        t0 = time.perf_counter()
        energies = np.asarray(
            energy_batch_fast(spins, problem.edges, problem.weights, problem.h),
            dtype=np.float64,
        )
        t_energy += time.perf_counter() - t0

        t0 = time.perf_counter()
        objs = normalize_energies(energies, lower, upper)
        t_norm += time.perf_counter() - t0

        t0 = time.perf_counter()
        nd = pg_non_dominated_indices(objs)
        t_nd_local += time.perf_counter() - t0

        t0 = time.perf_counter()
        nd_pool = merge_non_dominated_pool(nd_pool, objs[nd])
        t_merge += time.perf_counter() - t0
        remaining -= bs

    t0 = time.perf_counter()
    hv = hypervolume_pygmo(nd_pool, ref=HV_REF)
    t_hv = time.perf_counter() - t0
    print("random", t_random)
    print("energy", t_energy)
    print("norm", t_norm)
    print("nd_local", t_nd_local)
    print("merge", t_merge)
    print("hv", t_hv)
    print("nd_count", nd_pool.shape[0], "hv_value", hv)


if __name__ == "__main__":
    main()
