from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import large_random_frontier_hv, problem_from_npz


def main() -> None:
    case = Path("data/large/large_k5_grid40x50_00.npz")
    problem = problem_from_npz(str(case))
    outs = []
    for chunk in [4096, 8192, 16384, 32768, 65536]:
        t0 = time.perf_counter()
        out = large_random_frontier_hv(problem, shots=200000, chunk_size=chunk, rng_seed=101)
        elapsed = time.perf_counter() - t0
        print(
            "chunk",
            chunk,
            "elapsed_outer",
            f"{elapsed:.6f}",
            "elapsed_inner",
            f"{float(out['elapsed_s']):.6f}",
            "hv",
            f"{float(out['hv']):.12f}",
            "nd",
            int(out["nd_count"]),
            flush=True,
        )
        outs.append((chunk, out))

    base_front = np.asarray(outs[0][1]["frontier_objectives_norm"], dtype=np.float64)
    for chunk, out in outs[1:]:
        frontier = np.asarray(out["frontier_objectives_norm"], dtype=np.float64)
        match = frontier.shape == base_front.shape and np.allclose(
            frontier, base_front, atol=1e-8, rtol=0.0
        )
        max_diff = (
            float(np.max(np.abs(frontier - base_front)))
            if frontier.shape == base_front.shape and frontier.size
            else None
        )
        print("match", chunk, bool(match), "maxdiff", max_diff, flush=True)


if __name__ == "__main__":
    main()
