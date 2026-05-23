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

from run import baseline_hv, _hv_from_spins
from utils import (
    build_qaoa_circuit_from_projected_ising,
    load_transfer_params_csv,
    load_weight_pool,
    problem_from_npz,
    sampling_result_to_unique_spins,
)


def scan(case: Path, n_lambdas: int, shots: int, seed: int) -> None:
    problem = problem_from_npz(str(case))
    pool = load_weight_pool(int(problem.k), n=1000, seed=2026).astype(np.float64)
    table = load_transfer_params_csv(str(ROOT / "transfer_data.csv"), q_target=2, p_list=(3,))
    betas, gammas = table[3]
    proj_j = np.asarray(pool @ problem.weights, dtype=np.float64)
    proj_h = np.asarray(pool @ problem.h, dtype=np.float64)
    sim = Simulator("mqvector", int(problem.n), seed=int(seed))
    base = baseline_hv(case, problem)
    rows = []
    t0 = time.time()
    for lam_id in range(int(n_lambdas)):
        circ = build_qaoa_circuit_from_projected_ising(
            problem,
            proj_j[lam_id],
            proj_h[lam_id],
            betas=betas,
            gammas=gammas,
        )
        sim.reset()
        res = sim.sampling(circ, shots=int(shots), seed=int(seed + lam_id))
        u, c = sampling_result_to_unique_spins(res, int(problem.n))
        spins = np.repeat(u, c.astype(np.int32), axis=0).astype(np.int8)
        hv = _hv_from_spins(problem, spins)
        rows.append((lam_id, hv, max(hv - base, 0.0)))
    out = ROOT / "results" / f"lambda_scan_{case.stem}_n{n_lambdas}_s{shots}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "lambda_id,hv,gain\n"
        + "\n".join(f"{i},{hv:.16g},{gain:.16g}" for i, hv, gain in rows),
        encoding="utf-8",
    )
    best = sorted(rows, key=lambda x: x[2], reverse=True)[:10]
    print(case.name, "base", base, "elapsed", time.time() - t0, "out", out)
    print("best", best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--n-lambdas", type=int, default=1000)
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    scan(ROOT / args.case, args.n_lambdas, args.shots, args.seed)


if __name__ == "__main__":
    main()
