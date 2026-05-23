from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import answer  # noqa: E402
from run import _hv_from_spins, baseline_hv  # noqa: E402
from utils import problem_from_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()

    case = ROOT / args.case
    problem = problem_from_npz(str(case))
    base = baseline_hv(case, problem)
    for seed in args.seed:
        t0 = time.time()
        result = answer.main1(problem, rng_seed=int(seed))
        hv = _hv_from_spins(problem, result["sample_spins"])
        score = max(float(hv) - float(base), 0.0) * 100000.0
        print(
            f"{case.name},seed={int(seed)},hv={hv:.12f},base={base:.12f},"
            f"score={score:.6f},rows={len(result['sample_spins'])},elapsed={time.time() - t0:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
