from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.oracle_exact_quantum_sampler import _json_default, run_oracle_case


def _case_path(raw: str) -> Path:
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _default_cases() -> list[Path]:
    return sorted((ROOT / "data" / "public").glob("k5_grid4x5_*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the oracle exact sampler over public cases.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case suffixes, or explicit npz paths.")
    parser.add_argument("--frontier-dir", default="results/exact_frontiers")
    parser.add_argument("--sample-budget", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qubit-order", choices=["auto", "little", "reversed"], default="auto")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--out", default="results/oracle_exact_public_summary.json")
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()

    frontier_dir = Path(args.frontier_dir)
    if not frontier_dir.is_absolute():
        frontier_dir = ROOT / frontier_dir

    case_paths = [_case_path(raw) for raw in args.cases] if args.cases else _default_cases()
    rows: list[dict[str, Any]] = []
    for i, case_path in enumerate(case_paths):
        sample_out = None
        if args.save_samples:
            sample_out = ROOT / "results" / "oracle_samples" / f"{case_path.stem}_samples.npz"
        try:
            row = run_oracle_case(
                case_path,
                frontier_dir=frontier_dir,
                sample_budget=int(args.sample_budget),
                seed=int(args.seed) + i * 100003,
                qubit_order=str(args.qubit_order),
                out=None,
                sample_out=sample_out,
                atol=float(args.atol),
            )
        except Exception as exc:
            row = {
                "case": case_path.name,
                "case_npz": str(case_path),
                "frontier_npz": str(frontier_dir / case_path.name),
                "oracle_only": True,
                "contest_safe": False,
                "full_score_oracle_proof": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)

    hv_diffs = [float(row["hv_abs_diff"]) for row in rows if "hv_abs_diff" in row]
    payload = {
        "rows": rows,
        "case_count": int(len(rows)),
        "all_full_score_oracle_proof": bool(rows and all(row["full_score_oracle_proof"] for row in rows)),
        "max_hv_abs_diff": float(np.max(hv_diffs)) if hv_diffs else None,
        "oracle_only": True,
        "contest_safe": False,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({"event": "wrote_summary", "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
