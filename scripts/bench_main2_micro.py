from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CASE = "data/large/large_k5_grid40x50_00.npz"
DEFAULT_CHUNKS = (8192, 16384, 32768)
DEFAULT_SHOTS = 20000
FULL_SHOTS = 200000


def parse_chunks(raw: str) -> list[int]:
    chunks = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not chunks:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    if any(chunk <= 0 for chunk in chunks):
        raise argparse.ArgumentTypeError("chunk sizes must be positive integers")
    return chunks


def frontier_metrics(
    frontier: np.ndarray,
    reference: np.ndarray,
    hv: float,
    reference_hv: float,
) -> dict[str, Any]:
    shape_match = tuple(frontier.shape) == tuple(reference.shape)
    if shape_match:
        diff = np.abs(frontier - reference)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        allclose = bool(np.allclose(frontier, reference, atol=1e-8, rtol=0.0))
    else:
        max_diff = None
        allclose = False
    return {
        "frontier_shape_match": bool(shape_match),
        "frontier_allclose": bool(allclose),
        "hv_abs_diff": float(abs(float(hv) - float(reference_hv))),
        "max_frontier_abs_diff": max_diff,
    }


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("chunk", "chunk_size"),
        ("outer_s", "outer_elapsed_s"),
        ("reported_s", "reported_elapsed_s"),
        ("hv", "hv"),
        ("nd", "nd_count"),
        ("shape", "frontier_shape_match"),
        ("allclose", "frontier_allclose"),
        ("hv_diff", "hv_abs_diff"),
        ("max_diff", "max_frontier_abs_diff"),
    ]
    formatted: list[list[str]] = []
    for row in rows:
        formatted.append(
            [
                str(row[key])
                if key in {"chunk_size", "nd_count"}
                else format_float(row[key], digits=12 if key in {"hv", "hv_abs_diff", "max_frontier_abs_diff"} else 6)
                for _, key in columns
            ]
        )

    widths = [
        max(len(header), *(len(row[idx]) for row in formatted))
        for idx, (header, _) in enumerate(columns)
    ]
    header = "  ".join(header.rjust(widths[idx]) for idx, (header, _) in enumerate(columns))
    sep = "  ".join("-" * width for width in widths)
    print(header)
    print(sep)
    for row in formatted:
        print("  ".join(value.rjust(widths[idx]) for idx, value in enumerate(row)))


def write_outputs(out_base: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base if out_base.suffix.lower() == ".json" else out_base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark main2 large_random_frontier_hv chunk stability."
    )
    parser.add_argument("--case", default=DEFAULT_CASE, help="NPZ case path.")
    parser.add_argument("--shots", type=int, default=None, help=f"Shot count, default {DEFAULT_SHOTS}.")
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Use {FULL_SHOTS} shots unless --shots is explicitly set.",
    )
    parser.add_argument(
        "--chunks",
        type=parse_chunks,
        default=list(DEFAULT_CHUNKS),
        help="Comma-separated chunk sizes, e.g. 8192,16384,32768.",
    )
    parser.add_argument("--seed", type=int, default=101, help="RNG seed.")
    parser.add_argument(
        "--out",
        default="results/main2_micro_bench.json",
        help="Output JSON path. A sibling CSV is also written.",
    )
    args = parser.parse_args()

    shots = FULL_SHOTS if args.full and args.shots is None else int(args.shots or DEFAULT_SHOTS)
    if shots <= 0:
        parser.error("--shots must be a positive integer")

    case = Path(args.case)
    from utils import large_random_frontier_hv, problem_from_npz  # noqa: E402

    problem = problem_from_npz(str(case))

    rows: list[dict[str, Any]] = []
    reference_frontier: np.ndarray | None = None
    reference_hv: float | None = None

    for idx, chunk_size in enumerate(args.chunks):
        t0 = time.perf_counter()
        out = large_random_frontier_hv(
            problem,
            shots=shots,
            chunk_size=int(chunk_size),
            rng_seed=int(args.seed),
        )
        outer_elapsed_s = time.perf_counter() - t0

        frontier = np.asarray(out["frontier_objectives_norm"], dtype=np.float64)
        hv = float(out["hv"])
        if idx == 0:
            reference_frontier = frontier
            reference_hv = hv

        assert reference_frontier is not None
        assert reference_hv is not None
        metrics = frontier_metrics(frontier, reference_frontier, hv, reference_hv)
        row = {
            "case": str(case),
            "shots": int(shots),
            "seed": int(args.seed),
            "chunk_size": int(chunk_size),
            "outer_elapsed_s": float(outer_elapsed_s),
            "reported_elapsed_s": float(out["elapsed_s"]),
            "hv": hv,
            "nd_count": int(out["nd_count"]),
            "frontier_shape": list(frontier.shape),
            **metrics,
        }
        rows.append(row)

    payload = {
        "case": str(case),
        "shots": int(shots),
        "seed": int(args.seed),
        "chunks": [int(chunk) for chunk in args.chunks],
        "reference_chunk": int(args.chunks[0]),
        "rows": rows,
    }

    print_table(rows)
    write_outputs(Path(args.out), payload, rows)


if __name__ == "__main__":
    main()
