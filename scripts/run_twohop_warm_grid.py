from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_local_warm_grid as local_grid

BASE_SAMPLE_BUDGET = local_grid.BASE_SAMPLE_BUDGET
LAMBDA_POOL_SIZE = local_grid.LAMBDA_POOL_SIZE
TRANSFER_Q_TARGET = local_grid.TRANSFER_Q_TARGET
P_LAYERS = local_grid.P_LAYERS

DEFAULT_CASE = "09"
DEFAULT_SEED = 2031
DEFAULT_WARM_C = 0.1
DEFAULT_SOURCE_LIMITS = (100, 200)
DEFAULT_BROAD_WEIGHTS = 500
DEFAULT_BROAD_SHOTS = 100
DEFAULT_WARM_COUNT = 250
DEFAULT_WARM_SHOTS = 200

CSV_FIELDS = [
    "case",
    "case_suffix",
    "case_path",
    "seed",
    "source_limit",
    "warm_c",
    "warm_count",
    "broad_weights",
    "broad_shots",
    "warm_shots",
    "rows",
    "broad_rows",
    "warm_rows",
    "q_target",
    "p_layers",
    "hv",
    "base",
    "gain",
    "score",
    "elapsed",
    "broad_unique_count",
    "broad_nd_count",
    "base_count",
    "candidate_count",
    "candidate_nd_count",
    "error",
]


@dataclass(frozen=True)
class TwoHopConfig:
    seed: int
    source_limit: int
    warm_c: float
    warm_count: int
    broad_weights: int
    broad_shots: int
    warm_shots: int

    @property
    def broad_rows(self) -> int:
        return int(self.broad_weights) * int(self.broad_shots)

    @property
    def warm_rows(self) -> int:
        return int(self.warm_count) * int(self.warm_shots)

    @property
    def rows(self) -> int:
        return self.broad_rows + self.warm_rows


def _split_values(values: list[str] | None, default: Iterable[Any]) -> list[str]:
    raw = list(default) if not values else values
    out: list[str] = []
    for item in raw:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _dedupe_preserve(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _int_grid(values: list[str] | None, default: Iterable[int] | int) -> list[int]:
    defaults = default if isinstance(default, tuple) else (default,)
    return [int(v) for v in _dedupe_preserve(_split_values(values, defaults))]


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _case_values(values: list[str] | None) -> list[str]:
    return [str(v) for v in _dedupe_preserve(_split_values(values, [DEFAULT_CASE]))]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _default_out() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/twohop_warm_grid_{stamp}"


def _out_paths(out: str) -> tuple[Path, Path]:
    path = Path(out)
    if not path.is_absolute():
        path = ROOT / path
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path, path.with_suffix(".json")
    if suffix == ".json":
        return path.with_suffix(".csv"), path
    return path.with_suffix(".csv"), path.with_suffix(".json")


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


def _make_configs(args: argparse.Namespace) -> list[TwoHopConfig]:
    seeds = _int_grid(args.seed, DEFAULT_SEED)
    source_limits = _int_grid(args.source_limit, DEFAULT_SOURCE_LIMITS)
    warm_cs = _float_grid(args.warm_c, DEFAULT_WARM_C)
    warm_counts = _int_grid(args.warm_count, DEFAULT_WARM_COUNT)
    broad_weights = _int_grid(args.broad_weights, DEFAULT_BROAD_WEIGHTS)
    broad_shots = _int_grid(args.broad_shots, DEFAULT_BROAD_SHOTS)
    warm_shots = _int_grid(args.warm_shots, DEFAULT_WARM_SHOTS)

    configs = []
    for seed, source_limit, warm_c, warm_count, b_weights, b_shots, w_shots in product(
        seeds,
        source_limits,
        warm_cs,
        warm_counts,
        broad_weights,
        broad_shots,
        warm_shots,
    ):
        configs.append(
            TwoHopConfig(
                seed=int(seed),
                source_limit=int(source_limit),
                warm_c=float(warm_c),
                warm_count=int(warm_count),
                broad_weights=int(b_weights),
                broad_shots=int(b_shots),
                warm_shots=int(w_shots),
            )
        )
    return configs


def _config_error(cfg: TwoHopConfig) -> str:
    errors: list[str] = []
    if cfg.rows != BASE_SAMPLE_BUDGET:
        errors.append(f"rows={cfg.rows}, expected {BASE_SAMPLE_BUDGET}")
    if not 0.0 <= float(cfg.warm_c) <= 1.0:
        errors.append("warm_c must be in [0, 1]")
    if int(cfg.source_limit) < 1:
        errors.append("source_limit must be >= 1")
    if int(cfg.broad_weights) <= 0 or int(cfg.broad_shots) <= 0:
        errors.append("broad_weights and broad_shots must be > 0")
    if int(cfg.warm_count) < 0 or int(cfg.warm_shots) < 0:
        errors.append("warm_count and warm_shots must be non-negative")
    if int(cfg.warm_count) > 0 and int(cfg.warm_shots) <= 0:
        errors.append("warm_shots must be > 0 when warm_count > 0")
    if int(cfg.broad_weights) > LAMBDA_POOL_SIZE:
        errors.append(f"broad_weights must be <= {LAMBDA_POOL_SIZE}")
    if int(cfg.warm_count) > LAMBDA_POOL_SIZE:
        errors.append(f"warm_count must be <= {LAMBDA_POOL_SIZE}")
    return "; ".join(errors)


def _base_row(case: Path, cfg: TwoHopConfig) -> dict[str, Any]:
    return {
        "case": case.name,
        "case_suffix": local_grid._case_suffix(case),
        "case_path": str(case),
        "seed": int(cfg.seed),
        "source_limit": int(cfg.source_limit),
        "warm_c": float(cfg.warm_c),
        "warm_count": int(cfg.warm_count),
        "broad_weights": int(cfg.broad_weights),
        "broad_shots": int(cfg.broad_shots),
        "warm_shots": int(cfg.warm_shots),
        "rows": int(cfg.rows),
        "broad_rows": int(cfg.broad_rows),
        "warm_rows": int(cfg.warm_rows),
        "q_target": int(TRANSFER_Q_TARGET),
        "p_layers": int(P_LAYERS),
        "hv": "",
        "base": "",
        "gain": "",
        "score": "",
        "elapsed": "",
        "broad_unique_count": "",
        "broad_nd_count": "",
        "base_count": "",
        "candidate_count": "",
        "candidate_nd_count": "",
        "error": "",
    }


def _twohop_broad_neighbor_candidates(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    broad_unique: np.ndarray,
    *,
    source_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    unique = np.unique(np.asarray(broad_unique, dtype=np.int8), axis=0)
    stats = {
        "broad_unique_count": int(unique.shape[0]),
        "broad_nd_count": 0,
        "base_count": 0,
        "candidate_count": 0,
        "candidate_nd_count": 0,
    }
    if unique.size == 0:
        return (
            np.zeros((0, int(prepared.problem.n)), dtype=np.int8),
            np.zeros((0, int(prepared.problem.k)), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            stats,
        )

    objs = local_grid._objectives_for_spins(ablate, prepared, unique)
    nd = ablate.pg_non_dominated_indices(objs)
    nd_spins = unique[nd]
    nd_objs = objs[nd]
    nd_lams = local_grid._assign_lambdas(nd_objs, prepared.pool)
    stats["broad_nd_count"] = int(nd_spins.shape[0])

    base_count = min(int(nd_spins.shape[0]), int(source_limit))
    stats["base_count"] = int(base_count)
    if base_count <= 0:
        return (
            np.zeros((0, int(prepared.problem.n)), dtype=np.int8),
            np.zeros((0, int(prepared.problem.k)), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            stats,
        )

    base_spins, _ = ablate._select_diverse_warm_states(
        nd_spins,
        nd_objs,
        nd_lams,
        count=base_count,
    )
    bases = np.asarray(base_spins, dtype=np.int8)
    n = int(prepared.problem.n)
    candidate_parts = [bases]

    one_flips = np.repeat(bases, n, axis=0)
    bit_ids = np.tile(np.arange(n, dtype=np.int64), int(bases.shape[0]))
    row_ids = np.arange(int(one_flips.shape[0]), dtype=np.int64)
    one_flips[row_ids, bit_ids] *= np.int8(-1)
    candidate_parts.append(one_flips)

    bit_pairs = np.asarray(list(combinations(range(n), 2)), dtype=np.int64)
    if bit_pairs.size:
        two_flips = np.repeat(bases, int(bit_pairs.shape[0]), axis=0)
        pair_ids = np.tile(bit_pairs, (int(bases.shape[0]), 1))
        row_ids = np.arange(int(two_flips.shape[0]), dtype=np.int64)
        two_flips[row_ids, pair_ids[:, 0]] *= np.int8(-1)
        two_flips[row_ids, pair_ids[:, 1]] *= np.int8(-1)
        candidate_parts.append(two_flips)

    candidates = np.unique(np.vstack(candidate_parts).astype(np.int8, copy=False), axis=0)
    stats["candidate_count"] = int(candidates.shape[0])
    cand_objs = local_grid._objectives_for_spins(ablate, prepared, candidates)
    cand_nd = ablate.pg_non_dominated_indices(cand_objs)
    cand_spins = candidates[cand_nd]
    cand_objs = cand_objs[cand_nd]
    cand_lams = local_grid._assign_lambdas(cand_objs, prepared.pool)
    stats["candidate_nd_count"] = int(cand_spins.shape[0])
    return cand_spins, cand_objs, cand_lams, stats


def _run_config(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    cfg: TwoHopConfig,
) -> tuple[float, float, float, float, dict[str, int]]:
    t0 = time.time()
    broad_ids = np.arange(int(cfg.broad_weights), dtype=np.int64)
    broad_spins, broad_unique_blocks, *_ = ablate._sample_round(
        prepared.problem,
        broad_ids,
        int(cfg.broad_shots),
        prepared.proj_j,
        prepared.proj_h,
        prepared.betas,
        prepared.gammas,
        seed=int(cfg.seed),
    )
    parts = [np.asarray(broad_spins, dtype=np.int8)]

    broad_unique = np.unique(
        np.vstack([np.asarray(block, dtype=np.int8) for block in broad_unique_blocks]).astype(
            np.int8,
            copy=False,
        ),
        axis=0,
    )
    cand_spins, cand_objs, cand_lams, stats = _twohop_broad_neighbor_candidates(
        ablate,
        prepared,
        broad_unique,
        source_limit=int(cfg.source_limit),
    )

    if int(cfg.warm_count) > 0:
        warm_spins, warm_lams = ablate._select_diverse_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=int(cfg.warm_count),
        )
        warm_bits, warm_lams = ablate._materialize_warm_selection(
            warm_spins,
            warm_lams,
            count=int(cfg.warm_count),
        )
        warm_spins_sampled, *_ = ablate._sample_round(
            prepared.problem,
            np.asarray(warm_lams, dtype=np.int64),
            int(cfg.warm_shots),
            prepared.proj_j,
            prepared.proj_h,
            prepared.betas,
            prepared.gammas,
            seed=int(cfg.seed + 10000),
            warm_bits=warm_bits,
            warm_c=float(cfg.warm_c),
        )
        parts.append(np.asarray(warm_spins_sampled, dtype=np.int8))

    spins = np.vstack(parts).astype(np.int8, copy=False)
    if int(spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise RuntimeError(f"produced {spins.shape[0]} rows, expected {BASE_SAMPLE_BUDGET}")

    hv = float(ablate._hv_from_spins_safe(prepared.problem, spins))
    gain = max(float(hv - prepared.base), 0.0)
    score = float(gain * 100000.0)
    elapsed = float(time.time() - t0)
    return hv, gain, score, elapsed, stats


def _print_dry_run(cases: list[Path], configs: list[TwoHopConfig]) -> None:
    for case in cases:
        exists = "ok" if case.exists() else "missing"
        for cfg in configs:
            error = _config_error(cfg)
            status = "valid" if not error else f"invalid: {error}"
            print(
                f"{case.name} ({exists}): seed={cfg.seed}, source_limit={cfg.source_limit}, "
                f"warm_c={cfg.warm_c}, broad={cfg.broad_weights}x{cfg.broad_shots}, "
                f"warm={cfg.warm_count}x{cfg.warm_shots}, rows={cfg.rows}, {status}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep two-hop sampled-neighbor warm starts. Broad ND bases, one-bit flips, and two-bit flips "
            "are used only as warm_bits; all returned rows come from MindQuantum sampling."
        )
    )
    parser.add_argument("--case", action="append", help="Case suffix like 09, stem, or .npz path. Repeatable.")
    parser.add_argument("--seed", action="append", help="Seed value(s), comma-separated or repeatable.")
    parser.add_argument("--source-limit", action="append", help="Broad sampled ND base limit(s).")
    parser.add_argument("--warm-c", action="append", help="Warm-start c value(s), comma-separated or repeatable.")
    parser.add_argument("--warm-count", action="append", help="Warm-start lambda/state count(s).")
    parser.add_argument("--broad-weights", action="append", help="Broad no-warm lambda count(s).")
    parser.add_argument("--broad-shots", action="append", help="Shots per broad lambda.")
    parser.add_argument("--warm-shots", action="append", help="Shots per warm-start state.")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded grid and validation without sampling.")
    args = parser.parse_args()

    cases = [local_grid._case_path(case) for case in _case_values(args.case)]
    configs = _make_configs(args)
    out = args.out or _default_out()

    if args.dry_run:
        _print_dry_run(cases, configs)
        return

    missing_cases = [str(case) for case in cases if not case.exists()]
    if missing_cases:
        raise SystemExit(f"Missing case file(s): {', '.join(missing_cases)}")

    ablate = local_grid._load_ablate_helpers()
    rows: list[dict[str, Any]] = []
    for case in cases:
        valid_configs = [cfg for cfg in configs if not _config_error(cfg)]
        prepared: local_grid.PreparedCase | None = None
        if valid_configs:
            prepared = local_grid._prepare_case(ablate, case)

        for cfg in configs:
            row = _base_row(case, cfg)
            error = _config_error(cfg)
            if error:
                row["error"] = error
            else:
                assert prepared is not None
                row["base"] = float(prepared.base)
                try:
                    hv, gain, score, elapsed, stats = _run_config(ablate, prepared, cfg)
                    row["hv"] = float(hv)
                    row["gain"] = float(gain)
                    row["score"] = float(score)
                    row["elapsed"] = float(elapsed)
                    row.update(stats)
                except Exception as exc:
                    row["error"] = repr(exc)
            rows.append(row)
            _write_outputs(rows, out)
            print(
                f"{row['case']},seed={row['seed']},source_limit={row['source_limit']},"
                f"warm_c={row['warm_c']},rows={row['rows']},score={row['score']},"
                f"elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(
            f"best={best['case']},seed={best['seed']},source_limit={best['source_limit']},"
            f"warm_c={best['warm_c']},score={float(best['score']):.6f}",
            flush=True,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
