from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_local_warm_grid as local_grid

BASE_SAMPLE_BUDGET = local_grid.BASE_SAMPLE_BUDGET
LAMBDA_POOL_SIZE = local_grid.LAMBDA_POOL_SIZE
TRANSFER_Q_TARGET = local_grid.TRANSFER_Q_TARGET
P_LAYERS = local_grid.P_LAYERS

DEFAULT_CASE = local_grid.DEFAULT_CASE
DEFAULT_SEED = 2026
DEFAULT_WARM_C = local_grid.DEFAULT_WARM_C
DEFAULT_LOCAL_RESTARTS = local_grid.DEFAULT_LOCAL_RESTARTS
DEFAULT_BROAD_START = 500
DEFAULT_BROAD_WEIGHTS = 500
DEFAULT_BROAD_SHOTS = 100
DEFAULT_WARM_COUNT = local_grid.DEFAULT_WARM_COUNT
DEFAULT_WARM_SHOTS = local_grid.DEFAULT_WARM_SHOTS
DEFAULT_CANDIDATE_SOURCE = local_grid.DEFAULT_CANDIDATE_SOURCE
DEFAULT_NEIGHBOR_SOURCE_LIMIT = local_grid.DEFAULT_NEIGHBOR_SOURCE_LIMIT

CSV_FIELDS = [
    "case",
    "case_suffix",
    "case_path",
    "seed",
    "warm_c",
    "local_restarts",
    "warm_count",
    "broad_start",
    "broad_stop",
    "broad_weights",
    "broad_shots",
    "warm_shots",
    "candidate_source",
    "neighbor_source_limit",
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
    "local_candidate_count",
    "neighbor_candidate_count",
    "verified_score",
    "beats_verified",
    "error",
]


@dataclass(frozen=True)
class OffsetGridConfig:
    seed: int
    warm_c: float
    local_restarts: int
    warm_count: int
    broad_start: int
    broad_weights: int
    broad_shots: int
    warm_shots: int
    candidate_source: str
    neighbor_source_limit: int

    @property
    def broad_stop(self) -> int:
        return int(self.broad_start) + int(self.broad_weights)

    @property
    def broad_rows(self) -> int:
        return int(self.broad_weights) * int(self.broad_shots)

    @property
    def warm_rows(self) -> int:
        return int(self.warm_count) * int(self.warm_shots)

    @property
    def rows(self) -> int:
        return self.broad_rows + self.warm_rows


def _int_grid(values: list[str] | None, default: int) -> list[int]:
    return [int(v) for v in local_grid._dedupe_preserve(local_grid._split_values(values, [default]))]


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in local_grid._dedupe_preserve(local_grid._split_values(values, [default]))]


def _case_values(values: list[str] | None) -> list[str]:
    return [str(v) for v in local_grid._dedupe_preserve(local_grid._split_values(values, [DEFAULT_CASE]))]


def _source_grid(values: list[str] | None, default: str) -> list[str]:
    return [str(v) for v in local_grid._dedupe_preserve(local_grid._split_values(values, [default]))]


def _default_out() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/lambda_offset_grid_{stamp}"


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
        json.dump(rows, fh, ensure_ascii=False, indent=2, sort_keys=True, default=local_grid._json_default)
    return csv_path, json_path


def _make_configs(args: argparse.Namespace) -> list[OffsetGridConfig]:
    seeds = _int_grid(args.seed, DEFAULT_SEED)
    warm_cs = _float_grid(args.warm_c, DEFAULT_WARM_C)
    local_restarts = _int_grid(args.local_restarts, DEFAULT_LOCAL_RESTARTS)
    warm_counts = _int_grid(args.warm_count, DEFAULT_WARM_COUNT)
    broad_starts = _int_grid(args.broad_start, DEFAULT_BROAD_START)
    broad_weights = _int_grid(args.broad_weights, DEFAULT_BROAD_WEIGHTS)
    broad_shots = _int_grid(args.broad_shots, DEFAULT_BROAD_SHOTS)
    warm_shots = _int_grid(args.warm_shots, DEFAULT_WARM_SHOTS)
    candidate_sources = _source_grid(args.candidate_source, DEFAULT_CANDIDATE_SOURCE)
    neighbor_source_limits = _int_grid(args.neighbor_source_limit, DEFAULT_NEIGHBOR_SOURCE_LIMIT)

    configs: list[OffsetGridConfig] = []
    for seed, warm_c, restarts, warm_count, start, b_weights, b_shots, w_shots, source, neighbor_limit in product(
        seeds,
        warm_cs,
        local_restarts,
        warm_counts,
        broad_starts,
        broad_weights,
        broad_shots,
        warm_shots,
        candidate_sources,
        neighbor_source_limits,
    ):
        configs.append(
            OffsetGridConfig(
                seed=int(seed),
                warm_c=float(warm_c),
                local_restarts=int(restarts),
                warm_count=int(warm_count),
                broad_start=int(start),
                broad_weights=int(b_weights),
                broad_shots=int(b_shots),
                warm_shots=int(w_shots),
                candidate_source=str(source),
                neighbor_source_limit=int(neighbor_limit),
            )
        )
    return configs


def _config_error(cfg: OffsetGridConfig) -> str:
    local_cfg = local_grid.GridConfig(
        seed=int(cfg.seed),
        warm_c=float(cfg.warm_c),
        local_restarts=int(cfg.local_restarts),
        warm_count=int(cfg.warm_count),
        broad_weights=int(cfg.broad_weights),
        broad_shots=int(cfg.broad_shots),
        warm_shots=int(cfg.warm_shots),
        candidate_source=str(cfg.candidate_source),
        neighbor_source_limit=int(cfg.neighbor_source_limit),
    )
    errors = [err for err in [local_grid._config_error(local_cfg)] if err]
    if int(cfg.broad_start) < 0:
        errors.append("broad_start must be non-negative")
    if int(cfg.broad_stop) > LAMBDA_POOL_SIZE:
        errors.append(f"broad range [{cfg.broad_start}, {cfg.broad_stop}) exceeds pool size {LAMBDA_POOL_SIZE}")
    if int(cfg.broad_weights) == 1000 and int(cfg.broad_shots) == 50:
        errors.append("1000x50 broad scan is intentionally blocked; use an offset window instead")
    return "; ".join(errors)


def _base_row(case: Path, cfg: OffsetGridConfig) -> dict[str, Any]:
    return {
        "case": case.name,
        "case_suffix": local_grid._case_suffix(case),
        "case_path": str(case),
        "seed": int(cfg.seed),
        "warm_c": float(cfg.warm_c),
        "local_restarts": int(cfg.local_restarts),
        "warm_count": int(cfg.warm_count),
        "broad_start": int(cfg.broad_start),
        "broad_stop": int(cfg.broad_stop),
        "broad_weights": int(cfg.broad_weights),
        "broad_shots": int(cfg.broad_shots),
        "warm_shots": int(cfg.warm_shots),
        "candidate_source": str(cfg.candidate_source),
        "neighbor_source_limit": int(cfg.neighbor_source_limit),
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
        "local_candidate_count": "",
        "neighbor_candidate_count": "",
        "verified_score": local_grid.VERIFIED_CASE_SCORES.get(local_grid._case_suffix(case), ""),
        "beats_verified": "",
        "error": "",
    }


def _run_config(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    cfg: OffsetGridConfig,
) -> tuple[float, float, float, float, int, int]:
    t0 = time.time()
    parts: list[np.ndarray] = []
    local_candidate_count = 0
    neighbor_candidate_count = 0
    broad_unique_parts: list[np.ndarray] = []

    if int(cfg.broad_weights) > 0:
        broad_ids = np.arange(int(cfg.broad_start), int(cfg.broad_stop), dtype=np.int64)
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
        parts.append(np.asarray(broad_spins, dtype=np.int8))
        broad_unique_parts.extend(np.asarray(block, dtype=np.int8) for block in broad_unique_blocks)

    if int(cfg.warm_count) > 0:
        if cfg.candidate_source == "local":
            cand_spins, cand_objs, cand_lams = ablate._multiobjective_local_candidates(
                prepared.problem,
                prepared.pool,
                prepared.proj_j,
                prepared.proj_h,
                seed=int(cfg.seed + 70000),
                restarts=int(cfg.local_restarts),
            )
            local_candidate_count = int(np.asarray(cand_spins).shape[0])
        elif cfg.candidate_source == "broad_neighbors":
            if not broad_unique_parts:
                raise ValueError("broad_neighbors candidate source requires broad sampling rows")
            broad_unique = np.unique(np.vstack(broad_unique_parts).astype(np.int8, copy=False), axis=0)
            cand_spins, cand_objs, cand_lams, neighbor_candidate_count = local_grid._broad_neighbor_candidates(
                ablate,
                prepared,
                broad_unique,
                warm_count=int(cfg.warm_count),
                source_limit=int(cfg.neighbor_source_limit),
            )
            local_candidate_count = int(np.asarray(cand_spins).shape[0])
        elif cfg.candidate_source == "mixed":
            if not broad_unique_parts:
                raise ValueError("mixed candidate source requires broad sampling rows")
            local_spins, _local_objs, _local_lams = ablate._multiobjective_local_candidates(
                prepared.problem,
                prepared.pool,
                prepared.proj_j,
                prepared.proj_h,
                seed=int(cfg.seed + 70000),
                restarts=int(cfg.local_restarts),
            )
            local_candidate_count = int(np.asarray(local_spins).shape[0])
            broad_unique = np.unique(np.vstack(broad_unique_parts).astype(np.int8, copy=False), axis=0)
            neighbor_spins, _neighbor_objs, _neighbor_lams, neighbor_candidate_count = local_grid._broad_neighbor_candidates(
                ablate,
                prepared,
                broad_unique,
                warm_count=int(cfg.warm_count),
                source_limit=int(cfg.neighbor_source_limit),
            )
            cand_spins, cand_objs, cand_lams = local_grid._merge_candidate_front(
                ablate,
                prepared,
                (local_spins, neighbor_spins),
            )
        else:
            raise ValueError(f"unsupported candidate_source={cfg.candidate_source}")

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

    if not parts:
        raise ValueError("configuration produced no quantum sampling arms")

    spins = np.vstack(parts).astype(np.int8, copy=False)
    if int(spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise RuntimeError(f"produced {spins.shape[0]} rows, expected {BASE_SAMPLE_BUDGET}")

    hv = float(ablate._hv_from_spins_safe(prepared.problem, spins))
    gain = max(float(hv - prepared.base), 0.0)
    score = float(gain * 100000.0)
    elapsed = float(time.time() - t0)
    return hv, gain, score, elapsed, local_candidate_count, neighbor_candidate_count


def _print_dry_run(cases: list[Path], configs: list[OffsetGridConfig]) -> None:
    for case in cases:
        exists = "ok" if case.exists() else "missing"
        for cfg in configs:
            error = _config_error(cfg)
            status = "valid" if not error else f"invalid: {error}"
            print(
                f"{case.name} ({exists}): seed={cfg.seed}, warm_c={cfg.warm_c}, "
                f"local_restarts={cfg.local_restarts}, broad=[{cfg.broad_start},{cfg.broad_stop})x{cfg.broad_shots}, "
                f"warm={cfg.warm_count}x{cfg.warm_shots}, source={cfg.candidate_source}, "
                f"neighbor_limit={cfg.neighbor_source_limit}, rows={cfg.rows}, {status}",
                flush=True,
            )


def _run_grid(cases: list[Path], configs: list[OffsetGridConfig], out: str) -> None:
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
                    hv, gain, score, elapsed, local_candidate_count, neighbor_candidate_count = _run_config(
                        ablate,
                        prepared,
                        cfg,
                    )
                    row["hv"] = float(hv)
                    row["gain"] = float(gain)
                    row["score"] = float(score)
                    row["elapsed"] = float(elapsed)
                    row["local_candidate_count"] = int(local_candidate_count)
                    row["neighbor_candidate_count"] = int(neighbor_candidate_count)
                    verified = row["verified_score"]
                    row["beats_verified"] = "" if verified == "" else bool(float(score) > float(verified))
                except Exception as exc:
                    row["error"] = repr(exc)
            rows.append(row)
            _write_outputs(rows, out)
            print(
                f"{row['case']},seed={row['seed']},warm_c={row['warm_c']},"
                f"local_restarts={row['local_restarts']},broad=[{row['broad_start']},{row['broad_stop']})"
                f"x{row['broad_shots']},warm={row['warm_count']}x{row['warm_shots']},"
                f"source={row['candidate_source']},rows={row['rows']},score={row['score']},"
                f"elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(
            f"best={best['case']},seed={best['seed']},warm_c={best['warm_c']},"
            f"local_restarts={best['local_restarts']},broad=[{best['broad_start']},{best['broad_stop']})"
            f"x{best['broad_shots']},warm={best['warm_count']}x{best['warm_shots']},"
            f"source={best['candidate_source']},score={float(best['score']):.6f}",
            flush=True,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep offset broad lambda windows for local warm-start QAOA. Classical candidates only choose "
            "warm-start states/lambdas; every final sample row comes from MindQuantum sampling."
        )
    )
    parser.add_argument("--case", action="append", help="Case suffix like 04, stem, or .npz path. Repeatable.")
    parser.add_argument("--seed", action="append", help="Seed value(s), comma-separated or repeatable.")
    parser.add_argument("--warm-c", action="append", help="Warm-start c value(s), comma-separated or repeatable.")
    parser.add_argument("--local-restarts", action="append", help="Local candidate restart count(s).")
    parser.add_argument("--warm-count", action="append", help="Warm-start lambda/state count(s).")
    parser.add_argument("--broad-start", action="append", help="Broad lambda window start(s), 0-indexed.")
    parser.add_argument("--broad-weights", action="append", help="Broad no-warm lambda count(s).")
    parser.add_argument("--broad-shots", action="append", help="Shots per broad lambda.")
    parser.add_argument("--warm-shots", action="append", help="Shots per warm-start state.")
    parser.add_argument(
        "--candidate-source",
        action="append",
        help="Warm candidate source(s): local, broad_neighbors, or mixed. Comma-separated or repeatable.",
    )
    parser.add_argument(
        "--neighbor-source-limit",
        action="append",
        help="Broad-neighbor source ND point limit(s).",
    )
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--dry-run", dest="run", action="store_false", help="Print expanded grid without sampling.")
    run_group.add_argument("--run", dest="run", action="store_true", help="Opt in to MindQuantum sampling.")
    parser.set_defaults(run=False)
    args = parser.parse_args()

    cases = [local_grid._case_path(case) for case in _case_values(args.case)]
    configs = _make_configs(args)
    out = args.out or _default_out()

    if not args.run:
        _print_dry_run(cases, configs)
        return

    _run_grid(cases, configs, out)


if __name__ == "__main__":
    main()
