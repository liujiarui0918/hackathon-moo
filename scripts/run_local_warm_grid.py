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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SAMPLE_BUDGET = 100000
LAMBDA_POOL_SIZE = 1000
TRANSFER_Q_TARGET = 2
P_LAYERS = 3

DEFAULT_CASE = "04"
DEFAULT_WARM_C = 0.1
DEFAULT_LOCAL_RESTARTS = 6
DEFAULT_BROAD_WEIGHTS = 500
DEFAULT_BROAD_SHOTS = 100
DEFAULT_WARM_COUNT = 250
DEFAULT_WARM_SHOTS = 200
DEFAULT_CANDIDATE_SOURCE = "local"
DEFAULT_NEIGHBOR_SOURCE_LIMIT = 400
VERIFIED_CASE_SCORES = {
    "00": 473.398741,
    "01": 98.046167,
    "02": 232.729692,
    "03": 302.332699,
    "04": 253.517087,
    "05": 135.388737,
    "06": 247.503756,
    "07": 165.651423,
    "08": 96.919324,
    "09": 164.743738,
}

CSV_FIELDS = [
    "case",
    "case_suffix",
    "case_path",
    "seed",
    "warm_c",
    "local_restarts",
    "warm_count",
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
class GridConfig:
    seed: int
    warm_c: float
    local_restarts: int
    warm_count: int
    broad_weights: int
    broad_shots: int
    warm_shots: int
    candidate_source: str
    neighbor_source_limit: int

    @property
    def broad_rows(self) -> int:
        return int(self.broad_weights) * int(self.broad_shots)

    @property
    def warm_rows(self) -> int:
        return int(self.warm_count) * int(self.warm_shots)

    @property
    def rows(self) -> int:
        return self.broad_rows + self.warm_rows


@dataclass
class PreparedCase:
    problem: Any
    base: float
    pool: np.ndarray
    proj_j: np.ndarray
    proj_h: np.ndarray
    betas: np.ndarray
    gammas: np.ndarray


def _load_ablate_helpers() -> Any:
    from scripts import ablate_main1

    return ablate_main1


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


def _int_grid(values: list[str] | None, default: int) -> list[int]:
    return [int(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _case_values(values: list[str] | None) -> list[str]:
    return [str(v) for v in _dedupe_preserve(_split_values(values, [DEFAULT_CASE]))]


def _case_path(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    if raw.startswith("k5_grid4x5_"):
        return ROOT / "data" / "public" / f"{raw}.npz"
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


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
    return f"results/local_warm_grid_{stamp}"


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


def _config_error(cfg: GridConfig) -> str:
    errors: list[str] = []
    if cfg.rows != BASE_SAMPLE_BUDGET:
        errors.append(f"rows={cfg.rows}, expected {BASE_SAMPLE_BUDGET}")
    if not 0.0 <= float(cfg.warm_c) <= 1.0:
        errors.append("warm_c must be in [0, 1]")
    if int(cfg.local_restarts) < 1:
        errors.append("local_restarts must be >= 1")
    if cfg.candidate_source not in ("local", "broad_neighbors", "mixed"):
        errors.append("candidate_source must be local, broad_neighbors, or mixed")
    if int(cfg.neighbor_source_limit) < 1:
        errors.append("neighbor_source_limit must be >= 1")
    if int(cfg.broad_weights) < 0 or int(cfg.warm_count) < 0:
        errors.append("broad_weights and warm_count must be non-negative")
    if int(cfg.broad_weights) > LAMBDA_POOL_SIZE:
        errors.append(f"broad_weights must be <= {LAMBDA_POOL_SIZE}")
    if int(cfg.warm_count) > LAMBDA_POOL_SIZE:
        errors.append(f"warm_count must be <= {LAMBDA_POOL_SIZE}")
    if int(cfg.broad_weights) > 0 and int(cfg.broad_shots) <= 0:
        errors.append("broad_shots must be > 0 when broad_weights > 0")
    if int(cfg.warm_count) > 0 and int(cfg.warm_shots) <= 0:
        errors.append("warm_shots must be > 0 when warm_count > 0")
    if int(cfg.broad_shots) < 0 or int(cfg.warm_shots) < 0:
        errors.append("broad_shots and warm_shots must be non-negative")
    return "; ".join(errors)


def _base_row(case: Path, cfg: GridConfig) -> dict[str, Any]:
    return {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "case_path": str(case),
        "seed": int(cfg.seed),
        "warm_c": float(cfg.warm_c),
        "local_restarts": int(cfg.local_restarts),
        "warm_count": int(cfg.warm_count),
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
        "verified_score": VERIFIED_CASE_SCORES.get(_case_suffix(case), ""),
        "beats_verified": "",
        "error": "",
    }


def _make_configs(args: argparse.Namespace) -> list[GridConfig]:
    seeds = _int_grid(args.seed, 2026)
    warm_cs = _float_grid(args.warm_c, DEFAULT_WARM_C)
    local_restarts = _int_grid(args.local_restarts, DEFAULT_LOCAL_RESTARTS)
    warm_counts = _int_grid(args.warm_count, DEFAULT_WARM_COUNT)
    broad_weights = _int_grid(args.broad_weights, DEFAULT_BROAD_WEIGHTS)
    broad_shots = _int_grid(args.broad_shots, DEFAULT_BROAD_SHOTS)
    warm_shots = _int_grid(args.warm_shots, DEFAULT_WARM_SHOTS)
    candidate_sources = _dedupe_preserve(_split_values(args.candidate_source, [DEFAULT_CANDIDATE_SOURCE]))
    neighbor_source_limits = _int_grid(args.neighbor_source_limit, DEFAULT_NEIGHBOR_SOURCE_LIMIT)

    configs = []
    for seed, warm_c, restarts, warm_count, b_weights, b_shots, w_shots, source, neighbor_limit in product(
        seeds,
        warm_cs,
        local_restarts,
        warm_counts,
        broad_weights,
        broad_shots,
        warm_shots,
        candidate_sources,
        neighbor_source_limits,
    ):
        configs.append(
            GridConfig(
                seed=int(seed),
                warm_c=float(warm_c),
                local_restarts=int(restarts),
                warm_count=int(warm_count),
                broad_weights=int(b_weights),
                broad_shots=int(b_shots),
                warm_shots=int(w_shots),
                candidate_source=str(source),
                neighbor_source_limit=int(neighbor_limit),
            )
        )
    return configs


def _prepare_case(ablate: Any, case: Path) -> PreparedCase:
    problem = ablate.problem_from_npz(str(case))
    pool = ablate.load_weight_pool(int(problem.k), n=LAMBDA_POOL_SIZE, seed=2026).astype(np.float64)
    table = ablate.load_transfer_params_csv(
        str(ROOT / "transfer_data.csv"),
        q_target=TRANSFER_Q_TARGET,
        p_list=(P_LAYERS,),
    )
    if P_LAYERS not in table:
        raise ValueError(f"missing transfer params for p={P_LAYERS}")
    betas, gammas = table[P_LAYERS]
    proj_j = np.einsum("lk,km->lm", pool, problem.weights, optimize=False).astype(np.float64, copy=False)
    proj_h = np.einsum("lk,kn->ln", pool, problem.h, optimize=False).astype(np.float64, copy=False)
    base = ablate.baseline_hv(case, problem)
    return PreparedCase(
        problem=problem,
        base=float(base),
        pool=pool,
        proj_j=proj_j,
        proj_h=proj_h,
        betas=np.asarray(betas, dtype=np.float64),
        gammas=np.asarray(gammas, dtype=np.float64),
    )


def _objectives_for_spins(ablate: Any, prepared: PreparedCase, spins: np.ndarray) -> np.ndarray:
    lower, upper = ablate.objective_extrema(prepared.problem)
    energies = ablate._energy_batch_safe(
        np.asarray(spins, dtype=np.int8),
        prepared.problem.edges,
        prepared.problem.weights,
        prepared.problem.h,
    )
    return ablate.normalize_energies(energies, lower, upper)


def _assign_lambdas(objs: np.ndarray, pool: np.ndarray) -> np.ndarray:
    scalar = np.einsum(
        "ik,jk->ij",
        np.asarray(objs, dtype=np.float64),
        np.asarray(pool, dtype=np.float64),
        optimize=False,
    )
    return np.argmin(scalar, axis=1).astype(np.int64)


def _empty_candidates(prepared: PreparedCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, int(prepared.problem.n)), dtype=np.int8),
        np.zeros((0, int(prepared.problem.k)), dtype=np.float64),
        np.zeros((0,), dtype=np.int64),
    )


def _merge_candidate_front(
    ablate: Any,
    prepared: PreparedCase,
    spin_parts: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []
    for part in spin_parts:
        arr = np.asarray(part, dtype=np.int8)
        if arr.size:
            parts.append(arr)
    if not parts:
        return _empty_candidates(prepared)

    candidates = np.unique(np.vstack(parts).astype(np.int8, copy=False), axis=0)
    cand_objs = _objectives_for_spins(ablate, prepared, candidates)
    cand_nd = ablate.pg_non_dominated_indices(cand_objs)
    cand_spins = candidates[cand_nd]
    cand_objs = cand_objs[cand_nd]
    cand_lams = _assign_lambdas(cand_objs, prepared.pool)
    return cand_spins, cand_objs, cand_lams


def _broad_neighbor_candidates(
    ablate: Any,
    prepared: PreparedCase,
    broad_unique: np.ndarray,
    *,
    warm_count: int,
    source_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    unique = np.unique(np.asarray(broad_unique, dtype=np.int8), axis=0)
    if unique.size == 0:
        empty_spins, empty_objs, empty_lams = _empty_candidates(prepared)
        return empty_spins, empty_objs, empty_lams, 0

    objs = _objectives_for_spins(ablate, prepared, unique)
    nd = ablate.pg_non_dominated_indices(objs)
    nd_spins = unique[nd]
    nd_objs = objs[nd]
    nd_lams = _assign_lambdas(nd_objs, prepared.pool)

    base_count = min(int(nd_spins.shape[0]), max(int(warm_count), int(source_limit)))
    base_spins, _base_lams = ablate._select_diverse_warm_states(
        nd_spins,
        nd_objs,
        nd_lams,
        count=base_count,
    )
    bases = np.asarray(base_spins, dtype=np.int8)
    n = int(prepared.problem.n)
    neighbor_parts = [bases]
    if bases.size:
        flips = np.repeat(bases, n, axis=0)
        bit_ids = np.tile(np.arange(n, dtype=np.int64), int(bases.shape[0]))
        row_ids = np.arange(int(flips.shape[0]), dtype=np.int64)
        flips[row_ids, bit_ids] *= np.int8(-1)
        neighbor_parts.append(flips)

    candidates = np.unique(np.vstack(neighbor_parts).astype(np.int8, copy=False), axis=0)
    cand_objs = _objectives_for_spins(ablate, prepared, candidates)
    cand_nd = ablate.pg_non_dominated_indices(cand_objs)
    cand_spins = candidates[cand_nd]
    cand_objs = cand_objs[cand_nd]
    cand_lams = _assign_lambdas(cand_objs, prepared.pool)
    return cand_spins, cand_objs, cand_lams, int(candidates.shape[0])


def _run_config(ablate: Any, prepared: PreparedCase, cfg: GridConfig) -> tuple[float, float, float, float, int, int]:
    t0 = time.time()
    parts: list[np.ndarray] = []
    local_candidate_count = 0
    neighbor_candidate_count = 0
    broad_unique_parts: list[np.ndarray] = []

    if int(cfg.broad_weights) > 0:
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
            cand_spins, cand_objs, cand_lams, neighbor_candidate_count = _broad_neighbor_candidates(
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
            neighbor_spins, _neighbor_objs, _neighbor_lams, neighbor_candidate_count = _broad_neighbor_candidates(
                ablate,
                prepared,
                broad_unique,
                warm_count=int(cfg.warm_count),
                source_limit=int(cfg.neighbor_source_limit),
            )
            cand_spins, cand_objs, cand_lams = _merge_candidate_front(
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


def _print_dry_run(cases: list[Path], configs: list[GridConfig]) -> None:
    for case in cases:
        exists = "ok" if case.exists() else "missing"
        for cfg in configs:
            error = _config_error(cfg)
            status = "valid" if not error else f"invalid: {error}"
            print(
                f"{case.name} ({exists}): seed={cfg.seed}, warm_c={cfg.warm_c}, "
                f"local_restarts={cfg.local_restarts}, broad={cfg.broad_weights}x{cfg.broad_shots}, "
                f"warm={cfg.warm_count}x{cfg.warm_shots}, source={cfg.candidate_source}, "
                f"neighbor_limit={cfg.neighbor_source_limit}, rows={cfg.rows}, {status}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep legal local warm-start QAOA settings. Local descent candidates are used only as warm_bits; "
            "all returned rows come from MindQuantum sampling."
        )
    )
    parser.add_argument("--case", action="append", help="Case suffix like 04, stem, or .npz path. Repeatable.")
    parser.add_argument("--seed", action="append", help="Seed value(s), comma-separated or repeatable.")
    parser.add_argument("--warm-c", action="append", help="Warm-start c value(s), comma-separated or repeatable.")
    parser.add_argument("--local-restarts", action="append", help="Local candidate restart count(s).")
    parser.add_argument("--warm-count", action="append", help="Warm-start lambda/state count(s).")
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
    parser.add_argument("--dry-run", action="store_true", help="Print expanded grid and validation without sampling.")
    args = parser.parse_args()

    cases = [_case_path(case) for case in _case_values(args.case)]
    configs = _make_configs(args)
    out = args.out or _default_out()

    if args.dry_run:
        _print_dry_run(cases, configs)
        return

    missing_cases = [str(case) for case in cases if not case.exists()]
    if missing_cases:
        raise SystemExit(f"Missing case file(s): {', '.join(missing_cases)}")

    ablate = _load_ablate_helpers()
    rows: list[dict[str, Any]] = []
    for case in cases:
        valid_configs = [cfg for cfg in configs if not _config_error(cfg)]
        prepared: PreparedCase | None = None
        if valid_configs:
            prepared = _prepare_case(ablate, case)

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
                f"local_restarts={row['local_restarts']},broad={row['broad_weights']}x{row['broad_shots']},"
                f"warm={row['warm_count']}x{row['warm_shots']},source={row['candidate_source']},rows={row['rows']},"
                f"score={row['score']},elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(
            f"best={best['case']},seed={best['seed']},warm_c={best['warm_c']},"
            f"local_restarts={best['local_restarts']},warm={best['warm_count']}x{best['warm_shots']},"
            f"broad={best['broad_weights']}x{best['broad_shots']},source={best['candidate_source']},"
            f"score={float(best['score']):.6f}",
            flush=True,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
