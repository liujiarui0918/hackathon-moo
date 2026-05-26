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

from scripts import run_local_warm_grid as local_grid

BASE_SAMPLE_BUDGET = local_grid.BASE_SAMPLE_BUDGET
LAMBDA_POOL_SIZE = local_grid.LAMBDA_POOL_SIZE
TRANSFER_Q_TARGET = local_grid.TRANSFER_Q_TARGET
P_LAYERS = local_grid.P_LAYERS

DEFAULT_CASE = "09"
DEFAULT_SEED = 2031
DEFAULT_WARM_C = 0.1
DEFAULT_LOCAL_RESTARTS = 6
DEFAULT_BROAD_WEIGHTS = 500
DEFAULT_BROAD_SHOTS = 100
DEFAULT_WARM_COUNT = 250
DEFAULT_WARM_SHOTS = 200
DEFAULT_CANDIDATE_SOURCE = "mixed"
DEFAULT_NEIGHBOR_SOURCE_LIMIT = 400
DEFAULT_SELECTOR = "hv_proxy"
DEFAULT_HV_PREFILTER = 500
DEFAULT_MIN_DIST = 0.025
DEFAULT_LAMBDA_CAP = 4

CSV_FIELDS = [
    "case",
    "case_suffix",
    "case_path",
    "seed",
    "selector",
    "candidate_source",
    "neighbor_source_limit",
    "warm_c",
    "local_restarts",
    "warm_count",
    "broad_weights",
    "broad_shots",
    "warm_shots",
    "hv_prefilter",
    "min_dist",
    "lambda_cap",
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
    "merged_candidate_count",
    "selected_candidate_hv",
    "error",
]


@dataclass(frozen=True)
class HvWarmConfig:
    seed: int
    selector: str
    candidate_source: str
    neighbor_source_limit: int
    warm_c: float
    local_restarts: int
    warm_count: int
    broad_weights: int
    broad_shots: int
    warm_shots: int
    hv_prefilter: int
    min_dist: float
    lambda_cap: int

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


def _int_grid(values: list[str] | None, default: int) -> list[int]:
    return [int(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _str_grid(values: list[str] | None, default: str) -> list[str]:
    return [str(v) for v in _dedupe_preserve(_split_values(values, [default]))]


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
    return f"results/hv_warm_grid_{stamp}"


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


def _make_configs(args: argparse.Namespace) -> list[HvWarmConfig]:
    configs = []
    for (
        seed,
        selector,
        source,
        neighbor_limit,
        warm_c,
        restarts,
        warm_count,
        broad_weights,
        broad_shots,
        warm_shots,
        hv_prefilter,
        min_dist,
        lambda_cap,
    ) in product(
        _int_grid(args.seed, DEFAULT_SEED),
        _str_grid(args.selector, DEFAULT_SELECTOR),
        _str_grid(args.candidate_source, DEFAULT_CANDIDATE_SOURCE),
        _int_grid(args.neighbor_source_limit, DEFAULT_NEIGHBOR_SOURCE_LIMIT),
        _float_grid(args.warm_c, DEFAULT_WARM_C),
        _int_grid(args.local_restarts, DEFAULT_LOCAL_RESTARTS),
        _int_grid(args.warm_count, DEFAULT_WARM_COUNT),
        _int_grid(args.broad_weights, DEFAULT_BROAD_WEIGHTS),
        _int_grid(args.broad_shots, DEFAULT_BROAD_SHOTS),
        _int_grid(args.warm_shots, DEFAULT_WARM_SHOTS),
        _int_grid(args.hv_prefilter, DEFAULT_HV_PREFILTER),
        _float_grid(args.min_dist, DEFAULT_MIN_DIST),
        _int_grid(args.lambda_cap, DEFAULT_LAMBDA_CAP),
    ):
        configs.append(
            HvWarmConfig(
                seed=int(seed),
                selector=str(selector),
                candidate_source=str(source),
                neighbor_source_limit=int(neighbor_limit),
                warm_c=float(warm_c),
                local_restarts=int(restarts),
                warm_count=int(warm_count),
                broad_weights=int(broad_weights),
                broad_shots=int(broad_shots),
                warm_shots=int(warm_shots),
                hv_prefilter=int(hv_prefilter),
                min_dist=float(min_dist),
                lambda_cap=int(lambda_cap),
            )
        )
    return configs


def _config_error(cfg: HvWarmConfig) -> str:
    errors: list[str] = []
    if cfg.rows != BASE_SAMPLE_BUDGET:
        errors.append(f"rows={cfg.rows}, expected {BASE_SAMPLE_BUDGET}")
    if cfg.selector not in ("crowding", "hv_proxy"):
        errors.append("selector must be crowding or hv_proxy")
    if cfg.candidate_source not in ("local", "broad_neighbors", "mixed"):
        errors.append("candidate_source must be local, broad_neighbors, or mixed")
    if not 0.0 <= float(cfg.warm_c) <= 1.0:
        errors.append("warm_c must be in [0, 1]")
    if int(cfg.local_restarts) < 1:
        errors.append("local_restarts must be >= 1")
    if int(cfg.neighbor_source_limit) < 1:
        errors.append("neighbor_source_limit must be >= 1")
    if int(cfg.warm_count) < 0 or int(cfg.broad_weights) < 0:
        errors.append("warm_count and broad_weights must be non-negative")
    if int(cfg.broad_weights) > LAMBDA_POOL_SIZE or int(cfg.warm_count) > LAMBDA_POOL_SIZE:
        errors.append(f"broad_weights and warm_count must be <= {LAMBDA_POOL_SIZE}")
    if int(cfg.broad_weights) > 0 and int(cfg.broad_shots) <= 0:
        errors.append("broad_shots must be > 0 when broad_weights > 0")
    if int(cfg.warm_count) > 0 and int(cfg.warm_shots) <= 0:
        errors.append("warm_shots must be > 0 when warm_count > 0")
    if int(cfg.hv_prefilter) < int(cfg.warm_count):
        errors.append("hv_prefilter must be >= warm_count")
    if float(cfg.min_dist) < 0.0:
        errors.append("min_dist must be >= 0")
    if int(cfg.lambda_cap) < 1:
        errors.append("lambda_cap must be >= 1")
    return "; ".join(errors)


def _base_row(case: Path, cfg: HvWarmConfig) -> dict[str, Any]:
    return {
        "case": case.name,
        "case_suffix": local_grid._case_suffix(case),
        "case_path": str(case),
        "seed": int(cfg.seed),
        "selector": str(cfg.selector),
        "candidate_source": str(cfg.candidate_source),
        "neighbor_source_limit": int(cfg.neighbor_source_limit),
        "warm_c": float(cfg.warm_c),
        "local_restarts": int(cfg.local_restarts),
        "warm_count": int(cfg.warm_count),
        "broad_weights": int(cfg.broad_weights),
        "broad_shots": int(cfg.broad_shots),
        "warm_shots": int(cfg.warm_shots),
        "hv_prefilter": int(cfg.hv_prefilter),
        "min_dist": float(cfg.min_dist),
        "lambda_cap": int(cfg.lambda_cap),
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
        "merged_candidate_count": "",
        "selected_candidate_hv": "",
        "error": "",
    }


def _hv_safe(ablate: Any, points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    ref = np.full((int(arr.shape[1]),), 1.01, dtype=np.float64)
    arr = arr[np.all(arr <= ref[None, :], axis=1)]
    if arr.size == 0:
        return 0.0
    arr = np.unique(arr, axis=0)
    arr = arr[ablate.pg_non_dominated_indices(arr)]
    if int(arr.shape[0]) == 1:
        return float(np.prod(ref - arr[0]))
    return float(ablate.hypervolume_pygmo(arr))


def _prefilter_hv_proxy_indices(objs: np.ndarray, *, count: int, limit: int) -> np.ndarray:
    objs = np.asarray(objs, dtype=np.float64)
    m = int(objs.shape[0])
    if m == 0 or m <= int(limit):
        return np.arange(m, dtype=np.int64)
    ref = np.full((int(objs.shape[1]),), 1.01, dtype=np.float64)
    clipped = np.maximum(ref[None, :] - objs, 0.0)
    volume = np.prod(clipped, axis=1)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    scaled = (objs - mins[None, :]) / np.maximum(maxs - mins, 1e-12)
    cd = np.zeros((m,), dtype=np.float64)
    anchors = []
    for d in range(int(objs.shape[1])):
        order = np.argsort(scaled[:, d])
        anchors.append(int(order[0]))
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        if m > 2:
            cd[order[1:-1]] += scaled[order[2:], d] - scaled[order[:-2], d]

    selected: list[int] = []
    seen: set[int] = set()
    for idx in anchors:
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    cd_key = np.where(np.isfinite(cd), cd, 0.0)
    order = np.lexsort((np.arange(m, dtype=np.int64), -cd_key, -volume))
    for idx in order:
        val = int(idx)
        if val in seen:
            continue
        selected.append(val)
        seen.add(val)
        if len(selected) >= max(int(limit), int(count)):
            break
    return np.asarray(selected[: max(int(limit), int(count))], dtype=np.int64)


def _select_hv_proxy_warm_states(
    ablate: Any,
    spins: np.ndarray,
    objs: np.ndarray,
    lams: np.ndarray,
    *,
    count: int,
    prefilter: int,
    min_dist: float,
    lambda_cap: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    spins = np.asarray(spins, dtype=np.int8)
    objs = np.asarray(objs, dtype=np.float64)
    lams = np.asarray(lams, dtype=np.int64)
    if spins.size == 0 or objs.size == 0:
        return spins, lams, 0.0

    keep = _prefilter_hv_proxy_indices(objs, count=int(count), limit=int(prefilter))
    cand_spins = spins[keep]
    cand_objs = objs[keep]
    cand_lams = lams[keep]
    m = int(cand_objs.shape[0])
    k = int(cand_objs.shape[1])
    ref = np.full((k,), 1.01, dtype=np.float64)
    point_volume = np.prod(np.maximum(ref[None, :] - cand_objs, 0.0), axis=1)
    mins = cand_objs.min(axis=0)
    maxs = cand_objs.max(axis=0)
    scaled = (cand_objs - mins[None, :]) / np.maximum(maxs - mins, 1e-12)

    selected: list[int] = []
    seen = np.zeros((m,), dtype=bool)
    lam_counts = np.zeros((max(int(cand_lams.max()) + 1, 1),), dtype=np.int16)
    min_d2 = np.full((m,), np.inf, dtype=np.float64)
    lambda_cap = max(1, int(lambda_cap))

    def can_use(idx: int) -> bool:
        return int(lam_counts[int(cand_lams[idx])]) < lambda_cap

    def add(idx: int) -> None:
        ii = int(idx)
        selected.append(ii)
        seen[ii] = True
        lam_counts[int(cand_lams[ii])] += 1
        d = scaled - scaled[ii]
        d2 = np.einsum("ij,ij->i", d, d, optimize=False)
        min_d2[:] = np.minimum(min_d2, d2)
        min_d2[ii] = 0.0

    for d in range(k):
        idx = int(np.argmin(cand_objs[:, d]))
        if not seen[idx] and can_use(idx):
            add(idx)
        if len(selected) >= int(count):
            break

    thresholds = [
        float(min_dist) ** 2,
        (float(min_dist) * 0.35) ** 2,
        (float(min_dist) * 0.1) ** 2,
        0.0,
    ]
    for thr2 in thresholds:
        while len(selected) < int(count):
            best_idx = None
            best_key: tuple[float, float, int] | None = None
            for idx in range(m):
                if seen[idx] or not can_use(idx):
                    continue
                if thr2 > 0.0 and float(min_d2[idx]) < thr2:
                    continue
                novelty = 1.0 if not selected else float(np.sqrt(max(min_d2[idx], 0.0)))
                key = (float(point_volume[idx]) * max(novelty, 1e-6), novelty, -idx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            if best_idx is None:
                break
            add(best_idx)
        if len(selected) >= int(count):
            break

    if len(selected) < int(count):
        order = np.lexsort((np.arange(m, dtype=np.int64), -point_volume))
        for idx in order:
            if len(selected) >= int(count):
                break
            ii = int(idx)
            if not seen[ii] and can_use(ii):
                add(ii)

    if not selected:
        selected = [0]
    while len(selected) < int(count):
        selected.append(selected[len(selected) % len(selected)])

    sel = np.asarray(selected[: int(count)], dtype=np.int64)
    selected_hv = _hv_safe(ablate, cand_objs[sel])
    return cand_spins[sel], cand_lams[sel], selected_hv


def _select_warm_states(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    cand_spins: np.ndarray,
    cand_objs: np.ndarray,
    cand_lams: np.ndarray,
    cfg: HvWarmConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    if cfg.selector == "crowding":
        warm_spins, warm_lams = ablate._select_diverse_warm_states(
            cand_spins,
            cand_objs,
            cand_lams,
            count=int(cfg.warm_count),
        )
        selected_objs = local_grid._objectives_for_spins(ablate, prepared, warm_spins)
        return warm_spins, warm_lams, _hv_safe(ablate, selected_objs)
    return _select_hv_proxy_warm_states(
        ablate,
        cand_spins,
        cand_objs,
        cand_lams,
        count=int(cfg.warm_count),
        prefilter=int(cfg.hv_prefilter),
        min_dist=float(cfg.min_dist),
        lambda_cap=int(cfg.lambda_cap),
    )


def _candidate_bank(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    cfg: HvWarmConfig,
    broad_unique_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    local_count = 0
    neighbor_count = 0
    if cfg.candidate_source == "local":
        cand_spins, cand_objs, cand_lams = ablate._multiobjective_local_candidates(
            prepared.problem,
            prepared.pool,
            prepared.proj_j,
            prepared.proj_h,
            seed=int(cfg.seed + 70000),
            restarts=int(cfg.local_restarts),
        )
        local_count = int(np.asarray(cand_spins).shape[0])
        return cand_spins, cand_objs, cand_lams, local_count, neighbor_count

    if not broad_unique_parts:
        raise ValueError(f"{cfg.candidate_source} candidate source requires broad sampling rows")
    broad_unique = np.unique(np.vstack(broad_unique_parts).astype(np.int8, copy=False), axis=0)

    if cfg.candidate_source == "broad_neighbors":
        cand_spins, cand_objs, cand_lams, neighbor_count = local_grid._broad_neighbor_candidates(
            ablate,
            prepared,
            broad_unique,
            warm_count=int(cfg.warm_count),
            source_limit=int(cfg.neighbor_source_limit),
        )
        local_count = int(np.asarray(cand_spins).shape[0])
        return cand_spins, cand_objs, cand_lams, local_count, int(neighbor_count)

    local_spins, _local_objs, _local_lams = ablate._multiobjective_local_candidates(
        prepared.problem,
        prepared.pool,
        prepared.proj_j,
        prepared.proj_h,
        seed=int(cfg.seed + 70000),
        restarts=int(cfg.local_restarts),
    )
    local_count = int(np.asarray(local_spins).shape[0])
    neighbor_spins, _neighbor_objs, _neighbor_lams, neighbor_count = local_grid._broad_neighbor_candidates(
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
    return cand_spins, cand_objs, cand_lams, local_count, int(neighbor_count)


def _run_config(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    cfg: HvWarmConfig,
) -> tuple[float, float, float, float, dict[str, Any]]:
    t0 = time.time()
    parts: list[np.ndarray] = []
    broad_unique_parts: list[np.ndarray] = []
    stats: dict[str, Any] = {
        "local_candidate_count": 0,
        "neighbor_candidate_count": 0,
        "merged_candidate_count": 0,
        "selected_candidate_hv": 0.0,
    }

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
        cand_spins, cand_objs, cand_lams, local_count, neighbor_count = _candidate_bank(
            ablate,
            prepared,
            cfg,
            broad_unique_parts,
        )
        stats["local_candidate_count"] = int(local_count)
        stats["neighbor_candidate_count"] = int(neighbor_count)
        stats["merged_candidate_count"] = int(np.asarray(cand_spins).shape[0])
        warm_spins, warm_lams, selected_hv = _select_warm_states(
            ablate,
            prepared,
            cand_spins,
            cand_objs,
            cand_lams,
            cfg,
        )
        stats["selected_candidate_hv"] = float(selected_hv)
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
    return hv, gain, score, elapsed, stats


def _print_dry_run(cases: list[Path], configs: list[HvWarmConfig]) -> None:
    for case in cases:
        exists = "ok" if case.exists() else "missing"
        for cfg in configs:
            error = _config_error(cfg)
            status = "valid" if not error else f"invalid: {error}"
            print(
                f"{case.name} ({exists}): seed={cfg.seed}, selector={cfg.selector}, "
                f"source={cfg.candidate_source}, neighbor_limit={cfg.neighbor_source_limit}, "
                f"warm_c={cfg.warm_c}, broad={cfg.broad_weights}x{cfg.broad_shots}, "
                f"warm={cfg.warm_count}x{cfg.warm_shots}, hv_prefilter={cfg.hv_prefilter}, "
                f"min_dist={cfg.min_dist}, lambda_cap={cfg.lambda_cap}, rows={cfg.rows}, {status}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype HV/volume-proxy warm-state selection. Classical candidates are used only to choose "
            "warm_bits and lambda ids; returned samples still come from MindQuantum sampling."
        )
    )
    parser.add_argument("--case", action="append", help="Case suffix like 09, stem, or .npz path. Repeatable.")
    parser.add_argument("--seed", action="append", help="Seed value(s), comma-separated or repeatable.")
    parser.add_argument("--selector", action="append", help="Selector(s): hv_proxy or crowding.")
    parser.add_argument(
        "--candidate-source",
        action="append",
        help="Warm candidate source(s): local, broad_neighbors, or mixed. Comma-separated or repeatable.",
    )
    parser.add_argument("--neighbor-source-limit", action="append", help="Broad-neighbor source ND point limit(s).")
    parser.add_argument("--warm-c", action="append", help="Warm-start c value(s), comma-separated or repeatable.")
    parser.add_argument("--local-restarts", action="append", help="Local candidate restart count(s).")
    parser.add_argument("--warm-count", action="append", help="Warm-start lambda/state count(s).")
    parser.add_argument("--broad-weights", action="append", help="Broad no-warm lambda count(s).")
    parser.add_argument("--broad-shots", action="append", help="Shots per broad lambda.")
    parser.add_argument("--warm-shots", action="append", help="Shots per warm-start state.")
    parser.add_argument("--hv-prefilter", action="append", help="Candidate cap before HV-proxy greedy selection.")
    parser.add_argument("--min-dist", action="append", help="Minimum scaled objective distance for early greedy picks.")
    parser.add_argument("--lambda-cap", action="append", help="Max selected warm states per lambda before fallback repeats.")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print grid and validation only.")
    parser.add_argument("--run", dest="dry_run", action="store_false", help="Actually run MindQuantum sampling.")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    cases = [local_grid._case_path(case) for case in _case_values(args.case)]
    configs = _make_configs(args)
    out = args.out or _default_out()

    if args.dry_run:
        _print_dry_run(cases, configs)
        print("dry-run only; pass --run to execute MindQuantum sampling", flush=True)
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
                f"{row['case']},seed={row['seed']},selector={row['selector']},"
                f"source={row['candidate_source']},neighbor_limit={row['neighbor_source_limit']},"
                f"warm_c={row['warm_c']},rows={row['rows']},score={row['score']},"
                f"elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(
            f"best={best['case']},seed={best['seed']},selector={best['selector']},"
            f"source={best['candidate_source']},neighbor_limit={best['neighbor_source_limit']},"
            f"warm_c={best['warm_c']},score={float(best['score']):.6f}",
            flush=True,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
