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

DEFAULT_CASE = "04"
DEFAULT_SEED = 2026
DEFAULT_CANDIDATE_SOURCE = "twohop"
DEFAULT_SOURCE_LIMIT = 120
DEFAULT_SELECTOR = "gap_frontier_cap"
DEFAULT_LAMBDA_CAP = 2
DEFAULT_MIN_DIST = 1.0e-3
DEFAULT_GAP_RESERVE = 48
DEFAULT_BETA_SCALE = 1.0
DEFAULT_GAMMA_SCALE = 1.0
DEFAULT_WARM_C = 0.125
DEFAULT_BROAD_WEIGHTS = 414
DEFAULT_BROAD_SHOTS = 100
DEFAULT_WARM_COUNT = 293
DEFAULT_WARM_SHOTS = 200

CSV_FIELDS = [
    "case",
    "case_suffix",
    "case_path",
    "seed",
    "candidate_source",
    "source_limit",
    "selector",
    "lambda_cap",
    "min_dist",
    "gap_reserve",
    "broad_beta_scale",
    "broad_gamma_scale",
    "warm_beta_scale",
    "warm_gamma_scale",
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
    "guidance_lambda_count",
    "gap_region_count",
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
    "selected_unique_lambdas",
    "selected_recommended_lambda_hits",
    "error",
]


@dataclass(frozen=True)
class GapPlan:
    path: Path | None
    centroids: np.ndarray
    recommended_lambda_ids: np.ndarray


@dataclass(frozen=True)
class GapGridConfig:
    seed: int
    candidate_source: str
    source_limit: int
    selector: str
    lambda_cap: int
    min_dist: float
    gap_reserve: int
    broad_beta_scale: float
    broad_gamma_scale: float
    warm_beta_scale: float
    warm_gamma_scale: float
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


def _int_grid(values: list[str] | None, default: int) -> list[int]:
    return [int(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _str_grid(values: list[str] | None, default: str) -> list[str]:
    return [str(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _case_values(values: list[str] | None) -> list[str]:
    return [str(v) for v in _dedupe_preserve(_split_values(values, [DEFAULT_CASE]))]


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _default_out() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/gap_guided_warm_grid_{stamp}"


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


def _load_gap_plan(path: str | os.PathLike[str] | None) -> GapPlan:
    if path is None or str(path).strip() == "":
        return GapPlan(None, np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64))
    p = _resolve(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Guidance JSON must be an object: {p}")
    centroids: list[np.ndarray] = []
    for region in data.get("gap_regions", []):
        if not isinstance(region, dict):
            continue
        if "centroid_objective_norm" not in region:
            continue
        centroids.append(np.asarray(region["centroid_objective_norm"], dtype=np.float64).reshape(-1))
    lambdas = [int(x) for x in data.get("recommended_lambda_ids", [])]
    return GapPlan(
        p,
        np.asarray(centroids, dtype=np.float64) if centroids else np.zeros((0, 0), dtype=np.float64),
        np.asarray(_dedupe_preserve(lambdas), dtype=np.int64),
    )


def _make_configs(args: argparse.Namespace) -> list[GapGridConfig]:
    seeds = _int_grid(args.seed, DEFAULT_SEED)
    candidate_sources = _str_grid(args.candidate_source, DEFAULT_CANDIDATE_SOURCE)
    source_limits = _int_grid(args.source_limit, DEFAULT_SOURCE_LIMIT)
    selectors = _str_grid(args.selector, DEFAULT_SELECTOR)
    lambda_caps = _int_grid(args.lambda_cap, DEFAULT_LAMBDA_CAP)
    min_dists = _float_grid(args.min_dist, DEFAULT_MIN_DIST)
    gap_reserves = _int_grid(args.gap_reserve, DEFAULT_GAP_RESERVE)
    broad_beta_scales = _float_grid(args.broad_beta_scale, DEFAULT_BETA_SCALE)
    broad_gamma_scales = _float_grid(args.broad_gamma_scale, DEFAULT_GAMMA_SCALE)
    warm_beta_scales = _float_grid(args.warm_beta_scale, DEFAULT_BETA_SCALE)
    warm_gamma_scales = _float_grid(args.warm_gamma_scale, DEFAULT_GAMMA_SCALE)
    warm_cs = _float_grid(args.warm_c, DEFAULT_WARM_C)
    warm_counts = _int_grid(args.warm_count, DEFAULT_WARM_COUNT)
    broad_weights = _int_grid(args.broad_weights, DEFAULT_BROAD_WEIGHTS)
    broad_shots = _int_grid(args.broad_shots, DEFAULT_BROAD_SHOTS)
    warm_shots = _int_grid(args.warm_shots, DEFAULT_WARM_SHOTS)

    configs: list[GapGridConfig] = []
    for seed, source, source_limit, selector, lambda_cap, min_dist, gap_reserve, broad_beta, broad_gamma, warm_beta, warm_gamma, warm_c, warm_count, b_weights, b_shots, w_shots in product(
        seeds,
        candidate_sources,
        source_limits,
        selectors,
        lambda_caps,
        min_dists,
        gap_reserves,
        broad_beta_scales,
        broad_gamma_scales,
        warm_beta_scales,
        warm_gamma_scales,
        warm_cs,
        warm_counts,
        broad_weights,
        broad_shots,
        warm_shots,
    ):
        configs.append(
            GapGridConfig(
                seed=int(seed),
                candidate_source=str(source),
                source_limit=int(source_limit),
                selector=str(selector),
                lambda_cap=int(lambda_cap),
                min_dist=float(min_dist),
                gap_reserve=int(gap_reserve),
                broad_beta_scale=float(broad_beta),
                broad_gamma_scale=float(broad_gamma),
                warm_beta_scale=float(warm_beta),
                warm_gamma_scale=float(warm_gamma),
                warm_c=float(warm_c),
                warm_count=int(warm_count),
                broad_weights=int(b_weights),
                broad_shots=int(b_shots),
                warm_shots=int(w_shots),
            )
        )
    return configs


def _config_error(cfg: GapGridConfig, plan: GapPlan) -> str:
    errors: list[str] = []
    if cfg.rows != BASE_SAMPLE_BUDGET:
        errors.append(f"rows={cfg.rows}, expected {BASE_SAMPLE_BUDGET}")
    if cfg.candidate_source not in ("sampled_nd", "onehop", "twohop"):
        errors.append("candidate_source must be sampled_nd, onehop, or twohop")
    if cfg.selector not in ("crowding", "gap_nearest", "gap_frontier_cap", "gap_blend", "frontier_cap"):
        errors.append("selector must be crowding, gap_nearest, gap_frontier_cap, gap_blend, or frontier_cap")
    if cfg.selector.startswith("gap_") and int(plan.centroids.shape[0]) == 0:
        errors.append("gap selector requires guidance centroids")
    if int(cfg.gap_reserve) < 0:
        errors.append("gap_reserve must be >= 0")
    for label, value in (
        ("broad_beta_scale", cfg.broad_beta_scale),
        ("broad_gamma_scale", cfg.broad_gamma_scale),
        ("warm_beta_scale", cfg.warm_beta_scale),
        ("warm_gamma_scale", cfg.warm_gamma_scale),
    ):
        if float(value) <= 0.0:
            errors.append(f"{label} must be > 0")
    if int(cfg.source_limit) < 1:
        errors.append("source_limit must be >= 1")
    if int(cfg.lambda_cap) < 1:
        errors.append("lambda_cap must be >= 1")
    if float(cfg.min_dist) < 0.0:
        errors.append("min_dist must be >= 0")
    if not 0.0 <= float(cfg.warm_c) <= 1.0:
        errors.append("warm_c must be in [0, 1]")
    if int(cfg.broad_weights) <= 0 or int(cfg.broad_shots) <= 0:
        errors.append("broad_weights and broad_shots must be > 0")
    if int(cfg.warm_count) <= 0 or int(cfg.warm_shots) <= 0:
        errors.append("warm_count and warm_shots must be > 0")
    if int(cfg.broad_weights) > LAMBDA_POOL_SIZE or int(cfg.warm_count) > LAMBDA_POOL_SIZE:
        errors.append(f"broad_weights and warm_count must be <= {LAMBDA_POOL_SIZE}")
    return "; ".join(errors)


def _base_row(case: Path, cfg: GapGridConfig, plan: GapPlan) -> dict[str, Any]:
    return {
        "case": case.name,
        "case_suffix": local_grid._case_suffix(case),
        "case_path": str(case),
        "seed": int(cfg.seed),
        "candidate_source": str(cfg.candidate_source),
        "source_limit": int(cfg.source_limit),
        "selector": str(cfg.selector),
        "lambda_cap": int(cfg.lambda_cap),
        "min_dist": float(cfg.min_dist),
        "gap_reserve": int(cfg.gap_reserve),
        "broad_beta_scale": float(cfg.broad_beta_scale),
        "broad_gamma_scale": float(cfg.broad_gamma_scale),
        "warm_beta_scale": float(cfg.warm_beta_scale),
        "warm_gamma_scale": float(cfg.warm_gamma_scale),
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
        "guidance_lambda_count": int(plan.recommended_lambda_ids.size),
        "gap_region_count": int(plan.centroids.shape[0]),
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
        "selected_unique_lambdas": "",
        "selected_recommended_lambda_hits": "",
        "error": "",
    }


def _crowding_order(objs: np.ndarray) -> np.ndarray:
    arr = np.asarray(objs, dtype=np.float64)
    m = int(arr.shape[0])
    if m <= 1:
        return np.arange(m, dtype=np.int64)
    k = int(arr.shape[1])
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    scaled = (arr - mins[None, :]) / np.maximum(maxs - mins, 1.0e-12)
    cd = np.zeros((m,), dtype=np.float64)
    anchors: list[int] = []
    for d in range(k):
        order = np.argsort(scaled[:, d])
        anchors.append(int(order[0]))
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        if m > 2:
            cd[order[1:-1]] += scaled[order[2:], d] - scaled[order[:-2], d]
    anchors = list(dict.fromkeys(anchors))
    anchor_mask = np.zeros((m,), dtype=bool)
    anchor_mask[np.asarray(anchors, dtype=np.int64)] = True
    inf_mask = np.isinf(cd).astype(np.int8)
    cd_key = np.where(np.isfinite(cd), cd, 0.0)
    rest = np.lexsort((np.arange(m, dtype=np.int64), -cd_key, -inf_mask))
    return np.concatenate([np.asarray(anchors, dtype=np.int64), rest[~anchor_mask[rest]]])


def _gap_orders(objs: np.ndarray, plan: GapPlan, rec_lams: np.ndarray, lams: np.ndarray) -> list[np.ndarray]:
    centroids = np.asarray(plan.centroids, dtype=np.float64)
    if centroids.size == 0:
        return []
    arr = np.asarray(objs, dtype=np.float64)
    rec_set = {int(x) for x in np.asarray(rec_lams, dtype=np.int64).reshape(-1)}
    orders: list[np.ndarray] = []
    for centroid in centroids:
        if centroid.shape[0] != arr.shape[1]:
            continue
        d = np.sqrt(np.sum(np.square(arr - centroid[None, :]), axis=1))
        hit = np.asarray([1 if int(lam) in rec_set else 0 for lam in lams], dtype=np.int8)
        orders.append(np.lexsort((np.arange(int(arr.shape[0]), dtype=np.int64), -hit, d)))
    return orders


def _interleave_orders(orders: list[np.ndarray], *, limit: int | None = None) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    pointers = [0 for _ in orders]
    while orders:
        progress = False
        for order_id, order in enumerate(orders):
            while pointers[order_id] < int(order.size) and int(order[pointers[order_id]]) in seen:
                pointers[order_id] += 1
            if pointers[order_id] >= int(order.size):
                continue
            value = int(order[pointers[order_id]])
            pointers[order_id] += 1
            if value in seen:
                continue
            selected.append(value)
            seen.add(value)
            progress = True
            if limit is not None and len(selected) >= int(limit):
                return selected
        if not progress:
            break
    return selected


def _select_by_order(
    spins: np.ndarray,
    objs: np.ndarray,
    lams: np.ndarray,
    order: Iterable[int],
    *,
    count: int,
    lambda_cap: int,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr_spins = np.asarray(spins, dtype=np.int8)
    arr_objs = np.asarray(objs, dtype=np.float64)
    arr_lams = np.asarray(lams, dtype=np.int64)
    if arr_spins.size == 0 or arr_objs.size == 0:
        return arr_spins, arr_lams

    idx_order = [int(x) for x in order]
    selected: list[int] = []
    selected_mask = np.zeros((int(arr_objs.shape[0]),), dtype=bool)
    lam_counts = np.zeros((max(int(arr_lams.max()) + 1, 1),), dtype=np.int16)
    min_d2 = np.full((int(arr_objs.shape[0]),), np.inf, dtype=np.float64)

    def add(i: int) -> None:
        ii = int(i)
        selected.append(ii)
        selected_mask[ii] = True
        lam_counts[int(arr_lams[ii])] += 1
        diff = arr_objs - arr_objs[ii][None, :]
        d2 = np.einsum("ij,ij->i", diff, diff, optimize=False)
        min_d2[:] = np.minimum(min_d2, d2)
        min_d2[ii] = 0.0

    cap = max(1, int(lambda_cap))
    thr0 = max(float(min_dist), 0.0)
    for thr2 in [thr0 * thr0, (thr0 * 0.3) ** 2, (thr0 * 0.1) ** 2, 0.0]:
        for idx in idx_order:
            if len(selected) >= int(count):
                break
            ii = int(idx)
            if selected_mask[ii] or int(lam_counts[int(arr_lams[ii])]) >= cap:
                continue
            if thr2 > 0.0 and float(min_d2[ii]) < thr2:
                continue
            add(ii)
        if len(selected) >= int(count):
            break

    if len(selected) < int(count):
        for idx in idx_order:
            if len(selected) >= int(count):
                break
            ii = int(idx)
            if selected_mask[ii]:
                continue
            if int(lam_counts[int(arr_lams[ii])]) < cap:
                add(ii)

    if not selected:
        selected = [0]
    while len(selected) < int(count):
        selected.append(selected[len(selected) % len(selected)])

    sel = np.asarray(selected[: int(count)], dtype=np.int64)
    return arr_spins[sel], arr_lams[sel]


def _select_crowding_warm_states(
    ablate: Any,
    spins: np.ndarray,
    objs: np.ndarray,
    lams: np.ndarray,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    return ablate._select_diverse_warm_states(
        np.asarray(spins, dtype=np.int8),
        np.asarray(objs, dtype=np.float64),
        np.asarray(lams, dtype=np.int64),
        count=int(count),
    )


def _select_warm_states(
    ablate: Any,
    spins: np.ndarray,
    objs: np.ndarray,
    lams: np.ndarray,
    plan: GapPlan,
    *,
    selector: str,
    count: int,
    lambda_cap: int,
    min_dist: float,
    gap_reserve: int,
) -> tuple[np.ndarray, np.ndarray]:
    if selector == "crowding":
        return _select_crowding_warm_states(ablate, spins, objs, lams, count=int(count))
    if selector == "frontier_cap":
        order = _crowding_order(objs)
    elif selector == "gap_nearest":
        gap_orders = _gap_orders(objs, plan, plan.recommended_lambda_ids, lams)
        order = np.asarray(_interleave_orders(gap_orders), dtype=np.int64)
        if int(order.size) < int(objs.shape[0]):
            seen = set(int(x) for x in order)
            rest = [int(x) for x in _crowding_order(objs) if int(x) not in seen]
            order = np.asarray([*order.tolist(), *rest], dtype=np.int64)
    elif selector == "gap_frontier_cap":
        gap_orders = _gap_orders(objs, plan, plan.recommended_lambda_ids, lams)
        gap_prefix = _interleave_orders(gap_orders, limit=max(int(count) * 2, int(count) + 16))
        seen = set(gap_prefix)
        order = np.asarray([*gap_prefix, *[int(x) for x in _crowding_order(objs) if int(x) not in seen]], dtype=np.int64)
    elif selector == "gap_blend":
        reserve = min(max(int(gap_reserve), 0), int(count))
        gap_selected: list[int] = []
        if reserve > 0:
            gap_orders = _gap_orders(objs, plan, plan.recommended_lambda_ids, lams)
            gap_selected = _interleave_orders(gap_orders, limit=max(reserve * 4, reserve + 16))
        gap_spins = np.zeros((0, int(spins.shape[1])), dtype=np.int8)
        gap_lams = np.zeros((0,), dtype=np.int64)
        if gap_selected and reserve > 0:
            gap_spins, gap_lams = _select_by_order(
                spins,
                objs,
                lams,
                gap_selected,
                count=reserve,
                lambda_cap=int(lambda_cap),
                min_dist=float(min_dist),
            )
        remaining = int(count) - int(gap_lams.size)
        if remaining <= 0:
            return gap_spins, gap_lams
        selected_set = {tuple(row.tolist()) for row in np.asarray(gap_spins, dtype=np.int8)}
        mask = np.asarray([tuple(row.tolist()) not in selected_set for row in np.asarray(spins, dtype=np.int8)], dtype=bool)
        fill_spins, fill_lams = _select_crowding_warm_states(
            ablate,
            np.asarray(spins, dtype=np.int8)[mask],
            np.asarray(objs, dtype=np.float64)[mask],
            np.asarray(lams, dtype=np.int64)[mask],
            count=remaining,
        )
        return (
            np.vstack([gap_spins, np.asarray(fill_spins, dtype=np.int8)]).astype(np.int8, copy=False),
            np.concatenate([gap_lams, np.asarray(fill_lams, dtype=np.int64)]).astype(np.int64, copy=False),
        )
    else:
        raise ValueError(f"unsupported selector={selector}")

    return _select_by_order(
        spins,
        objs,
        lams,
        order,
        count=int(count),
        lambda_cap=int(lambda_cap),
        min_dist=float(min_dist),
    )


def _candidate_bank(
    ablate: Any,
    prepared: local_grid.PreparedCase,
    broad_unique: np.ndarray,
    *,
    candidate_source: str,
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
        empty_spins, empty_objs, empty_lams = local_grid._empty_candidates(prepared)
        return empty_spins, empty_objs, empty_lams, stats

    objs = local_grid._objectives_for_spins(ablate, prepared, unique)
    nd = ablate.pg_non_dominated_indices(objs)
    nd_spins = unique[nd]
    nd_objs = objs[nd]
    nd_lams = local_grid._assign_lambdas(nd_objs, prepared.pool)
    stats["broad_nd_count"] = int(nd_spins.shape[0])
    if candidate_source == "sampled_nd":
        stats["base_count"] = int(nd_spins.shape[0])
        stats["candidate_count"] = int(nd_spins.shape[0])
        stats["candidate_nd_count"] = int(nd_spins.shape[0])
        return nd_spins, nd_objs, nd_lams, stats

    base_count = min(int(nd_spins.shape[0]), int(source_limit))
    stats["base_count"] = int(base_count)
    if base_count <= 0:
        empty_spins, empty_objs, empty_lams = local_grid._empty_candidates(prepared)
        return empty_spins, empty_objs, empty_lams, stats

    base_spins, _ = ablate._select_diverse_warm_states(
        nd_spins,
        nd_objs,
        nd_lams,
        count=int(base_count),
    )
    bases = np.asarray(base_spins, dtype=np.int8)
    n = int(prepared.problem.n)
    parts = [bases]

    one_flips = np.repeat(bases, n, axis=0)
    bit_ids = np.tile(np.arange(n, dtype=np.int64), int(bases.shape[0]))
    row_ids = np.arange(int(one_flips.shape[0]), dtype=np.int64)
    one_flips[row_ids, bit_ids] *= np.int8(-1)
    parts.append(one_flips)

    if candidate_source == "twohop":
        first, second = np.triu_indices(n, k=1)
        if int(first.size) > 0:
            pair_count = int(first.size)
            two_flips = np.repeat(bases, pair_count, axis=0)
            first_ids = np.tile(first.astype(np.int64, copy=False), int(bases.shape[0]))
            second_ids = np.tile(second.astype(np.int64, copy=False), int(bases.shape[0]))
            row_ids = np.arange(int(two_flips.shape[0]), dtype=np.int64)
            two_flips[row_ids, first_ids] *= np.int8(-1)
            two_flips[row_ids, second_ids] *= np.int8(-1)
            parts.append(two_flips)

    candidates = np.unique(np.vstack(parts).astype(np.int8, copy=False), axis=0)
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
    cfg: GapGridConfig,
    plan: GapPlan,
    cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[float, float, float, float, dict[str, int]]:
    t0 = time.time()
    if cache is None:
        cache = {}

    broad_key = (
        "broad",
        str(prepared.problem.name),
        int(cfg.seed),
        int(cfg.broad_weights),
        int(cfg.broad_shots),
        float(cfg.broad_beta_scale),
        float(cfg.broad_gamma_scale),
    )
    if broad_key in cache:
        broad_spins, broad_unique = cache[broad_key]
    else:
        broad_ids = np.arange(int(cfg.broad_weights), dtype=np.int64)
        broad_spins, broad_unique_blocks, *_ = ablate._sample_round(
            prepared.problem,
            broad_ids,
            int(cfg.broad_shots),
            prepared.proj_j,
            prepared.proj_h,
            np.asarray(prepared.betas, dtype=np.float64) * float(cfg.broad_beta_scale),
            np.asarray(prepared.gammas, dtype=np.float64) * float(cfg.broad_gamma_scale),
            seed=int(cfg.seed),
        )
        broad_unique = np.unique(
            np.vstack([np.asarray(block, dtype=np.int8) for block in broad_unique_blocks]).astype(np.int8, copy=False),
            axis=0,
        )
        cache[broad_key] = (np.asarray(broad_spins, dtype=np.int8), np.asarray(broad_unique, dtype=np.int8))

    candidate_key = (
        "candidate",
        broad_key,
        str(cfg.candidate_source),
        int(cfg.source_limit),
    )
    if candidate_key in cache:
        cand_spins, cand_objs, cand_lams, cached_stats = cache[candidate_key]
        stats = dict(cached_stats)
    else:
        cand_spins, cand_objs, cand_lams, stats = _candidate_bank(
            ablate,
            prepared,
            broad_unique,
            candidate_source=str(cfg.candidate_source),
            source_limit=int(cfg.source_limit),
        )
        cache[candidate_key] = (
            np.asarray(cand_spins, dtype=np.int8),
            np.asarray(cand_objs, dtype=np.float64),
            np.asarray(cand_lams, dtype=np.int64),
            dict(stats),
        )
    warm_spins, warm_lams = _select_warm_states(
        ablate,
        cand_spins,
        cand_objs,
        cand_lams,
        plan,
        selector=str(cfg.selector),
        count=int(cfg.warm_count),
        lambda_cap=int(cfg.lambda_cap),
        min_dist=float(cfg.min_dist),
        gap_reserve=int(cfg.gap_reserve),
    )
    rec_set = {int(x) for x in plan.recommended_lambda_ids.reshape(-1)}
    stats["selected_unique_lambdas"] = int(np.unique(warm_lams).size)
    stats["selected_recommended_lambda_hits"] = int(sum(1 for lid in warm_lams if int(lid) in rec_set))

    warm_bits, warm_lams = ablate._materialize_warm_selection(
        warm_spins,
        warm_lams,
        count=int(cfg.warm_count),
    )
    warm_sampled, *_ = ablate._sample_round(
        prepared.problem,
        np.asarray(warm_lams, dtype=np.int64),
        int(cfg.warm_shots),
        prepared.proj_j,
        prepared.proj_h,
        np.asarray(prepared.betas, dtype=np.float64) * float(cfg.warm_beta_scale),
        np.asarray(prepared.gammas, dtype=np.float64) * float(cfg.warm_gamma_scale),
        seed=int(cfg.seed + 10000),
        warm_bits=warm_bits,
        warm_c=float(cfg.warm_c),
    )
    spins = np.vstack([np.asarray(broad_spins, dtype=np.int8), np.asarray(warm_sampled, dtype=np.int8)]).astype(
        np.int8,
        copy=False,
    )
    if int(spins.shape[0]) != BASE_SAMPLE_BUDGET:
        raise RuntimeError(f"produced {spins.shape[0]} rows, expected {BASE_SAMPLE_BUDGET}")

    hv = float(ablate._hv_from_spins_safe(prepared.problem, spins))
    gain = max(float(hv - prepared.base), 0.0)
    score = float(gain * 100000.0)
    elapsed = float(time.time() - t0)
    return hv, gain, score, elapsed, stats


def _print_dry_run(cases: list[Path], configs: list[GapGridConfig], plan: GapPlan) -> None:
    for case in cases:
        exists = "ok" if case.exists() else "missing"
        for cfg in configs:
            error = _config_error(cfg, plan)
            status = "valid" if not error else f"invalid: {error}"
            print(
                f"{case.name} ({exists}): seed={cfg.seed}, source={cfg.candidate_source}, "
                f"source_limit={cfg.source_limit}, selector={cfg.selector}, lambda_cap={cfg.lambda_cap}, "
                f"min_dist={cfg.min_dist}, gap_reserve={cfg.gap_reserve}, "
                f"broad_scale={cfg.broad_beta_scale}/{cfg.broad_gamma_scale}, "
                f"warm_scale={cfg.warm_beta_scale}/{cfg.warm_gamma_scale}, "
                f"broad={cfg.broad_weights}x{cfg.broad_shots}, "
                f"warm={cfg.warm_count}x{cfg.warm_shots}, rows={cfg.rows}, {status}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run legal gap-guided warm-start probes. Gap plans affect only warm-start selection; "
            "all returned rows are still produced by MindQuantum sampling."
        )
    )
    parser.add_argument("--case", action="append", help="Case suffix like 04, stem, or .npz path. Repeatable.")
    parser.add_argument("--guidance-json", default=None, help="Redacted gap plan JSON.")
    parser.add_argument("--seed", action="append", help="Seed value(s), comma-separated or repeatable.")
    parser.add_argument("--candidate-source", action="append", help="sampled_nd, onehop, or twohop.")
    parser.add_argument("--source-limit", action="append", help="Broad sampled ND base limit(s).")
    parser.add_argument("--selector", action="append", help="crowding, gap_nearest, gap_frontier_cap, gap_blend, or frontier_cap.")
    parser.add_argument("--lambda-cap", action="append", help="Max selected warm states per lambda.")
    parser.add_argument("--min-dist", action="append", help="Objective-space minimum distance between selected warm states.")
    parser.add_argument("--gap-reserve", action="append", help="For gap_blend, number of warm slots reserved for gap-nearest anchors.")
    parser.add_argument("--broad-beta-scale", action="append", help="Scale broad-arm transfer betas.")
    parser.add_argument("--broad-gamma-scale", action="append", help="Scale broad-arm transfer gammas.")
    parser.add_argument("--warm-beta-scale", action="append", help="Scale warm-arm transfer betas.")
    parser.add_argument("--warm-gamma-scale", action="append", help="Scale warm-arm transfer gammas.")
    parser.add_argument("--warm-c", action="append", help="Warm-start c value(s).")
    parser.add_argument("--warm-count", action="append", help="Warm-start lambda/state count(s).")
    parser.add_argument("--broad-weights", action="append", help="Broad no-warm lambda count(s).")
    parser.add_argument("--broad-shots", action="append", help="Shots per broad lambda.")
    parser.add_argument("--warm-shots", action="append", help="Shots per warm-start state.")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path. Writes both CSV and JSON.")
    parser.add_argument("--run", action="store_true", help="Actually run sampling. Default prints expanded configs only.")
    args = parser.parse_args()

    plan = _load_gap_plan(args.guidance_json)
    cases = [local_grid._case_path(case) for case in _case_values(args.case)]
    configs = _make_configs(args)
    out = args.out or _default_out()

    if not args.run:
        _print_dry_run(cases, configs, plan)
        return

    missing_cases = [str(case) for case in cases if not case.exists()]
    if missing_cases:
        raise SystemExit(f"Missing case file(s): {', '.join(missing_cases)}")

    ablate = local_grid._load_ablate_helpers()
    rows: list[dict[str, Any]] = []
    for case in cases:
        valid_configs = [cfg for cfg in configs if not _config_error(cfg, plan)]
        prepared: local_grid.PreparedCase | None = None
        cache: dict[tuple[Any, ...], Any] = {}
        if valid_configs:
            prepared = local_grid._prepare_case(ablate, case)

        for cfg in configs:
            row = _base_row(case, cfg, plan)
            error = _config_error(cfg, plan)
            if error:
                row["error"] = error
            else:
                assert prepared is not None
                row["base"] = float(prepared.base)
                try:
                    hv, gain, score, elapsed, stats = _run_config(ablate, prepared, cfg, plan, cache)
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
                f"{row['case']},seed={row['seed']},source={row['candidate_source']},selector={row['selector']},"
                f"lambda_cap={row['lambda_cap']},min_dist={row['min_dist']},gap_reserve={row['gap_reserve']},score={row['score']},"
                f"elapsed={row['elapsed']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(
            f"best={best['case']},source={best['candidate_source']},selector={best['selector']},"
            f"lambda_cap={best['lambda_cap']},min_dist={best['min_dist']},score={float(best['score']):.6f}",
            flush=True,
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
