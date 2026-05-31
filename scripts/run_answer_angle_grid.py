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

import answer  # noqa: E402
from run import baseline_hv  # noqa: E402
from utils import (  # noqa: E402
    hypervolume_pygmo,
    merge_non_dominated_pool,
    normalize_energies,
    objective_extrema,
    pg_non_dominated_indices,
    problem_from_npz,
)


BASE_SAMPLE_BUDGET = 100000
DEFAULT_CASE = "00"
DEFAULT_BETA_SCALE = 1.0
DEFAULT_GAMMA_SCALE = 1.0
CSV_FIELDS = [
    "case",
    "case_suffix",
    "seed",
    "beta_scale",
    "gamma_scale",
    "rows",
    "sample_used",
    "hv",
    "base",
    "gain",
    "score",
    "elapsed",
    "status",
    "error",
]


@dataclass(frozen=True)
class AngleConfig:
    seed: int | None
    beta_scale: float
    gamma_scale: float


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


def _float_grid(values: list[str] | None, default: float) -> list[float]:
    return [float(v) for v in _dedupe_preserve(_split_values(values, [default]))]


def _seed_grid(values: list[str] | None) -> list[int | None]:
    if values is None:
        return [None]
    out: list[int | None] = []
    for token in _dedupe_preserve(_split_values(values, ["default"])):
        raw = str(token).lower()
        out.append(None if raw in {"default", "none"} else int(token))
    return out


def _case_path(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
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
    return f"results/answer_angle_grid_{stamp}"


def _out_paths(out: str) -> tuple[Path, Path]:
    path = Path(out)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() == ".csv":
        return path, path.with_suffix(".json")
    if path.suffix.lower() == ".json":
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


def _energy_batch_safe(spins: np.ndarray, edges: np.ndarray, weights: np.ndarray, h: np.ndarray) -> np.ndarray:
    s = np.asarray(spins, dtype=np.float64)
    pair = s[:, edges[:, 0]] * s[:, edges[:, 1]]
    edge_term = np.einsum("sm,km->sk", pair, weights, optimize=False)
    linear_term = np.einsum("sn,kn->sk", s, h, optimize=False)
    return np.asarray(edge_term + linear_term, dtype=np.float64)


def _hv_from_spins(problem: Any, spins: np.ndarray) -> float:
    arr = np.asarray(spins, dtype=np.int8)
    lower, upper = objective_extrema(problem)
    nd_pool = np.zeros((0, int(problem.k)), dtype=np.float64)
    for start in range(0, int(arr.shape[0]), 4096):
        block = arr[start : start + 4096]
        objs = normalize_energies(
            _energy_batch_safe(block, problem.edges, problem.weights, problem.h),
            lower,
            upper,
        )
        nd_pool = merge_non_dominated_pool(nd_pool, objs[pg_non_dominated_indices(objs)])
    return 0.0 if nd_pool.size == 0 else float(hypervolume_pygmo(nd_pool))


def _make_configs(args: argparse.Namespace) -> list[AngleConfig]:
    return [
        AngleConfig(seed=seed, beta_scale=float(beta), gamma_scale=float(gamma))
        for seed, beta, gamma in product(
            _seed_grid(args.seed),
            _float_grid(args.beta_scale, DEFAULT_BETA_SCALE),
            _float_grid(args.gamma_scale, DEFAULT_GAMMA_SCALE),
        )
    ]


def _patch_transfer(beta_scale: float, gamma_scale: float) -> tuple[np.ndarray, np.ndarray]:
    p = int(answer.P_LAYERS)
    original_betas, original_gammas = answer._TRANSFER_TABLE[p]
    answer._TRANSFER_TABLE[p] = (
        np.asarray(original_betas, dtype=np.float64) * float(beta_scale),
        np.asarray(original_gammas, dtype=np.float64) * float(gamma_scale),
    )
    return original_betas, original_gammas


def _restore_transfer(original: tuple[np.ndarray, np.ndarray]) -> None:
    answer._TRANSFER_TABLE[int(answer.P_LAYERS)] = original


def _run_config(case: Path, problem: Any, cfg: AngleConfig, base: float) -> dict[str, Any]:
    row = {
        "case": case.name,
        "case_suffix": _case_suffix(case),
        "seed": "default" if cfg.seed is None else int(cfg.seed),
        "beta_scale": float(cfg.beta_scale),
        "gamma_scale": float(cfg.gamma_scale),
        "rows": "",
        "sample_used": "",
        "hv": "",
        "base": float(base),
        "gain": "",
        "score": "",
        "elapsed": "",
        "status": "",
        "error": "",
    }
    t0 = time.time()
    original = _patch_transfer(float(cfg.beta_scale), float(cfg.gamma_scale))
    try:
        result = answer.main1(problem) if cfg.seed is None else answer.main1(problem, rng_seed=int(cfg.seed))
        spins = np.asarray(result["sample_spins"], dtype=np.int8)
        rows = int(spins.shape[0])
        sample_used = int(result.get("sample_used", rows))
        hv = float(_hv_from_spins(problem, spins))
        budget_ok = rows == BASE_SAMPLE_BUDGET and sample_used == BASE_SAMPLE_BUDGET
        gain = max(float(hv - base), 0.0) if budget_ok else 0.0
        row.update(
            {
                "rows": rows,
                "sample_used": sample_used,
                "hv": hv,
                "gain": gain,
                "score": float(gain * 100000.0),
                "status": "ok" if budget_ok else "invalid_budget",
                "error": "" if budget_ok else f"expected {BASE_SAMPLE_BUDGET} rows/sample_used",
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "error", "error": repr(exc)})
    finally:
        _restore_transfer(original)
        row["elapsed"] = float(time.time() - t0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate answer.main1 with temporary transfer beta/gamma scaling."
    )
    parser.add_argument("--case", action="append", help="Case suffix, stem, or explicit NPZ path.")
    parser.add_argument("--seed", action="append", help="default/none or explicit rng_seed; repeatable/comma-separated.")
    parser.add_argument("--beta-scale", action="append", help="Transfer beta scale(s).")
    parser.add_argument("--gamma-scale", action="append", help="Transfer gamma scale(s).")
    parser.add_argument("--out", default=None, help="Output stem or .csv/.json path.")
    parser.add_argument("--run", action="store_true", help="Execute sampling. Default prints planned configs.")
    args = parser.parse_args()

    cases = [_case_path(c) for c in (args.case or [DEFAULT_CASE])]
    configs = _make_configs(args)
    out = args.out or _default_out()
    if not args.run:
        for case in cases:
            exists = "ok" if case.exists() else "missing"
            for cfg in configs:
                seed = "default" if cfg.seed is None else str(int(cfg.seed))
                print(
                    f"{case.name} ({exists}): seed={seed}, beta_scale={cfg.beta_scale}, gamma_scale={cfg.gamma_scale}",
                    flush=True,
                )
        return

    rows: list[dict[str, Any]] = []
    for case in cases:
        problem = problem_from_npz(str(case))
        base = float(baseline_hv(case, problem))
        for cfg in configs:
            row = _run_config(case, problem, cfg, base)
            rows.append(row)
            _write_outputs(rows, out)
            print(
                f"{row['case']},seed={row['seed']},beta={row['beta_scale']},gamma={row['gamma_scale']},"
                f"score={row['score']},elapsed={row['elapsed']},status={row['status']},error={row['error']}",
                flush=True,
            )

    csv_path, json_path = _write_outputs(rows, out)
    valid = [row for row in rows if row.get("score") not in ("", None)]
    if valid:
        best = max(valid, key=lambda row: float(row["score"]))
        print(f"best={best['case']},beta={best['beta_scale']},gamma={best['gamma_scale']},score={float(best['score']):.6f}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
