from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_weight_pool, problem_from_npz


REDACT_KEYS = {
    "state_index",
    "anchor_state_index",
    "representative_state_indices",
    "exact_index",
    "anchor_exact_index",
    "representative_exact_indices",
    "exact_indices",
}


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _resolve_case(case: str) -> Path:
    raw = str(case).strip()
    if raw.endswith(".npz") or "/" in raw or "\\" in raw:
        return _resolve(raw)
    return ROOT / "data" / "public" / f"k5_grid4x5_{int(raw):02d}.npz"


def _case_suffix(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return _project_path(value)
    return value


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return data


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact(v) for k, v in value.items() if str(k) not in REDACT_KEYS}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _dedupe(values: Iterable[int], *, limit: int | None = None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _centroid(region: dict[str, Any]) -> np.ndarray | None:
    for key in ("centroid_objective_norm", "anchor_objective_norm", "objective_norm"):
        if key in region:
            arr = np.asarray(region[key], dtype=np.float64).reshape(-1)
            if arr.size:
                return arr
    return None


def _lambda_neighbors(pool: np.ndarray, center_id: int, *, count: int) -> list[int]:
    center = int(center_id)
    if center < 0 or center >= int(pool.shape[0]):
        return []
    diff = pool - pool[center][None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff, optimize=False)
    return [int(x) for x in np.argsort(dist2)[: max(1, int(count))]]


def _centroid_lambda_ids(pool: np.ndarray, centroid: np.ndarray, *, count: int) -> list[int]:
    c = np.asarray(centroid, dtype=np.float64).reshape(-1)
    target = np.maximum(1.01 - c, 1.0e-9)
    target = target / max(float(np.sum(target)), 1.0e-12)
    diff = pool - target[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff, optimize=False)
    return [int(x) for x in np.argsort(dist2)[: max(1, int(count))]]


def _guidance_lambda_ids(guidance_regions: list[dict[str, Any]], centroid: np.ndarray, *, radius: float) -> list[int]:
    found: list[tuple[float, int]] = []
    for region in guidance_regions:
        if "lambda_id" not in region:
            continue
        gc = _centroid(region)
        if gc is None or gc.shape != centroid.shape:
            continue
        dist = float(np.linalg.norm(gc - centroid))
        if dist <= float(radius):
            found.append((dist, int(region["lambda_id"])))
    found.sort(key=lambda item: (item[0], item[1]))
    return _dedupe([lid for _, lid in found])


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    case_path = _resolve_case(args.case)
    if not case_path.exists():
        raise FileNotFoundError(case_path)
    problem = problem_from_npz(str(case_path))
    pool = load_weight_pool(int(problem.k), n=int(args.pool_size), seed=int(args.pool_seed)).astype(np.float64)

    coverage_path = _resolve(args.coverage_json)
    coverage = _load_json(coverage_path)
    guidance_path = _resolve(args.exact_guidance_json) if args.exact_guidance_json else None
    guidance = _load_json(guidance_path) if guidance_path else {}

    coverage_regions = list(coverage.get("missing_regions", []))[: max(1, int(args.top_regions))]
    guidance_regions = list(guidance.get("missing_regions", [])) if isinstance(guidance.get("missing_regions", []), list) else []
    global_guidance_lambdas = _dedupe(
        int(region["lambda_id"])
        for region in guidance_regions
        if isinstance(region, dict) and "lambda_id" in region
    )

    regions: list[dict[str, Any]] = []
    global_recommendations: list[int] = []
    families: list[dict[str, Any]] = []
    for rank, raw_region in enumerate(coverage_regions, start=1):
        if not isinstance(raw_region, dict):
            continue
        centroid = _centroid(raw_region)
        if centroid is None:
            continue
        matched = _guidance_lambda_ids(
            guidance_regions,
            centroid,
            radius=float(args.guidance_match_radius),
        )
        centroid_ids = _centroid_lambda_ids(pool, centroid, count=max(1, int(args.lambda_neighbors)))
        center_ids = _dedupe([*matched, *centroid_ids, *global_guidance_lambdas[:2]])

        region_recs: list[int] = []
        for source, center_id in [("matched_or_centroid", lid) for lid in center_ids]:
            members = _lambda_neighbors(pool, center_id, count=max(1, int(args.lambda_neighbors)))
            region_recs.extend(members)
            families.append(
                {
                    "source": source,
                    "region_rank": int(rank),
                    "center_lambda_id": int(center_id),
                    "member_lambda_ids": members,
                }
            )
        region_recs = _dedupe(region_recs, limit=max(1, int(args.recommend_count)))
        global_recommendations.extend(region_recs)

        safe_region = _redact(raw_region) if args.redact_state_indices else dict(raw_region)
        regions.append(
            {
                "rank": int(rank),
                "centroid_objective_norm": [float(x) for x in centroid],
                "coverage_score_sum": float(raw_region.get("coverage_score_sum", 0.0)),
                "approx_marginal_hv_gain_sum": float(raw_region.get("approx_marginal_hv_gain_sum", 0.0)),
                "action": str(raw_region.get("action", "")),
                "matched_guidance_lambda_ids": matched,
                "centroid_lambda_ids": centroid_ids,
                "recommended_lambda_ids": region_recs,
                "redacted_region": safe_region,
            }
        )

    recommended = _dedupe(global_recommendations, limit=max(1, int(args.recommend_count)))
    return {
        "case": case_path.name,
        "case_suffix": _case_suffix(case_path),
        "case_path": _project_path(case_path),
        "created_at_unix": float(time.time()),
        "note": (
            "Diagnostic plan only. It contains objective-space centroids and lambda IDs for legal warm-start "
            "experiments; it must not be loaded by answer.py."
        ),
        "sources": {
            "coverage_json": _project_path(coverage_path),
            "exact_guidance_json": None if guidance_path is None else _project_path(guidance_path),
        },
        "redaction": {
            "enabled": bool(args.redact_state_indices),
            "removed_keys": sorted(REDACT_KEYS) if args.redact_state_indices else [],
        },
        "lambda_pool": {
            "pool_size": int(args.pool_size),
            "pool_seed": int(args.pool_seed),
            "lambda_neighbors": int(args.lambda_neighbors),
            "guidance_match_radius": float(args.guidance_match_radius),
        },
        "recommended_lambda_ids": recommended,
        "gap_regions": regions,
        "lambda_families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a redacted objective-space gap/lambda plan for legal warm-start experiments."
    )
    parser.add_argument("--case", required=True, help="Case suffix such as 04, or explicit NPZ path.")
    parser.add_argument("--coverage-json", required=True, help="Coverage report from sample_coverage_gap_report.py.")
    parser.add_argument("--exact-guidance-json", default=None, help="Optional exact guidance JSON used only for lambda IDs.")
    parser.add_argument("--top-regions", type=int, default=12, help="Number of coverage regions to keep.")
    parser.add_argument("--lambda-neighbors", type=int, default=6, help="Neighbor lambdas per center lambda.")
    parser.add_argument("--recommend-count", type=int, default=48, help="Max global lambda recommendations.")
    parser.add_argument("--guidance-match-radius", type=float, default=0.35, help="Objective L2 radius for matching guidance regions.")
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--pool-seed", type=int, default=2026)
    parser.add_argument("--redact-state-indices", action="store_true", help="Remove exact state/index fields from output.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    args = parser.parse_args()

    payload = build_plan(args)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "case": payload["case"],
                "regions": len(payload["gap_regions"]),
                "recommended_lambda_ids": len(payload["recommended_lambda_ids"]),
                "top_lambdas": payload["recommended_lambda_ids"][:12],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
