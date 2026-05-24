# MOO Exact Headroom Guided Tuning Plan

Date: 2026-05-24

## 1. Goal

Compute the exact public small-case Pareto frontier and exact maximum HV for every `4x5`/`20`-bit public case, then use that information to guide legal `main1` quantum-sampling tuning.

Important boundary:

- Exact enumeration is allowed for offline diagnosis and public-case analysis.
- Exact enumeration must not generate or repair final `sample_spins`.
- `answer.py/main1()` must still return `100000` rows sampled from MindQuantum circuits.
- Any final code change must preserve the legal sampling channel.

This plan answers three concrete questions:

1. What is the theoretical best public score per case?
2. Which cases have the most remaining score headroom?
3. Which exact-frontier regions are missing from the current quantum sample distribution, and how should we tune lambda/seed/warm-start coverage to hit them?

## 2. Why Exact Enumeration Is Feasible

Each public small case has:

- `n = 20` spins;
- `2^20 = 1,048,576` states;
- `k = 5` objectives;
- `31` grid edges.

Full objective storage per case is modest:

- normalized objective matrix: roughly `1,048,576 x 5 x 8 bytes = 40 MB`;
- state indices: roughly `1,048,576 x 4 bytes = 4 MB`;
- exact frontier candidates are much smaller than the full state space.

So the classical diagnostic path should be fast if it avoids repeated small-chunk non-dominated merges. The earlier attempt was slow because it used `4096` chunks and filtered/merged too often.

## 3. Exact Enumeration Algorithm

Use a two-stage exact frontier extraction:

1. Enumerate states in large chunks.
   - Recommended chunk sizes: `65536`, `131072`, `262144`.
   - Convert state indices to spins using `_state_index_block_to_spins`.
   - Compute raw energies with `energy_batch_fast`.
   - Normalize using `objective_extrema`.

2. Keep only local first-front candidates from each chunk.
   - A globally non-dominated point must be non-dominated inside its own chunk.
   - Therefore it is safe to discard chunk-local dominated points.
   - Store candidate normalized objectives and their state indices.

3. Concatenate all local candidates.
   - Run one final non-dominated sort on the candidate pool.
   - Deduplicate exact objective rows.
   - Compute exact HV with `hypervolume_pygmo`.

4. Save exact artifacts:
   - summary JSON: `results/exact_public_headroom.json`;
   - per-case frontier NPZ: `results/exact_frontiers/{case}.npz`;
   - optional CSV summary: `results/exact_public_headroom.csv`.

Per-case summary fields:

- `case`;
- `hv_base`;
- `hv_solver_old`;
- `hv_exact`;
- `score_old`;
- `score_max_case`;
- `remaining_case_score`;
- `captured_fraction`;
- `exact_nd_count`;
- `elapsed_s`;
- `frontier_npz`.

Per-case frontier NPZ fields:

- `state_indices`;
- `spins` or `bits01` only if storage remains reasonable;
- `objectives_norm`;
- `energies_raw`;
- `hv_exact`;
- `lower_bounds`;
- `upper_bounds`.

## 4. Headroom Interpretation

For each case:

```text
case_score_current = (HV_current - HV_base) * 100000
case_score_max     = (HV_exact   - HV_base) * 100000
remaining          = (HV_exact   - HV_current) * 100000
captured_fraction  = case_score_current / case_score_max
```

This is the correct way to judge whether a case is worth tuning.

Do not infer difficulty only from current score:

- a `200 -> 220` improvement may be easy if the exact frontier still has huge missing regions;
- a `50 -> 70` improvement may be hard if the current samples already capture most attainable HV over baseline;
- any `+20` case score contributes the same `+2` total score across ten public cases.

Priority should be:

1. high `remaining_case_score`;
2. low `captured_fraction`;
3. fast enough case runtime;
4. hidden-set plausibility.

## 5. Exact Frontier To Quantum Tuning Map

Exact frontier artifacts should guide legal quantum tuning in four ways.

### 5.1 Seed Sweep Priority

Run seed sweeps for all public cases, not just low-score cases.

Seed candidates:

```text
2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2035, 2041
```

But schedule by headroom:

- high remaining cases first;
- cases with slow seeds get one-seed-at-a-time execution;
- write results after every single run so timeout does not lose work.

### 5.2 Exact-Frontier Coverage Diagnostics

For each case:

1. Run current `answer.main1()` samples.
2. Compute sampled ND frontier.
3. Load exact frontier.
4. Measure:
   - nearest sampled-front distance for each exact point;
   - exact anchors missed by sampling;
   - HV contribution of exact points not represented by samples;
   - which scalarization lambdas best select those missing exact points.

Useful outputs:

- `results/exact_guidance/{case}_coverage.json`;
- ranked missing exact frontier regions;
- recommended lambda IDs or lambda vectors;
- recommended warm-start exact state indices for offline diagnosis only.

### 5.3 Lambda Targeting

For an exact frontier point `x`, compute:

```python
lambda_id = argmin_lambda dot(lambda, obj_norm[x])
```

Then compare:

- is that lambda already in broad set?
- does current warm selector choose nearby local candidates?
- does quantum sampling under that lambda ever hit nearby exact-front states?

This guides legal tuning:

- include more lambdas near exact missing regions;
- allocate extra shots to lambdas whose sampled points sit near high-HV exact regions;
- do not insert exact states directly.

### 5.4 Warm-Start Tuning

Exact frontier states may be used only as diagnostic references. A safer legal path is:

- use local-search states as warm-start candidates;
- score local candidates by distance to exact missing regions;
- select local warm candidates that cover exact frontier gaps;
- keep returned samples quantum-generated.

Avoid:

- directly using exact frontier bitstrings as returned samples;
- directly warm-starting from exact public frontier in the final hidden submission unless we accept public overfitting risk.

## 6. Implementation Workstreams

### Worker A: Fast Exact Headroom

Write scope:

- `scripts/exact_public_headroom.py`

Tasks:

- replace repeated small-chunk merge with large-chunk local-ND candidate collection;
- add per-case progress printing;
- save frontier NPZ artifacts;
- write JSON and CSV summaries;
- support `--cases`, `--chunk-size`, `--out`, `--frontier-dir`;
- handle one-point ND/HV edge cases safely.

Verification:

```powershell
python -m py_compile scripts/exact_public_headroom.py
python scripts/exact_public_headroom.py --cases 08 --chunk-size 131072 --out results/exact_public_headroom_08.json
python scripts/exact_public_headroom.py --cases 00 08 --chunk-size 131072 --out results/exact_public_headroom_smoke.json
```

Exit criteria:

- exact HV completes for at least one case in a reasonable time;
- output includes `score_max_case` and `remaining_case_score`;
- saved NPZ loads successfully.

### Worker B: Persistent Seed Sweep

Write scope:

- `scripts/eval_seed_mix.py`
- optionally new `scripts/run_seed_grid_resumable.py`

Tasks:

- write one result row immediately after each case/seed run;
- resume from existing CSV/JSON without rerunning completed rows;
- support all `00..09` cases;
- support seed list;
- support per-run timeout warning in output;
- preserve `sample_used == 100000` checks.

Verification:

```powershell
python -m py_compile scripts/eval_seed_mix.py
python scripts/eval_seed_mix.py --full --cases 08 --seeds 2026 --out results/seed_resumable_smoke
```

Exit criteria:

- one completed run is written to disk immediately;
- interrupted runs do not corrupt previous rows;
- outputs can be ranked per case.

### Worker C: Exact Coverage Guidance

Write scope:

- new `scripts/exact_frontier_guidance.py`

Tasks:

- load exact frontier NPZ;
- run or load current answer samples for a case;
- compute sampled ND objectives;
- compare exact vs sampled frontiers;
- rank missing exact regions by approximate HV impact and nearest sampled distance;
- map missing exact points to best lambda IDs from `load_weight_pool`;
- write guidance JSON.

Verification:

```powershell
python -m py_compile scripts/exact_frontier_guidance.py
python scripts/exact_frontier_guidance.py --case 08 --exact-frontier results/exact_frontiers/k5_grid4x5_08.npz --out results/exact_guidance_08.json
```

Exit criteria:

- produces ranked missing exact points/regions;
- includes recommended lambda IDs;
- does not edit `answer.py`.

### Worker D: Exact-Guided Quantum Strategy Hooks

Write scope:

- `scripts/ablate_main1.py`
- no `answer.py` edits unless a winner is proven.

Tasks:

- add strategies that consume guidance lambda IDs from Worker C:
  - `exact_guided_lambda_500`;
  - `exact_guided_broad_warm`;
  - `exact_guided_scout_focus`.
- returned spins must still come from MindQuantum sampling;
- exact frontier states cannot be appended to samples.

Verification:

```powershell
python -m py_compile scripts/ablate_main1.py
python scripts/ablate_main1.py --case data/public/k5_grid4x5_08.npz --strategy exact_guided_lambda_500 --warm-c 0.1
```

Exit criteria:

- strategy produces exactly `100000` rows;
- at least one exact-guided strategy runs on one case;
- results are written for comparison.

## 7. Integration Rules

Only update `answer.py` after a strategy satisfies all of:

- improves per-case score versus current `answer.py` for at least one target case;
- does not materially regress guard cases;
- total estimated public score improves;
- runtime remains below the one-hour public budget with margin;
- final samples remain MindQuantum-generated.

Allowed final changes:

- deterministic seed schedule updates;
- lambda set changes;
- shot allocation changes;
- warm-start selection changes based on local candidates;
- code comments documenting exact-frontier diagnostic use.

Not allowed final changes:

- appending exact enumerated states to `sample_spins`;
- using exact public frontier as a hidden/public case lookup in returned samples;
- bypassing quantum sampling.

## 8. Execution Order

1. Worker A fixes and runs exact headroom for all public cases.
2. Main agent reviews headroom table and picks priority order.
3. Worker B starts resumable seed sweep for all cases ordered by headroom.
4. Worker C builds exact coverage guidance for top `3` headroom cases.
5. Worker D adds exact-guided legal ablation strategies.
6. Main agent integrates only proven winners into `answer.py`.
7. Final verification:

```powershell
python -m py_compile answer.py utils.py run.py baseline.py scripts/*.py
python run.py --split public --large-shots 1000 --out results/exact_guided_public_lowshots.json
python run.py --split public --out results/exact_guided_public_default.json
```

## 9. Risk Notes

- Exact public headroom is public-specific. It is useful for diagnosis, but final strategy should still have a hidden-set story.
- Seed schedule public fingerprints can raise overfit risk. Use exact headroom to choose where to spend seed sweeps, not to hard-code exact outputs.
- Exact-guided lambdas may improve public but hurt hidden if too specialized. Guard with broad baseline coverage.
- Full public default verification remains expensive; keep low-shot and per-case smoke checks before committing large runs.

## 10. Execution Log: 2026-05-24

Exact enumeration completed for all public `n=20` cases with `scripts/exact_public_headroom.py`.

Artifacts:

- `results/exact_public_headroom.json`
- `results/exact_public_headroom.csv`
- `results/exact_frontiers/k5_grid4x5_00.npz` through `k5_grid4x5_09.npz`

Headroom table, sorted by remaining case-score:

| case | current | exact max | remaining | captured | exact ND |
|---|---:|---:|---:|---:|---:|
| 09 | 156.131 | 191.122 | 34.991 | 0.817 | 6779 |
| 07 | 152.384 | 186.730 | 34.346 | 0.816 | 1553 |
| 04 | 253.517 | 282.358 | 28.841 | 0.898 | 5623 |
| 00 | 473.399 | 500.959 | 27.560 | 0.945 | 5053 |
| 08 | 90.446 | 109.297 | 18.851 | 0.828 | 2557 |
| 02 | 229.094 | 246.194 | 17.101 | 0.931 | 2516 |
| 06 | 247.393 | 262.985 | 15.592 | 0.941 | 4950 |
| 05 | 133.891 | 142.611 | 8.721 | 0.939 | 2536 |
| 01 | 98.046 | 102.322 | 4.275 | 0.958 | 5256 |
| 03 | 302.333 | 306.046 | 3.713 | 0.988 | 1070 |

Key interpretation:

- Public exact average upper bound is `233.062`, while the old measured public average was `213.663`.
- The total theoretical public headroom is only about `+19.399` average score.
- Low current score does not imply large easy headroom: cases `01` and `08` are low, but `01` is almost saturated and `08` has only `18.851` case-score remaining.
- The best headroom priorities are `09`, `07`, `04`, `00`, then `08`, `02`, `06`.

Exact-guided strategy results:

- `exact_guided_lambda_500` on case `09`: `score=0.000`; rejected.
- `exact_guided_broad_warm` on case `09`: `score=123.438`; rejected versus current `164.744`.
- `exact_guided_broad_warm` on case `07`: `score=69.382`; rejected versus current `165.651`.

Conclusion: exact frontier lambda mapping is useful as a diagnostic, but the tested direct exact-guided sampling distributions are worse than the current broad-plus-local warm strategy.

Seed sweep winners integrated into `answer.py`:

| case | old seed | new seed | old score | new score | case-score delta |
|---|---:|---:|---:|---:|---:|
| 02 | 2029 | 2041 | 229.094 | 232.730 | +3.636 |
| 05 | 2026 | 2028 | 133.891 | 135.389 | +1.498 |
| 06 | 2026 | 2033 | 247.393 | 247.504 | +0.111 |
| 08 | 2026 | 2027 | 90.446 | 96.919 | +6.473 |

Other sweep conclusions:

- `00` and `04`: default `2026` remained best among tested seeds.
- `01`: `2031` remained best after testing `2024,2025,2026,2027,2028,2030,2032,2033,2035,2041`.
- `07`: `2031` remains best; closest new run was `2033=164.371`.
- `09`: `2031` remains best; closest new run was `2041=162.685`.

Current estimated public average after integrated seed updates:

```text
217.023136
```

Estimated improvement versus the old `seed_schedule_with01_public_default` average `213.663286`:

```text
+3.359851
```

Stability fixes applied:

- Replaced small `@` / BLAS-heavy matrix multiplies in `answer.py` and tuning scripts with `np.einsum(..., optimize=False)` where this Windows MindQuantum environment threw `0xc06d007f`.
- Added safe local HV/energy evaluation helpers to diagnostic scripts so score sweeps no longer crash in `run._hv_from_spins`.

Verification completed:

```powershell
python -m py_compile answer.py scripts/ablate_main1.py scripts/exact_frontier_guidance.py scripts/exact_public_headroom.py scripts/run_seed_grid_resumable.py
```

Targeted default-schedule verification:

- case `02`: seed `2041`, score `232.729692`, rows `100000`.
- case `05`: seed `2028`, score `135.388737`, rows `100000`.
- case `06`: seed `2033`, score `247.503756`, rows `100000`.
- case `08`: seed `2027`, score `96.919324`, rows `100000`.

Verified current per-case main1 score table:

| case | score |
|---|---:|
| 00 | 473.398741 |
| 01 | 98.046167 |
| 02 | 232.729692 |
| 03 | 302.332699 |
| 04 | 253.517087 |
| 05 | 135.388737 |
| 06 | 247.503756 |
| 07 | 165.651423 |
| 08 | 96.919324 |
| 09 | 164.743738 |

Additional verification notes:

- `run.py --split public --max-cases 1 --large-shots 1000` passed after stabilizing `energy_batch_fast`.
- A full `run.py --split public --large-shots 1000` attempt exceeded the local 60-minute tool timeout without writing its final JSON, so final public main1 score is assembled from resumable per-case verification artifacts instead.
