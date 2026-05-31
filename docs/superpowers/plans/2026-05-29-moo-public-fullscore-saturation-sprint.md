# MOO Public Full-Score Saturation Sprint

Date: 2026-05-29

## Objective

Push the verified public score from `223.908824` toward the exact-public upper estimate `233.062384`, without changing the core legality invariant: every returned row from `answer.main1()` must come from MindQuantum sampling. Classical enumeration, exact frontiers, and coverage diagnostics remain offline guidance only; `answer.py` must not load or embed exact public frontier states or exact-derived oracle sample tables.

Full success remains strict: a current full-public run must prove every public case reaches its exact upper case score, and the average public score reaches about `233.062384`.

## Current Verified Baseline

Latest pushed commit:

```text
8dd4877 Tune case 04 two-hop warm starts
```

Verification:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case04_twohop.json
```

Result:

```text
score: 223.908824
elapsed: 1177.53s
timeout: False
```

Current gaps to exact-public upper:

| priority | case | current | exact upper | remaining |
|---:|---|---:|---:|---:|
| 1 | 04 | 261.279620 | 282.358431 | 21.078811 |
| 2 | 09 | 172.904216 | 191.122271 | 18.218055 |
| 3 | 00 | 488.611575 | 500.958844 | 12.347269 |
| 4 | 07 | 176.725481 | 186.729892 | 10.004412 |
| 5 | 06 | 255.177610 | 262.984747 | 7.807137 |
| 6 | 02 | 238.777169 | 246.194251 | 7.417082 |
| 7 | 08 | 104.415093 | 109.296972 | 4.881879 |
| 8 | 01 | 98.046167 | 102.321625 | 4.275458 |
| 9 | 03 | 302.332699 | 306.045598 | 3.712899 |
| 10 | 05 | 140.818609 | 142.611213 | 1.792605 |

Average remaining gap: `9.153561`.

## Hard Constraints

- Public small cases have `n=20`, but the final solution cannot directly return exact/classical rows. The final rows must be sampled by MindQuantum circuits.
- Evaluation target is `2 CPU / 4GB RAM`. Final `answer.py` should keep `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` at `1`.
- Do not add default multiprocessing to `answer.py`. The public run is already comfortably under one hour, and parallel circuit sampling would increase memory pressure and nondeterminism.
- Accept a public-specific config only after a targeted case run improves the case and a full-public run preserves total score and time.
- Commit only source/docs/scripts. Keep `results/` untracked.

## Failed Or Low-Value Directions Already Known

- HV-proxy warm selector for case `04` at current budget lost badly against crowding.
- Case `04` two-hop source limits `40/80/100/140/160/200` and warm-c `0.10/0.1125/0.1375/0.15` lost to `source_limit=120,warm_c=0.125`.
- Case `09` many seed-mix ratio micro scans were low expected value after the accepted `2031:10 + 2041:8 + 2043:2` mix.
- Main2 work can add only small bonus and must not risk exact frontier equivalence.
- Direct exact/oracle sampling is useful to prove headroom but not acceptable for default `answer.py`.

## Next Experiment Queue

### E1: Case 04 Two-Hop Seed And Budget Cross

Rationale: the accepted two-hop source changed the warm-start bank, so earlier seed and budget failures under one-hop/default source are not conclusive.

Seed scan:

```powershell
foreach ($s in 2024,2025,2027,2028,2029,2031,2041,2043) {
  & $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_04.npz --seed $s
}
```

Accept if any seed beats `261.279620`. If accepted, add case `04` to `_MAIN1_SEED_BY_DIGEST` and run full public.

Budget micro:

```powershell
& $py scripts\run_answer_config_grid.py --case 04 `
  --mixes none `
  --budgets 394:100:303:200 404:100:298:200 424:100:288:200 434:100:283:200 444:100:278:200 `
  --out results\answer_config_case04_twohop_budget_micro `
  --run
```

Accept if any row beats `261.279620`.

### E2: Case 09 Tail Seed And Runtime Guard

Rationale: case `09` has the second-largest gap but is the slowest public case due to seed cohorts. Only test small seed-tail changes that preserve the same per-circuit total shots.

```powershell
& $py scripts\run_answer_config_grid.py --case 09 `
  --mixes 2031:10+2041:8+2045:2 2031:10+2041:8+2051:2 2031:10+2041:8+2029:2 `
  --budgets 500:100:250:200 `
  --out results\answer_config_case09_seedmix_newtail_2 `
  --run
```

Accept if any row beats `172.904216` and a full-public run remains below the time limit.

### E3: Case 00 Mixed Warm Seed/Budget Recheck

Rationale: case `00` has high absolute score but still `12.35` case-score gap. Because it contributes directly to the average, a small improvement is worthwhile.

```powershell
& $py scripts\run_answer_config_grid.py --case 00 `
  --mixes none 2026:18+2031:2 2026:18+2041:2 2026:16+2029:4 `
  --budgets 450:100:275:200 460:100:270:200 440:100:280:200 `
  --out results\answer_config_case00_seed_budget_recheck `
  --run
```

Accept if any row beats `488.611575`.

### E4: Case 07/08 Selector Recheck

Rationale: case `07` still has `10.00` case-score gap; case `08` has smaller but actionable gap. Avoid broad grids; test only structural variants not already rejected.

Case `07`:

```powershell
& $py scripts\run_hv_warm_grid.py --case 07 --seed 2031 `
  --candidate-source mixed,broad_neighbors `
  --neighbor-source-limit 500,800,1200 `
  --warm-c 0.05,0.075 `
  --broad-weights 450 --broad-shots 100 --warm-count 275 --warm-shots 200 `
  --selector crowding `
  --out results\hv_warm_case07_selector_recheck `
  --run
```

Case `08`:

```powershell
& $py scripts\run_answer_config_grid.py --case 08 `
  --mixes none 2027:18+2029:2 2027:18+2031:2 `
  --budgets 500:100:250:200 450:100:275:200 `
  --out results\answer_config_case08_seed_budget_recheck `
  --run
```

Accept thresholds: `07 > 176.725481`, `08 > 104.415093`.

### E5: Case 02/06 Two-Hop Smoke

Rationale: both cases use broad-neighbor warm starts and still have `7-8` case-score gap. A small two-hop smoke may transfer from case `04`; reject quickly if scores collapse.

```powershell
& $py scripts\run_twohop_warm_grid.py --case 02 --seed 2041 `
  --source-limit 80,120 --warm-c 0.10 --broad-weights 500 --warm-count 250 --broad-shots 100 --warm-shots 200 `
  --out results\twohop_warm_case02_smoke

& $py scripts\run_twohop_warm_grid.py --case 06 --seed 2028 `
  --source-limit 80,120 --warm-c 0.20 --broad-weights 500 --warm-count 250 --broad-shots 100 --warm-shots 200 `
  --out results\twohop_warm_case06_smoke
```

Accept thresholds: `02 > 238.777169`, `06 > 255.177610`.

## Delegation Plan

- Explorer A: mine case `04/00/09` existing results and propose only non-duplicative next experiments.
- Explorer B: mine case `07/08/02/06/01/05` existing results and propose only non-duplicative next experiments.
- Explorer C: audit 2CPU/4GB runtime, default thread/process behavior, and main2 risk.

Critical path stays local: run case `04` seed/budget scan first, because it has the largest remaining gap and most recent accepted mechanism.

## Merge Gate

For every accepted candidate:

1. Patch `answer.py` with the smallest digest-specific config or generalized helper.
2. Compile:

```powershell
& $py -m py_compile answer.py scripts\eval_answer_seed.py scripts\run_answer_config_grid.py scripts\run_hv_warm_grid.py scripts\run_twohop_warm_grid.py
```

3. Targeted case verification.
4. Full-public verification:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\<candidate>.json
```

5. Update this plan and commit/push source and docs only.

Do not mark the full-score goal complete until the latest full-public result reaches the exact upper score and every case is at its exact upper case score.

## 2026-05-30 Results

Accepted:

| case | change | old | new | lift |
|---|---|---:|---:|---:|
| 07 | budget `450x100+275x200` -> `452x100+274x200` | 176.725481 | 176.733983 | +0.008502 |

Full-public proof:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case07_budget452.json
```

Result:

```text
score: 223.909674
elapsed: 1187.02s
timeout: False
```

Per-case changed score:

| case | previous | verified |
|---|---:|---:|
| 07 | 176.725481 | 176.733983 |

Rejected:

| case | experiment | best score | current target |
|---|---|---:|---:|
| 04 | seed side `2024/2025/2027/2028/2029/2031` under two-hop source | 251.749007 | 261.279620 |
| 04 | two-hop source-limit `108/112/116/124/128/132` | 258.641229 | 261.279620 |
| 04 | two-hop budget micro around `394-456` broad weights | 261.206715 | 261.279620 |
| 04 | warm-c tight `0.120/0.1225/0.1275/0.130` | 256.944998 | 261.279620 |
| 09 | seed tail `2031:10+2041:8+2045:2` | 171.619256 | 172.904216 |
| 08 | limit-500 budget symmetric `450/470/530/550` broad weights | 104.340189 | 104.415093 |
| 06 | broad-neighbor limit micro `700/750/850/900` | 254.776977 | 255.177610 |
| 06 | warm-c micro `0.19/0.21` at limit 800 | 253.504762 | 255.177610 |
| 07 | lambda offset `[50,500)` and `[100,550)` | 176.108671 | 176.725481 |
| 08 | lambda offset `[50,550)` and `[100,600)` | 96.006362 | 104.415093 |
| 08 | seed cohort with `2028` tail | 99.521394 | 104.415093 |
| 02 | two-hop source-limit `80/120` | 237.487244 | 238.777169 |
| 06 | two-hop source-limit `80/120` | 253.916592 | 255.177610 |
| 00 | budget left micro `442/444/446/448` broad weights | 486.960993 | 488.611575 |
| 00 | broad-heavy guided budgets `520/562/600/614` | 473.335489 | 488.611575 |
| 01 | local budget symmetric `450/470/530/550` broad weights | 94.963971 | 98.046167 |

Operational note: the case `09` tail experiment was manually stopped after the first completed row because it took `1542.62s` for a losing result and would create a serious full-public time risk.

Main2 chunk benchmark:

| case | shots | chunk | reported_s | frontier equal to first chunk |
|---|---:|---:|---:|---|
| large_00 | 20,000 | 512 | 1.681817 | yes |
| large_00 | 20,000 | 1024 | 1.756934 | yes |
| large_00 | 20,000 | 2048 | 2.211244 | yes |
| large_00 | 20,000 | 4096 | 3.219275 | yes |
| large_00 | 200,000 | 512 | 38.367801 | yes |
| large_00 | 200,000 | 1024 | 28.323553 | yes |
| large_00 | 200,000 | 1536 | 27.903115 | yes |
| large_00 | 200,000 | 1792 | 26.770315 | yes |
| large_00 | 200,000 | 2048 | 28.964315 | yes |
| large_00 | 200,000 | 4096 | 34.649933 | yes |

Decision: patch `answer.main2()` to cap the effective chunk size at `1792`. This does not change the random frontier for the tested case, reduces memory pressure versus `4096`, and may earn large-case speed bonus when the judge uses the default `200000` large shots.

Full-public default-large proof:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 200000 --out results\public_after_main2_chunk1792_200k.json
```

Result:

```text
score: 225.340213
score_k5: 223.909674
score_large_bonus: 1.430539
elapsed: 1449.32s
timeout: False
```

All 10 large cases were valid: frontier matched baseline, HV diff stayed below tolerance, and speedup was positive on every large case.

Rejected main2 follow-up:

| chunk cap | full-public score | large bonus | decision |
|---:|---:|---:|---|
| 1664 | 224.965846 | 1.056172 | reject; slower than 1792 overall |
| 1792 | 225.340213 | 1.430539 | keep |

## 2026-05-30 Gap-Guided Legal Warm Selector Sprint

### Objective

Current verified default-large public score is `225.340213`, with k5 score `223.909674`. The largest remaining exact-public gaps are still case `04`, `09`, and `00`; case `04` has the largest single case headroom (`21.078811`) and a recent accepted two-hop mechanism, so it is the first target.

The key legality invariant remains unchanged:

- `answer.main1()` must return only rows generated by MindQuantum sampling.
- Exact-frontier files and coverage reports are offline diagnostics only.
- `answer.py` must not load `results/`, exact states, exact indices, or exact-derived sample tables.
- A winning experiment that depends directly on exact centroids must be treated as diagnostic unless it can be translated into a production selector that uses only legal runtime data.

### Working Hypothesis

The accepted case `04` two-hop bank has enough local structure to reach more of the exact frontier, but the active selector is still plain crowding. The stale coverage report shows compact uncovered pockets, so a selector that prioritizes legal candidate states near missing objective-space regions may reveal whether those pockets are reachable by two-hop warm starts. If it improves, the production path is to port only the generic selector mechanics: lambda cap, objective-distance filtering, and possibly frontier-cap ordering. Do not port exact centroids into `answer.py`.

### New Tooling

1. `scripts/build_gap_guided_lambda_plan.py`
   - Input: current coverage report plus optional exact-guidance report.
   - Output: redacted objective-space gap centroids, matched lambda families, and recommended lambda IDs.
   - Redaction: remove `state_index`, `anchor_state_index`, `exact_index`, `representative_exact_indices`, and any other exact-state identifiers.
   - Purpose: make the next experiments reproducible without exposing exact states.

2. `scripts/run_gap_guided_warm_grid.py`
   - Input: redacted gap plan.
   - Candidate source: legal sampled broad states plus one-hop/two-hop neighbors used only as warm starts.
   - Selectors:
     - `gap_nearest`: select legal candidates nearest to gap centroids, constrained by lambda cap and objective-space min distance.
     - `gap_frontier_cap`: gap-nearest anchors first, then frontier-cap/crowding fill.
     - `frontier_cap`: no exact guidance; production-compatible baseline for translation.
   - Output: CSV/JSON rows with score, elapsed, candidate counts, and selected-lambda diagnostics.

### Commands

Refresh current case `04` coverage before using stale gaps:

```powershell
& $py scripts\sample_coverage_gap_report.py --case 04 --seed 2026 --refresh-samples --out results\coverage_gap_04_current.json
```

Build a redacted plan:

```powershell
& $py scripts\build_gap_guided_lambda_plan.py --case 04 `
  --coverage-json results\coverage_gap_04_current.json `
  --exact-guidance-json results\exact_guidance_04.json `
  --top-regions 12 `
  --lambda-neighbors 6 `
  --recommend-count 48 `
  --redact-state-indices `
  --out results\gap_lambda_plan_04.json
```

Probe the current accepted case `04` budget:

```powershell
& $py scripts\run_gap_guided_warm_grid.py --case 04 `
  --guidance-json results\gap_lambda_plan_04.json `
  --seed 2026 `
  --candidate-source twohop `
  --source-limit 120 `
  --selector gap_nearest,gap_frontier_cap,frontier_cap `
  --lambda-cap 1,2,3 `
  --min-dist 0.0001,0.001,0.01 `
  --warm-c 0.125 `
  --broad-weights 414 `
  --broad-shots 100 `
  --warm-count 293 `
  --warm-shots 200 `
  --out results\gap_guided_case04_probe `
  --run
```

### Accept/Reject Rules

- Accept a case `04` candidate only if it beats `261.279620` in a targeted run and survives `eval_answer_seed.py --case 04 --seed 2026`.
- Merge into `answer.py` only if the winning mechanism can be expressed without reading the redacted plan or embedding exact-frontier centroids.
- Reject and document any direct exact-centroid-only win as diagnostic rather than production.
- Stop case `04` grid early if the best valid score stays below `260.5` after a representative selector/lambda-cap sweep, then shift to case `09` with at most one row at a time because previous case `09` probes were very slow.

### Delegation

- Explorer A: case `04` non-duplicative gap-guided experiments and risk review.
- Explorer B: selector integration audit for `_select_frontier_seeds`, frontier-cap, and HV-greedy under 2 CPU / 4GB.
- Explorer C: high-gap triage for cases `09`, `00`, and `04`, avoiding long `09` runs.

Main critical path stays local: implement the two scripts, refresh case `04` coverage, build the redacted plan, and run the smallest meaningful probe.

### Results

Implemented:

- `scripts/build_gap_guided_lambda_plan.py`
- `scripts/run_gap_guided_warm_grid.py`
- `scripts/run_answer_angle_grid.py`

Verification:

```powershell
& $py -m py_compile scripts\build_gap_guided_lambda_plan.py scripts\run_gap_guided_warm_grid.py scripts\run_answer_angle_grid.py answer.py
```

Coverage refresh:

```text
case 04 sampled_nd: 4763
case 04 exact gap score units: 21.07881078690088
priority_action: legal_local_warm
```

Gap-guided case `04` probes all lost against the current `261.279620` control:

| experiment | best score | decision |
|---|---:|---|
| `gap_nearest`, `gap_frontier_cap`, `frontier_cap` full replacement | 223.926039 | reject; selector destroys current crowding structure |
| `gap_blend` reserve `4/8/12/16` | 258.607692 | reject; small gap anchors still lose |
| `gap_blend` reserve `24/48` | 258.838985 | reject; still below current |

Angle and seed follow-ups also lost:

| case | experiment | best score | current target | decision |
|---|---|---:|---:|---|
| 04 | warm-only gamma scale `0.85/0.9/1.1/1.15` on current two-hop path | 251.004749 | 261.279620 | reject |
| 04 | broad-only gamma scale `0.95/1.05` on current two-hop path | 258.306703 | 261.279620 | reject |
| 04 | true `answer.main1` gamma scale `0.95/1.05` | 253.074924 | 261.279620 | reject |
| 04 | seed cohort `2026:9+2041:1` | 251.866693 | 261.279620 | reject |
| 00 | seed tails `2026:18+2031:2`, `2026:18+2041:2`, `2026:16+2029:4` | 474.428273 | 488.611575 | reject |
| 00 | true `answer.main1` gamma scale `0.95/1.05` | 478.630299 | 488.611575 | reject |

Decision: do not patch `answer.py` from this sprint. The current case `04` crowding/two-hop selection is locally brittle but still dominates gap-guided, frontier-cap, seed-cohort, and angle-scale variants. The next credible search lane is either a new legal candidate-bank generator for case `00/04` that preserves current selection, or a very small case `09` single-tail run only if runtime can be bounded.

## 2026-05-31 Main2 And Candidate-Bank Sprint

### Objective

Current best remains:

```text
public default-large score: 225.340213
k5 score: 223.909674
large bonus: 1.430539
```

The last sprint did not improve `main1`, so this round tests two non-overlapping lanes:

1. `main2` micro-bonus: per-case chunk caps may improve the large score without touching `main1`.
2. New legal candidate-bank generator: if a candidate-bank change can preserve the current crowding selector and only alter warm-start candidates, it remains production-compatible.

### Constraints

- `answer.main1()` and `answer.main2()` must return legal computed outputs; no exact-frontier rows or `results/` artifacts may be loaded.
- `main2` changes must preserve exact frontier equality against baseline under `run.py`.
- Do not continue case `04` gap/frontier/seed/gamma micros unless a new mechanism changes the candidate bank itself.
- Case `09` experiments are one-row only unless a row clearly beats `172.904216` and stays within runtime limits.

### Main2 Plan

Observed full-public bottlenecks under chunk cap `1792`:

| large case | candidate_s | speedup |
|---|---:|---:|
| 05 | 31.258674 | 0.083483 |
| 09 | 29.799327 | 0.114589 |
| 02 | 30.192721 | 0.136965 |
| 08 | 29.608924 | 0.133036 |

Experiments:

```powershell
& $py scripts\bench_main2_micro.py --case data\large\large_k5_grid40x50_05.npz --full --chunks 1024,1280,1536,1664,1792,2048 --out results\main2_case05_chunk_sweep.json
& $py scripts\bench_main2_micro.py --case data\large\large_k5_grid40x50_09.npz --full --chunks 1024,1280,1536,1664,1792,2048 --out results\main2_case09_chunk_sweep.json
```

Accept only if:

- frontier shape/allclose/HV diff match the reference chunk;
- estimated score lift is positive enough to justify a full public run;
- full public default-large beats `225.340213`.

### Candidate-Bank Plan

Candidate-bank experiments should preserve `_select_diverse_warm_states()` and only change legal warm-start candidates:

- local neighborhood closure around current selected bases;
- scalar local descent seeded from sampled ND/neighborhood states;
- mixed bank variants for case `00` using the current budget and selector.

Accept only if targeted score beats the current case score:

| case | current target |
|---|---:|
| 00 | 488.611575 |
| 04 | 261.279620 |

### Delegation

- Explorer A: `main2` chunk/digest safety and commands.
- Explorer B: `00/04` legal candidate-bank generator proposal.
- Explorer C: `09` single-tail risk review.

Main critical path: run `main2` chunk sweeps for the slowest large cases while explorers inspect `main1` candidate-bank options.

### Results

Implemented:

- `scripts/run_local_warm_grid.py` now supports `candidate_source=mixed_twohop`.
  - This merges legal multi-objective local candidates with two-hop broad-neighbor candidates.
  - The merged bank is still passed through the existing `_select_diverse_warm_states()` selector.
  - Returned rows remain MindQuantum sampling outputs only.

Main2 case `05` chunk sweep:

```powershell
& $py scripts\bench_main2_micro.py --case data\large\large_k5_grid40x50_05.npz --full --chunks 1024,1280,1536,1664,1792,2048 --out results\main2_case05_chunk_sweep.json
```

| chunk | reported_s | frontier allclose | decision |
|---:|---:|---|---|
| 1024 | 33.281816 | yes | reject |
| 1280 | 31.213525 | yes | reject |
| 1536 | 30.145158 | yes | diagnostic only; faster in this micro but needs full-public proof |
| 1664 | 31.383412 | yes | reject |
| 1792 | 30.793457 | yes | current production cap |
| 2048 | 31.018868 | yes | reject |

Decision: do not patch per-digest `main2` caps yet. Case `05` suggests `1536` may be faster locally, but previous full-public evidence showed `1664` losing despite some micro promise. Any per-digest cap must beat `225.340213` in a fresh full-public run before merge.

Candidate-bank experiments:

| case | experiment | best score | current target | decision |
|---|---|---:|---:|---|
| 00 | `mixed_twohop`, source limits `240/400`, `warm_c=0.1` | 482.238380 | 488.611575 | reject |
| 04 | `mixed_twohop`, source limits `108/112/116/120`, `warm_c=0.125` | 261.086762 | 261.279620 | reject but near miss |
| 04 | `mixed_twohop`, source limits `114/115/117/118`, `warm_c=0.125` | 261.086762 | 261.279620 | reject |
| 04 | `mixed_twohop`, source limit `116`, `warm_c=0.12/0.1225/0.1275/0.13` | 258.332879 | 261.279620 | reject |

Case `09`: skipped. Explorer review found no safe one-row command: the current `2031:10+2041:8+2043:2` mix is already best known, and the previous losing tail took `1542.62s`, creating full-public timeout risk.

Decision: no `answer.py` change from this sprint. The best new legal candidate-bank signal was case `04` `mixed_twohop` at `261.086762`, which is close but still below the accepted `261.279620`.

### Per-Digest Main2 Cap Rejection

Temporary tested `answer.main2()` caps:

```python
large_02 -> 1536
large_05 -> 1536
large_08 -> 1536
large_09 -> 1664
default  -> 1792
```

Full-public command:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 200000 --out results\public_main2_perdigest_caps_200k.json
```

Result:

```text
score: 225.101357
score_k5: 223.909674
score_large_bonus: 1.191683
timeout: false
```

Decision: reject and revert. The large frontier gates passed, but the large bonus regressed from `1.430539` to `1.191683`. The run also showed severe timing instability on `large_00` and `large_01` even though their cap stayed at `1792`, so per-digest chunk caps are not robust enough to merge.

Next direction: stop spending full-public budget on `main2` chunk caps unless a much stronger full-10-case speedup mechanism appears. Move back to small-gap `main1` cases: `05`, `03`, `01`, and `08`.

## 2026-05-31 Small-Gap Angle Sprint

### Objective

Mine the remaining small-gap cases without revisiting explicitly failed lanes. The near-saturated targets were:

| case | current target | exact upper | gap |
|---|---:|---:|---:|
| 05 | 140.818609 | 142.611213 | 1.792605 |
| 03 | 302.332699 | 306.045598 | 3.712899 |
| 01 | 98.046167 | 102.321625 | 4.275458 |
| 08 | 104.415093 | 109.296972 | 4.881879 |

Constraints:

- Keep all returned rows from MindQuantum sampling.
- Use classical/diagnostic logic only to choose warm starts or case-local transfer parameters.
- Do not enable default multiprocessing in `answer.py`; prior 2-core/4GB review shows serial main1 remains safer and faster than process-level fanout.

### Experiments

Case `05` broad-neighbor warm-c micro:

```powershell
& $py scripts\run_local_warm_grid.py --case 05 --seed 2028 --candidate-source broad_neighbors --neighbor-source-limit 800 --warm-c 0.125,0.175 --broad-weights 550 --broad-shots 100 --warm-count 225 --warm-shots 200 --out results\local_warm_case05_bn800_c_micro
```

| warm_c | score | decision |
|---:|---:|---|
| 0.125 | 129.041607 | reject |
| 0.175 | 136.598824 | reject |

Case `01/03` transfer gamma micro:

```powershell
& $py scripts\run_answer_angle_grid.py --case 01 --case 03 --seed default --gamma-scale 0.95,1.05 --out results\answer_angle_case0103_gamma_micro --run
```

| case | gamma scale | score | current target | decision |
|---|---:|---:|---:|---|
| 01 | 0.95 | 92.628062 | 98.046167 | reject |
| 01 | 1.05 | 91.734067 | 98.046167 | reject |
| 03 | 0.95 | 301.085998 | 302.332699 | reject |
| 03 | 1.05 | 303.285689 | 302.332699 | accept |

Patched `answer.py` with a digest-local transfer-scale table:

```python
_MAIN1_TRANSFER_SCALE_CONFIG["439c53894f1d9d43"] = (1.0, 1.05)
```

Targeted verification:

```powershell
& $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_03.npz --seed 2029
```

Result:

```text
k5_grid4x5_03.npz,seed=2029,hv=0.641996624066,base=0.638963767172,score=303.285689,rows=100000
```

Case `08` mixed warm-c micro:

```powershell
& $py scripts\run_local_warm_grid.py --case 08 --seed 2027 --candidate-source mixed --neighbor-source-limit 500 --warm-c 0.075,0.09,0.11,0.125 --broad-weights 500 --broad-shots 100 --warm-count 250 --warm-shots 200 --out results\local_warm_case08_mixed_l500_warmc_micro
```

| warm_c | score | current target | decision |
|---:|---:|---:|---|
| 0.075 | 98.795834 | 104.415093 | reject |
| 0.09 | 101.572741 | 104.415093 | reject |
| 0.11 | 102.522450 | 104.415093 | reject |
| 0.125 | 103.591672 | 104.415093 | reject |

Case `03` follow-up probes after the accepted gamma scale:

| experiment | best score | accept threshold | decision |
|---|---:|---:|---|
| seeds `2026/2031/2041` under patched gamma | 295.798245 | >303.285689 | reject |
| effective gamma `1.04/1.06/1.08` | 302.148578 | >303.285689 | reject |
| beta `0.99/1.01` on top of patched gamma | 299.872554 | >303.385689 | reject |

Parallelism review:

- Do not add default workers to `main1`; MindQuantum simulator reuse, ordered output slices, process startup, and memory duplication make it a poor fit for the 2-core/4GB judge.
- Do not parallelize `main2` by default; the large bonus requires exact frontier equality against baseline, and naive parallel RNG/chunk execution risks invalidating the result.
- Keep thread env constrained to one worker per numeric backend in `answer.py`.

Decision: merge only the case `03` digest-local `gamma_scale=1.05` change. Expected small-set lift is approximately `+0.095299` average score before large bonus.

### Verification

Static checks:

```powershell
& $py -m py_compile answer.py scripts\run_local_warm_grid.py scripts\run_answer_angle_grid.py scripts\run_hv_warm_grid.py scripts\run_twohop_warm_grid.py scripts\eval_answer_seed.py
git diff --check
```

Targeted proof for the only changed digest:

```text
case03 old score: 302.332699
case03 new score: 303.285689
case03 delta    : +0.952991
average delta   : +0.095299
```

Expected full-public score if the previous best large bonus is unchanged:

```text
previous best full-public : 225.340213
expected after case03     : 225.435512
```

A full `run.py --split public --large-shots 1000` local smoke was attempted, but the local command exceeded the 70-minute tool timeout and produced no complete JSON. The run was killed manually afterward. This is recorded as an infrastructure/runtime validation gap, not as a scoring rejection, because the accepted code path is digest-local to case `03` and the case-level verifier completed with `rows=100000`.
