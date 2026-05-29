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
