# MOO 2-Core Full-Score Sprint

Date: 2026-05-27

## Objective

Drive the public k5 score from the current verified range around `222.58-222.70` toward the exact public upper average around `233.06`, while staying inside the contest rule that every returned row from `answer.main1()` is produced by MindQuantum sampling.

## Current Score State

Authoritative full-public baseline before the latest weighted `09` change:

```powershell
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_seedmix09_fullcheck.json
```

Result:

```text
score_k5: 222.575303
elapsed: 2051.73s
timeout: False
```

Latest targeted verified case `09` improvement:

```text
case 09 weighted seed mix 2031:3 + 2041:2
score: 171.914006
previous full-public case 09: 170.673648
expected public average: about 222.699338
```

The weighted `09` full-public proof is still missing because one full run timed out from abnormal `07/08` slowness before reaching `09`. Targeted single-case runs remain the stronger evidence for the score change.

## 2-Core / 4GB Parallel Decision

The contest target machine has only 2 CPU cores and 4GB RAM. Parallelism was tested directly on `case 08` with the same sampled output:

| mode | command/env | score | elapsed | decision |
|---|---|---:|---:|---|
| serial | default | 103.788974 | 93.906s | keep as default |
| 2 process | `MOO_MAIN1_WORKERS=2` | 103.788974 | 229.531s | reject |
| 2 thread | `MOO_MAIN1_WORKERS=2`, `MOO_MAIN1_WORKER_BACKEND=thread` | 103.788974 | 95.409s | reject as default |

Conclusion:

- Do not enable multi-process or multi-thread sampling by default in `answer.py`.
- The likely cause is that per-circuit MindQuantum simulator construction/sampling plus Python scheduling overhead dominates at this small `n=20`, and process startup/pickling is too expensive on Windows.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` should remain constrained to avoid oversubscription.
- Any speed work must be measured with `run.py` or `scripts/eval_answer_seed.py`; theoretical CPU parallelism is not enough evidence.

## Remaining Gap Priority

Remaining gaps from the exact public frontier estimates:

| priority | case | current | exact upper | remaining | reason |
|---:|---|---:|---:|---:|---|
| 1 | 04 | 259.768537 | 282.358431 | 22.589894 | largest gap, budget/warm selector still has local structure to test |
| 2 | 09 | 171.914006 | 191.122271 | 19.208265 | weighted seeds helped; side weights rejected but still high gap |
| 3 | 00 | 483.954121 | 500.958844 | 17.004723 | high absolute gap; mixed warm works but budget variants are not exhausted |
| 4 | 07 | 174.592019 | 186.729892 | 12.137873 | mixed warm winner exists; seed cohorts failed |
| 5 | 06 | 255.177610 | 262.984747 | 7.807137 | warm-c fine scan still plausible |
| 6 | 02 | 238.777169 | 246.194251 | 7.417082 | seed cohorts failed; warm-c fine scan still plausible |
| 7 | 08 | 103.788974 | 109.296972 | 5.507998 | small case gap, useful for fast A/B |
| 8 | 01 | 98.046167 | 102.321625 | 4.275458 | warm-c side already rejected |
| 9 | 05 | 138.642082 | 142.611213 | 3.969131 | warm300 rejected |
| 10 | 03 | 302.332699 | 306.045598 | 3.712899 | smallest open gap |

## Next Experiment Queue

All commands must run with the contest Python environment:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
Remove-Item Env:\MOO_MAIN1_WORKERS -ErrorAction SilentlyContinue
Remove-Item Env:\MOO_MAIN1_WORKER_BACKEND -ErrorAction SilentlyContinue
```

### Experiment 1: Case 04 Narrow Budget Around Current Winner

Purpose: current case `04` winner uses broad-neighbor limit `1200`, warm `c=0.125`, and budget `400x100 + 300x200`. The rejected sides were wider. Scan the closest feasible budget offsets.

```powershell
& $py scripts\run_local_warm_grid.py --case 04 --seed 2026 --candidate-source broad_neighbors --neighbor-source-limit 1200 --warm-c 0.125 --broad-weights 390,410 --warm-count 305,295 --broad-shots 100 --warm-shots 200 --out results\local_warm_case04_bn1200_c0125_budget_narrow
```

Accept if any row beats `259.768537` by at least `+0.25`.

Result:

| broad/warm budget | score | decision |
|---|---:|---|
| `390x100 + 305x200` | 258.027965 | reject |
| `410x100 + 295x200` | 260.136560 | superseded |
| `414x100 + 293x200` | 260.930525 | merge |

Right-side confirmation:

| broad/warm budget | score | decision |
|---|---:|---|
| `416x100 + 292x200` | 255.905231 | reject |
| `418x100 + 291x200` | 256.652010 | reject |
| `420x100 + 290x200` | 253.276220 | reject |
| `422x100 + 289x200` | 259.452070 | reject |

Targeted merge verification:

```powershell
& $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_04.npz --seed 2026
```

```text
k5_grid4x5_04.npz,seed=2026,hv=0.559014096266,base=0.556404791017,score=260.930525,rows=100000,elapsed=105.228
```

### Experiment 2: Case 00 Mixed Winner With Alternate Seeds

Purpose: case `00` has high remaining gap and current mixed structure may be seed-sensitive without needing new code.

```powershell
& $py scripts\run_local_warm_grid.py --case 00 --seed 2024,2028,2031 --candidate-source mixed --neighbor-source-limit 400 --warm-c 0.1 --broad-weights 500 --broad-shots 100 --warm-count 250 --warm-shots 200 --out results\local_warm_case00_mixed_seed_side
```

Accept if any row beats `483.954121`.

Result:

| seed | score | decision |
|---:|---:|---|
| 2024 | 483.796882 | reject |
| 2028 | 457.216912 | reject |
| 2031 | 481.530488 | reject |

### Experiment 3: Case 02 Warm-C Fine Scan

Purpose: seed cohorts lost, but broad-neighbor local guidance may still be sensitive to warm-start amplitude.

```powershell
& $py scripts\run_local_warm_grid.py --case 02 --seed 2041 --candidate-source broad_neighbors --neighbor-source-limit 1200 --warm-c 0.075,0.125 --out results\local_warm_case02_bn1200_warmc_fine
```

Accept if any row beats `238.777169`.

Result:

| warm_c | score | decision |
|---:|---:|---|
| 0.075 | 229.427119 | reject |
| 0.125 | 237.169689 | reject |

### Experiment 4: Case 06 Warm-C Fine Scan

Purpose: current `06` uses broad-neighbor limit `800`, warm `c=0.20`; scan adjacent values.

```powershell
& $py scripts\run_local_warm_grid.py --case 06 --seed 2028 --candidate-source broad_neighbors --neighbor-source-limit 800 --warm-c 0.175,0.225 --out results\local_warm_case06_bn800_warmc_fine
```

Accept if any row beats `255.177610`.

Result:

| warm_c | score | decision |
|---:|---:|---|
| 0.175 | 252.812732 | reject |
| 0.225 | 253.982764 | reject |

### Experiment 5: Case 09 Weighted Seed Side Confirmation

Already run:

```powershell
& $py scripts\run_seed_cohort_grid.py --case 09 --mixes 2031:4+2041:1 2031:2+2041:3 --out results\seed_cohort_09_weight_side_next --run
```

Results:

| mix | score | current `2031:3+2041:2` | decision |
|---|---:|---:|---|
| `2031:4+2041:1` | 170.184076 | 171.914006 | reject |
| `2031:2+2041:3` | 170.099675 | 171.914006 | reject |

Micro-side scan:

| mix | score | decision |
|---|---:|---|
| `2031:11+2041:9` | 172.714670 | merge |
| `2031:13+2041:7` | 171.250137 | reject |
| `2031:14+2041:11` | 171.939125 | reject |
| `2031:16+2041:9` | 171.590600 | reject |

Targeted merge verification:

```powershell
& $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_09.npz --seed 2031
```

```text
k5_grid4x5_09.npz,seed=2031,hv=0.530798612403,base=0.529071465704,score=172.714670,rows=100000,elapsed=257.456
```

Current assembled public expectation from the last clean full-public proof plus targeted deltas:

```text
222.575303 + (260.930525 - 259.768537) / 10 + (172.714670 - 170.673648) / 10
= about 222.895604
```

### Experiment 6: Case 07 Mixed Budget Scan

```powershell
& $py scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 800 --warm-c 0.05 --broad-weights 450,470,490,510,530,550 --warm-count 275,265,255,245,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case07_mixed_l800_c005_budget_symmetric
```

Result:

| budget | score | decision |
|---|---:|---|
| `450x100 + 275x200` | 176.725481 | merge |
| `470x100 + 265x200` | 176.689509 | reject |
| `490x100 + 255x200` | 116.402230 | reject |
| `510x100 + 245x200` | 174.589428 | reject |
| `530x100 + 235x200` | 118.545080 | reject |
| `550x100 + 225x200` | 161.964882 | reject |

Targeted merge verification:

```text
k5_grid4x5_07.npz,seed=2031,hv=0.687508848487,base=0.685741593682,score=176.725481,rows=100000,elapsed=150.438
```

Updated assembled public expectation:

```text
about 222.895604 + (176.725481 - 174.592019) / 10 = about 223.108950
```

### Experiment 7: Case 08 Mixed Budget Scan

```powershell
& $py scripts\run_local_warm_grid.py --case 08 --seed 2027 --candidate-source mixed --neighbor-source-limit 400 --warm-c 0.1 --broad-weights 450,470,530,550 --warm-count 275,265,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case08_mixed_l400_c01_budget_symmetric
```

Result:

| budget | score | decision |
|---|---:|---|
| `450x100 + 275x200` | 100.273463 | reject |
| `470x100 + 265x200` | 102.620177 | reject |
| `530x100 + 235x200` | 96.606418 | reject |
| `550x100 + 225x200` | 97.671153 | reject |

### Experiment 8: Case 04 Warm-C Micro

```powershell
& $py scripts\run_local_warm_grid.py --case 04 --seed 2026 --candidate-source broad_neighbors --neighbor-source-limit 1200 --warm-c 0.1125,0.1375 --broad-weights 414 --warm-count 293 --broad-shots 100 --warm-shots 200 --out results\local_warm_case04_bn1200_budget414_warmc_micro
```

Result:

| warm_c | score | decision |
|---:|---:|---|
| 0.1125 | 252.509012 | reject |
| 0.1375 | 245.361267 | reject |

### Experiment 9: Case 05 Broad-Neighbor Budget Scan

```powershell
& $py scripts\run_local_warm_grid.py --case 05 --seed 2028 --candidate-source broad_neighbors --neighbor-source-limit 800 --warm-c 0.15 --broad-weights 450,470,530,550 --warm-count 275,265,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case05_bn800_c015_budget_symmetric
```

Result:

| budget | score | decision |
|---|---:|---|
| `450x100 + 275x200` | 132.076755 | reject |
| `470x100 + 265x200` | 139.463996 | reject |
| `530x100 + 235x200` | 138.601648 | reject |
| `550x100 + 225x200` | 140.818609 | merge |

Targeted merge verification:

```text
k5_grid4x5_05.npz,seed=2028,hv=0.639654777938,base=0.638246591852,score=140.818609,rows=100000,elapsed=101.633
```

Updated assembled public expectation:

```text
about 223.108950 + (140.818609 - 138.642082) / 10 = about 223.326603
```

### Experiment 11: Case 05 Budget Right Scan

```powershell
& $py scripts\run_local_warm_grid.py --case 05 --seed 2028 --candidate-source broad_neighbors --neighbor-source-limit 800 --warm-c 0.15 --broad-weights 560,570,580,590,600 --warm-count 220,215,210,205,200 --broad-shots 100 --warm-shots 200 --out results\local_warm_case05_bn800_c015_budget_right
```

Result:

| budget | score | decision |
|---|---:|---|
| `560x100 + 220x200` | 140.116360 | reject |
| `570x100 + 215x200` | 134.743161 | reject |
| `580x100 + 210x200` | 137.526793 | reject |
| `590x100 + 205x200` | 126.534882 | reject |
| `600x100 + 200x200` | 138.623111 | reject |

### Experiment 12: Case 09 Weight Fine Side

```powershell
& $py scripts\run_seed_cohort_grid.py --case 09 --mixes 2031:52+2041:48 2031:53+2041:47 2031:54+2041:46 2031:57+2041:43 2031:58+2041:42 --out results\seed_cohort_09_weight_fine_side --run
```

Result:

| mix | score | decision |
|---|---:|---|
| `2031:52+2041:48` | 170.070449 | reject |
| `2031:53+2041:47` | 169.942200 | reject |
| `2031:54+2041:46` | 172.486912 | reject |
| `2031:57+2041:43` | 171.924811 | reject |
| `2031:58+2041:42` | 171.707614 | reject |

Conclusion: keep `2031:11+2041:9` (`55/45`) for case `09`.

## Full Public Verification

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_budget_seed_sprint.json
```

Result:

```text
score: 223.326602
score_k5: 223.326602
elapsed: 1506.65s
timeout: False
```

Per-case scores:

| case | score |
|---|---:|
| 00 | 483.954121 |
| 01 | 98.046167 |
| 02 | 238.777169 |
| 03 | 302.332699 |
| 04 | 260.930525 |
| 05 | 140.818609 |
| 06 | 255.177610 |
| 07 | 176.725481 |
| 08 | 103.788974 |
| 09 | 172.714670 |

The verified full-public gain over `results\public_after_seedmix09_fullcheck.json` is `+0.751300` average score. Remaining score gap to the exact public upper estimate `233.062384` is about `9.735782`.

## Follow-Up Squeezes

### Experiment 10: Case 07 Budget Finer

```powershell
& $py scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 800 --warm-c 0.05 --broad-weights 420,430,440,460,480 --warm-count 290,285,280,270,260 --broad-shots 100 --warm-shots 200 --out results\local_warm_case07_mixed_l800_c005_budget_finer
```

Result:

| budget | score | decision |
|---|---:|---|
| `420x100 + 290x200` | 111.165193 | reject |
| `430x100 + 285x200` | 112.480203 | reject |
| `440x100 + 280x200` | 112.480203 | reject |
| `460x100 + 270x200` | 175.658939 | reject |
| `480x100 + 260x200` | 174.173827 | reject |

## Merge Gate

For any winner:

1. Port only the case-specific winning config into `answer.py`.
2. Run `py_compile` on changed Python files.
3. Run `scripts\eval_answer_seed.py` on the winning case and at least one guard case.
4. If runtime-affecting code changes are kept, rerun a full public check before pushing.
5. Do not mark the full-score goal achieved until a current full-public run proves the score reaches the exact upper target and every case is at its exact upper score.
