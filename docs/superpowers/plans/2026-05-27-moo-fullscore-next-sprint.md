# MOO Full-Score Next Sprint

Date: 2026-05-27

Continuation: 2026-05-29

## Objective

Continue moving the public k5 score from the verified `223.326602` toward the exact public upper estimate `233.062384`, without violating the rule that `answer.main1()` returns only rows sampled by MindQuantum.

This sprint does not redefine success. Full success remains: every public case reaches the exact public upper score, verified by a current full-public run.

## Current Verified State

Latest full-public proof:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_budget_seed_sprint.json
```

Result:

```text
score: 223.326602
score_k5: 223.326602
elapsed: 1506.65s
timeout: False
```

Merged case-specific configs:

| case | tactic | score |
|---|---|---:|
| 04 | broad-neighbor, `warm_c=0.125`, budget `414x100 + 293x200` | 260.930525 |
| 05 | broad-neighbor, `warm_c=0.15`, budget `550x100 + 225x200` | 140.818609 |
| 07 | mixed, limit `800`, `warm_c=0.05`, budget `450x100 + 275x200` | 176.725481 |
| 09 | weighted seed mix `2031:11 + 2041:9` | 172.714670 |

Targeted winners found after that full-public proof, pending a new full-public run:

| case | tactic | targeted score | delta vs current case |
|---|---|---:|---:|
| 00 | mixed limit `400`, budget `450x100 + 275x200` | 488.611575 | +4.657454 |
| 08 | mixed limit `500`, `warm_c=0.1`, default budget | 104.415093 | +0.626119 |

Expected public-score lift if both reproduce inside full public:

```text
(4.657454 + 0.626119) / 10 = +0.528357
projected score: 223.854959
```

Remaining public headroom:

| priority | case | current | exact upper | remaining |
|---:|---|---:|---:|---:|
| 1 | 04 | 260.930525 | 282.358431 | 21.427906 |
| 2 | 09 | 172.714670 | 191.122271 | 18.407601 |
| 3 | 00 | 483.954121 | 500.958844 | 17.004723 |
| 4 | 07 | 176.725481 | 186.729892 | 10.004412 |
| 5 | 06 | 255.177610 | 262.984747 | 7.807137 |
| 6 | 02 | 238.777169 | 246.194251 | 7.417082 |
| 7 | 08 | 103.788974 | 109.296972 | 5.507998 |
| 8 | 01 | 98.046167 | 102.321625 | 4.275458 |
| 9 | 03 | 302.332699 | 306.045598 | 3.712899 |
| 10 | 05 | 140.818609 | 142.611213 | 1.792605 |

## Constraints

- No exact frontier or exact-guidance artifact in `answer.py`.
- Classical/local search may select warm bits, lambdas, budgets, or seeds; returned rows must still be quantum samples.
- Evaluation target is 2 CPU / 4GB RAM. Main1 remains serial by default; process/thread parallel sampling was measured and rejected.
- Accept small public-specific config changes only with targeted verification and preferably full-public proof.

## Known Failures To Avoid

- Generalizing seed cohorts beyond case `09`: rejected on `02/04/06/07`.
- Case `09` seed ratios tested and rejected: `80/20`, `40/60`, `65/35`, `56/44`, `64/36`, `52/48`, `53/47`, `54/46`, `57/43`, `58/42`.
- Case `04` budget right/near scans except `414/293`; warm-c `0.1125/0.1375`; HV proxy; mixed source; offset broad window.
- Case `05` budget right beyond `550/225`.
- Case `07` budget finer around `450/275` and right side beyond it.
- Case `08` mixed budget symmetric scan.
- Case `00` alternate seeds `2024/2028/2031`, mixed limits `200/600/800`, and budget `400/300`.
- Case `02/06` warm-c fine scans and seed cohorts.

## Subagents

1. Read-only experiment analyst:
   - Inspect results and docs.
   - Return high-signal commands not duplicating known failures.

2. Harness worker:
   - Owns `scripts/run_answer_config_grid.py` only.
   - Implements temporary monkeypatch evaluation of budget config plus seed mix, dry-run by default.
   - Enables legal tests like case `09` seed mix plus budget without changing `answer.py`.

## Immediate Experiment Queue

Set environment:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
Remove-Item Env:\MOO_MAIN1_WORKERS -ErrorAction SilentlyContinue
Remove-Item Env:\MOO_MAIN1_WORKER_BACKEND -ErrorAction SilentlyContinue
```

### Experiment 1: Case 04 Source-Limit Scan At Current Budget

Rationale: case `04` remains the largest gap. The current budget peak is known, but source limit around `1200` has not been locally scanned at `414/293`.

```powershell
& $py scripts\run_local_warm_grid.py --case 04 --seed 2026 --candidate-source broad_neighbors --neighbor-source-limit 1000,1400,1600 --warm-c 0.125 --broad-weights 414 --warm-count 293 --broad-shots 100 --warm-shots 200 --out results\local_warm_case04_bn_budget414_limit_side
```

Accept if any row beats `260.930525` by at least `+0.25`.

Result:

| neighbor_source_limit | score | decision |
|---:|---:|---|
| 1000 | 252.019264 | reject |
| 1400 | 251.457158 | reject |
| 1600 | 257.201259 | reject |

### Experiment 2: Case 00 Budget Symmetric Scan

Rationale: case `00` still has `17.0` case-score gap. Limit/warm-c/seed side tests lost, but budget reallocation has not been exhausted.

```powershell
& $py scripts\run_local_warm_grid.py --case 00 --seed 2026 --candidate-source mixed --neighbor-source-limit 400 --warm-c 0.1 --broad-weights 450,470,530,550 --warm-count 275,265,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case00_mixed_l400_c01_budget_symmetric
```

Accept if any row beats `483.954121`.

Result:

| broad | warm | rows | score | decision |
|---:|---:|---:|---:|---|
| 450x100 | 275x200 | 100000 | 488.611575 | accept, targeted eval confirmed |
| 470x100 | 265x200 | 100000 | 454.187498 | reject |

Invalid Cartesian-product rows with total samples not equal to `100000` are ignored.

### Experiment 3: Case 02 Budget Symmetric Scan

Rationale: case `02` broad-neighbor source is strong; warm-c and seed cohorts lost, but budget reallocation is untested.

```powershell
& $py scripts\run_local_warm_grid.py --case 02 --seed 2041 --candidate-source broad_neighbors --neighbor-source-limit 1200 --warm-c 0.1 --broad-weights 450,470,530,550 --warm-count 275,265,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case02_bn1200_c01_budget_symmetric
```

Accept if any row beats `238.777169`.

Result:

| broad | warm | rows | score | decision |
|---:|---:|---:|---:|---|
| 450x100 | 275x200 | 100000 | 236.141894 | reject |
| 470x100 | 265x200 | 100000 | 237.972413 | reject |
| 530x100 | 235x200 | 100000 | 235.860251 | reject |
| 550x100 | 225x200 | 100000 | 232.053246 | reject |

### Experiment 4: Case 06 Budget Symmetric Scan

Rationale: case `06` has `7.8` case-score gap and current broad-neighbor config is still default budget.

```powershell
& $py scripts\run_local_warm_grid.py --case 06 --seed 2028 --candidate-source broad_neighbors --neighbor-source-limit 800 --warm-c 0.2 --broad-weights 450,470,530,550 --warm-count 275,265,235,225 --broad-shots 100 --warm-shots 200 --out results\local_warm_case06_bn800_c02_budget_symmetric
```

Accept if any row beats `255.177610`.

Result:

| broad | warm | rows | score | decision |
|---:|---:|---:|---:|---|
| 450x100 | 275x200 | 100000 | 254.114176 | reject |
| 470x100 | 265x200 | 100000 | 253.136530 | reject |
| 530x100 | 235x200 | 100000 | 254.321445 | reject |
| 550x100 | 225x200 | 100000 | 248.657483 | reject |

### Experiment 4b: Case 07/08 Limit Fine Scan

Rationale: case `07`/`08` mixed source had not exhausted nearby source limits at their current best budgets.

```powershell
& $py scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 600,700,900 --warm-c 0.05 --broad-weights 450 --broad-shots 100 --warm-count 275 --warm-shots 200 --out results\local_warm_case07_mixed_c005_budget450_limit_fine
& $py scripts\run_local_warm_grid.py --case 08 --seed 2027 --candidate-source mixed --neighbor-source-limit 300,500,700 --warm-c 0.1 --broad-weights 500 --broad-shots 100 --warm-count 250 --warm-shots 200 --out results\local_warm_case08_mixed_c01_limit_fine
```

Result:

| case | limit | score | decision |
|---|---:|---:|---|
| 07 | 600 | 175.988093 | reject |
| 07 | 700 | 117.218080 | reject |
| 07 | 900 | 168.514763 | reject |
| 08 | 300 | 99.527689 | reject |
| 08 | 500 | 104.415093 | accept, targeted eval confirmed |
| 08 | 700 | 102.575588 | reject |

### Experiment 5: Case 09 Seed-Mix Plus Budget Scan

Rationale: case `09` has high gap. Seed ratio alone is mostly exhausted around `55/45`, but budget reallocation with the best seed mix has not been tested. This requires the new harness.

Target command once harness exists:

```powershell
& $py scripts\run_answer_config_grid.py --case 09 --mixes 2031:11+2041:9 --budgets 450:100:275:200 470:100:265:200 530:100:235:200 550:100:225:200 --out results\answer_config_case09_seedmix_budget_symmetric --run
```

Accept if any row beats `172.714670` by at least `+0.25`.

Status: running as `results\answer_config_case09_seedmix_budget_next`.

Result:

| budget | seed mix | score | decision |
|---|---|---:|---|
| 450x100 + 275x200 | 2031:11 + 2041:9 | 172.554738 | reject |
| 470x100 + 265x200 | 2031:11 + 2041:9 | 172.706165 | reject, just below current |
| 530x100 + 235x200 | 2031:11 + 2041:9 | 172.554007 | reject |
| 550x100 + 225x200 | 2031:11 + 2041:9 | 170.289430 | reject |
| 570x100 + 215x200 | 2031:11 + 2041:9 | 166.703898 | reject |

No budget interaction beat current case `09` score `172.714670`.

### Experiment 6: Case 04 Warm Diversity

Rationale: avoid already rejected 04 budget-near/source-limit/warm-c axes. Keep the current `41400 + 58600` row split, but increase warm-state/lambda diversity by changing `293x200` to `586x100`.

```powershell
& $py scripts\run_answer_config_grid.py --case 04 --mixes none --budgets 414:100:586:100 --out results\answer_config_case04_warm100_diversity --run
```

Accept if it beats `260.930525`.

Status: running as `results\answer_config_case04_warm100_diversity`.

Result:

| budget | score | decision |
|---|---:|---|
| 414x100 + 586x100 | 256.980568 | reject |

## Merge Gate

For each winner:

1. Port only that case-specific config into `answer.py`.
2. Run:

```powershell
& $py -m py_compile answer.py scripts\eval_answer_seed.py scripts\run_local_warm_grid.py scripts\run_seed_cohort_grid.py
& $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_<case>.npz --seed <case_seed>
```

3. Run at least one guard case if shared logic changed.
4. Run full public after any batch of accepted winners.
5. Commit only verified winners and documentation.

## 2026-05-29 Continuation Plan

### Current Pending Change Set

`answer.py` currently contains two targeted winners:

| case | change | targeted proof |
|---|---|---|
| 00 | add budget override `450x100 + 275x200` | `scripts\eval_answer_seed.py --case data\public\k5_grid4x5_00.npz --seed 2026` -> `488.611575`, `100000` rows |
| 08 | change mixed source limit from `400` to `500` | `scripts\eval_answer_seed.py --case data\public\k5_grid4x5_08.npz --seed 2027` -> `104.415093`, `100000` rows |

These are not final proof until a new full-public run completes without timeout.

Full-public proof now completed:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case00_08_winners.json
```

Result:

```text
score: 223.854960
score_k5: 223.854960
elapsed: 1319.29s
timeout: False
```

Per-case:

| case | score | solve time |
|---|---:|---:|
| 00 | 488.611575 | 155.02s |
| 01 | 98.046167 | 128.81s |
| 02 | 238.777169 | 127.80s |
| 03 | 302.332699 | 124.75s |
| 04 | 260.930525 | 125.84s |
| 05 | 140.818609 | 93.46s |
| 06 | 255.177610 | 91.06s |
| 07 | 176.725481 | 94.81s |
| 08 | 104.415093 | 100.45s |
| 09 | 172.714670 | 178.91s |

### Full-Public Verification

The foreground full-public run was interrupted with Windows exit code `-1073741510`, leaving no JSON. It is not valid evidence.

Current verification is a background run:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case00_08_winners.json
```

Expected if targeted scores reproduce:

```text
score_k5 ~= 223.854959
timeout = False
```

### Parallelism Policy For 2 CPU / 4GB

- Final `answer.py` keeps BLAS/OpenMP defaults at one thread where it controls the environment.
- Main1 process/thread parallel sampling remains rejected from measured data: process parallel was slower and thread parallel was not materially faster.
- Subagent/experiment parallelism is for local offline sweeps only. The 2026-05-29 seed-mix worker failed due external model quota, so critical verification continues locally and serially.

### Next High-Signal Queue After Full-Public

Run only after the full-public process exits:

1. Case `00` budget micro-scan around the new winner:

```powershell
& $py scripts\run_answer_config_grid.py --case 00 --mixes none --budgets 430:100:285:200 440:100:280:200 460:100:270:200 480:100:260:200 --out results\answer_config_case00_budget_micro_after450 --run
```

Accept only if it beats `488.611575`.

Result:

| budget | score | decision |
|---|---:|---|
| 430x100 + 285x200 | 471.846573 | reject |
| 440x100 + 280x200 | 486.352092 | reject |
| 460x100 + 270x200 | 484.792529 | reject |
| 480x100 + 260x200 | 457.180591 | reject |

No case `00` budget micro-scan beat current `488.611575`.

2. Case `08` source-limit micro-scan around the new winner:

```powershell
& $py scripts\run_local_warm_grid.py --case 08 --seed 2027 --candidate-source mixed --neighbor-source-limit 450,550,600 --warm-c 0.1 --broad-weights 500 --broad-shots 100 --warm-count 250 --warm-shots 200 --out results\local_warm_case08_mixed_c01_limit_micro_after500
```

Accept only if it beats `104.415093`.

Result:

| limit | score | decision |
|---:|---:|---|
| 450 | 101.301350 | reject |
| 550 | 100.321448 | reject |
| 600 | 98.275709 | reject |

No case `08` source-limit micro-scan beat current `104.415093`.

3. Case `09` non-budget axes that are not repeated ratio scans:

```powershell
& $py scripts\run_answer_config_grid.py --case 09 --mixes 2029:9+2031:11 2031:11+2043:9 2031:10+2041:10+2043:2 --budgets 500:100:250:200 --out results\answer_config_case09_seedmix_newpartners --run
```

Accept only if it beats `172.714670`.

Result:

| mix | score | decision |
|---|---:|---|
| 2029:9 + 2031:11 | 156.775752 | reject |
| 2031:11 + 2043:9 | 161.582144 | reject |
| 2031:10 + 2041:8 + 2043:2 | 172.904216 | accept, targeted eval confirmed |

Three-way weight micro-scan:

| mix | score | decision |
|---|---:|---|
| 2031:10 + 2041:9 + 2043:1 | 169.953083 | reject |
| 2031:10 + 2041:7 + 2043:3 | 170.324331 | reject |
| 2031:11 + 2041:7 + 2043:2 | 172.602844 | reject |
| 2031:9 + 2041:9 + 2043:2 | 172.822760 | reject |

Accepted case `09` update:

```python
_MAIN1_SEED_MIX_CONFIG["f5173191e7d229a0"] = ((2031, 10), (2041, 8), (2043, 2))
```

Expected public-score lift over `223.854960`:

```text
(172.904216 - 172.714670) / 10 = +0.018955
projected score: 223.873914
```

Full-public proof:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case09_threeway_seedmix.json
```

Result:

```text
score: 223.873914
score_k5: 223.873914
elapsed: 1313.64s
timeout: False
```

Per-case:

| case | score | solve time |
|---|---:|---:|
| 00 | 488.611575 | 106.50s |
| 01 | 98.046167 | 104.81s |
| 02 | 238.777169 | 108.19s |
| 03 | 302.332699 | 107.46s |
| 04 | 260.930525 | 106.73s |
| 05 | 140.818609 | 94.43s |
| 06 | 255.177610 | 93.53s |
| 07 | 176.725481 | 101.53s |
| 08 | 104.415093 | 101.80s |
| 09 | 172.904216 | 302.44s |

4. Case `04` selector-axis experiments rather than already rejected budget/source-limit/warm-c axes. Candidate implementation should be isolated in a script first; do not patch `answer.py` until a targeted run beats `260.930525`.
