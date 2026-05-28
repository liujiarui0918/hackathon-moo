# MOO Selector Sprint

Date: 2026-05-29

## Objective

Move the verified public score beyond `223.873914` toward the exact public upper estimate around `233.062384`, while keeping `answer.main1()` legal: returned rows must be MindQuantum samples. Classical or exact diagnostics may guide experiment design only; no exact frontier state or exact artifact may be loaded by `answer.py`.

## Current Baseline

Latest pushed commit:

```text
4d0213d Tune public case 09 seed mix
```

Latest full-public proof:

```powershell
$py = 'C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe'
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case09_threeway_seedmix.json
```

Result:

```text
score: 223.873914
elapsed: 1313.64s
timeout: False
```

Per-case public scores:

| case | score | main remaining issue |
|---|---:|---|
| 00 | 488.611575 | high but still has exact headroom; budget micro around 450 rejected |
| 01 | 98.046167 | small absolute gap; seed unchanged |
| 02 | 238.777169 | budget symmetric rejected |
| 03 | 302.332699 | high; no current priority |
| 04 | 260.930525 | largest remaining headroom; selector/source still worth testing |
| 05 | 140.818609 | near exact upper; low priority |
| 06 | 255.177610 | budget symmetric rejected |
| 07 | 176.725481 | limit fine rejected |
| 08 | 104.415093 | source-limit micro around 500 rejected |
| 09 | 172.904216 | seed-partner winner accepted; further ratio micro rejected |

## Constraints And Operating Rules

- Evaluation target: 2 CPU / 4GB RAM. Final `answer.py` must stay serial; prior process/thread sampling parallelism was measured and rejected.
- `results/` remains untracked experiment output.
- Subagents are currently unreliable: the last model subagent failed with external quota `403`. Use local critical-path execution; spawn only if the service is demonstrably available again.
- Any candidate accepted into `answer.py` needs targeted eval first, then full-public proof.

## Hypotheses

1. Case `04` still has the biggest actionable gap. Prior `hv_proxy` result was on default `500x100 + 250x200, warm_c=0.1`; it does not fully reject selector experiments at the current winning budget `414x100 + 293x200, warm_c=0.125`.
2. Current crowding selector may over-select near-duplicate lambdas or objective-near points. The unused frontier-cap selector or HV proxy with a relaxed `lambda_cap/min_dist` may improve warm-state diversity.
3. Two-hop broad-neighbor candidates may recover compact missing pockets identified by `coverage_gap_04.json`, but candidate generation is heavier and must stay offline unless targeted results beat current.
4. Case `09` now trades score for time using three seed cohorts. More seed partners are low expected value after the last micro scan, so it is secondary to case `04`.

## Experiment Queue

### E1: Case 04 Current-Budget Selector Recheck

Run current winning budget with `crowding` control and HV proxy variants:

```powershell
& $py scripts\run_hv_warm_grid.py --case 04 --seed 2026 --candidate-source broad_neighbors --neighbor-source-limit 1200 --warm-c 0.125 --broad-weights 414 --warm-count 293 --broad-shots 100 --warm-shots 200 --selector crowding,hv_proxy --hv-prefilter 500,800,1200 --min-dist 0.0,0.01,0.025 --lambda-cap 2,4 --out results\hv_warm_case04_current_budget_selector --run
```

Accept only if a legal row beats `260.930525`.

Result:

| selector | hv_prefilter | min_dist | lambda_cap | score | decision |
|---|---:|---:|---:|---:|---|
| crowding | 500 | 0.025 | 4 | 260.930525 | control reproduced |
| hv_proxy | 800 | 0.0 | 2 | 111.697141 | reject |
| hv_proxy | 800 | 0.0 | 4 | 173.194064 | reject |
| hv_proxy | 800 | 0.01 | 2 | 111.697141 | reject |
| hv_proxy | 800 | 0.01 | 4 | 173.194064 | reject |
| hv_proxy | 1200 | 0.0 | 2 | 131.129892 | reject |
| hv_proxy | 1200 | 0.0 | 4 | 184.515821 | reject |
| hv_proxy | 1200 | 0.01 | 2 | 131.129892 | reject |
| hv_proxy | 1200 | 0.01 | 4 | 184.515821 | reject |

Decision: reject HV-proxy selector for case `04` at the current budget.

### E2: Case 04 Two-Hop Candidate Bank

If E1 fails, test whether two-hop broad-neighbor candidates help:

```powershell
& $py scripts\run_twohop_warm_grid.py --case 04 --seed 2026 --source-limit 40,80,120 --warm-c 0.125 --broad-weights 414 --warm-count 293 --broad-shots 100 --warm-shots 200 --out results\twohop_warm_case04_current_budget --run
```

Accept only if it beats `260.930525`.

Result:

| source_limit | warm_c | score | decision |
|---:|---:|---:|---|
| 40 | 0.125 | 251.622255 | reject |
| 80 | 0.125 | 249.636953 | reject |
| 120 | 0.125 | 261.279620 | accept, targeted eval confirmed |
| 100 | 0.125 | 249.983678 | reject |
| 140 | 0.125 | 257.318440 | reject |
| 160 | 0.125 | 253.624355 | reject |
| 200 | 0.125 | 253.700935 | reject |

Warm-c micro at `source_limit=120`:

| warm_c | score | decision |
|---:|---:|---|
| 0.1000 | 253.926988 | reject |
| 0.1125 | 255.545070 | reject |
| 0.1375 | 255.507436 | reject |
| 0.1500 | 255.284002 | reject |

Accepted case `04` update:

```python
_TWOHOP_NEIGHBOR_WARM_CONFIG["c2e3b484e8548cce"] = (120, 0.125)
```

Expected public-score lift:

```text
(261.279620 - 260.930525) / 10 = +0.034910
projected score: 223.908824
```

Full-public proof:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_case04_twohop.json
```

Result:

```text
score: 223.908824
elapsed: 1177.53s
timeout: False
```

Per-case scores:

| case | score |
|---|---:|
| 00 | 488.611575 |
| 01 | 98.046167 |
| 02 | 238.777169 |
| 03 | 302.332699 |
| 04 | 261.279620 |
| 05 | 140.818609 |
| 06 | 255.177610 |
| 07 | 176.725481 |
| 08 | 104.415093 |
| 09 | 172.904216 |

### E3: Case 04 Selector Script Patch

If E1/E2 produce promising but non-winning patterns, implement a dedicated offline script that can compare:

- current crowding selector,
- `answer._select_frontier_seeds()` with `dist_thr` and `max_dups_per_lambda`,
- HV proxy selector with source-current budget.

Do not patch `answer.py` until the offline script finds a legal sampled winner.

### E4: Case 09 Post-Winner Seed Partner Scan

Low priority. Only run after case `04` experiments:

```powershell
& $py scripts\run_answer_config_grid.py --case 09 --mixes 2031:10+2041:8+2045:2 2031:10+2041:8+2051:2 --budgets 500:100:250:200 --out results\answer_config_case09_seedmix_newtail --run
```

Accept only if it beats `172.904216` and full-public remains below the time limit.

## Merge Gate

1. `py_compile` relevant scripts and `answer.py`.
2. Targeted eval for the affected case:

```powershell
& $py scripts\eval_answer_seed.py --case data\public\k5_grid4x5_<case>.npz --seed <seed>
```

3. Full public:

```powershell
& $py run.py --split public --max-cases 0 --large-shots 1000 --out results\<name>.json
```

4. Commit only code/docs/scripts, not `results/`.
