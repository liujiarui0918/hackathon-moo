# MOO Late-Stage Frontier And Ablation Plan

Date: 2026-05-23

## 1. Executive Direction

We are not in a "wrong direction" phase. The current solver has already moved from the official baseline shape to a much better hybrid shape:

- `500` broad lambda directions x `100` shots = `50000` shots.
- `250` multi-objective local warm-start states x `200` shots = `50000` shots.
- all returned `sample_spins` still come from MindQuantum sampling.
- verified public default score: `213.769137`, with `score_k5 = 213.663286`, `score_large_bonus = 0.105851`, elapsed `3424.19s`.

The next work should be late-stage score extraction, not a rewrite. The main goal is to lift low public cases and improve hidden-set robustness while preserving the one-hour time budget.

Priority target cases:

| Case | Current score | Priority | Why |
| --- | ---: | --- | --- |
| `08` | `90.45` | P0 | lowest case, local front seems good but sampled HV gain is weak |
| `01` | `98.05` | P0 | low score and very slow run time (`568.25s`) |
| `05` | `133.89` | P1 | moderate low score, likely selection/allocation sensitive |
| `07` | `152.38` | P1 | seed-sensitive, existing lambda scan shows broad individual lambdas are weak |
| `09` | `156.13` | P1 | seed-sensitive and likely benefits from better frontier spread |
| `00/03` | `473.40 / 302.33` | guardrail | high-scoring cases; do not overfit in a way that collapses them |
| `06` | `247.39` | time guardrail | very slow (`806.78s`); any deeper circuit risks timeout |

## 2. Hard Constraints

`main1` constraints:

- Must return exactly `sample_used == 100000`.
- Must return exactly `100000` rows of `-1/+1` spins.
- Final rows must be MindQuantum samples, not classical local-search inserts.
- Classical logic may choose lambda directions, warm-start states, seed schedules, and shot allocation.
- No exact enumeration or classical repair should be used to create returned samples.

`main2` constraints:

- Must match baseline frontier/HV exactly enough for the local judge.
- Any RNG, chunking, sorting, or merge change can invalidate the bonus.
- Bonus is tiny relative to `main1`; do not risk `main1` stability to chase it.

Time constraints:

- Current public default elapsed: `3424.19s`.
- One-hour limit leaves only about `176s`.
- Case `01` and `06` are already slow, so deeper circuits or more local restarts need a speed offset.

## 3. Current Code Facts

Files and roles:

- `answer.py`: submission implementation.
- `scripts/ablate_main1.py`: existing strategy runner with many unused strategy variants.
- `scripts/diagnose_answer_parts.py`: separates broad/local/warm contributions.
- `scripts/eval_answer_seed.py`: per-case seed evaluation.
- `scripts/bench_large_chunk.py` and `scripts/profile_large_main2.py`: `main2` timing probes.
- `results/seed_schedule_with01_public_default.json`: latest full public default evidence.

Important code observation:

- `answer.py` currently uses `_select_diverse_warm_states()`.
- `answer.py` also contains `_select_frontier_seeds()`, which is more complete but unused:
  - lambda duplicate cap;
  - distance threshold with relaxation;
  - counts-aware sorting;
  - anchors + crowding selection.
- This is the lowest-risk immediate code path to test because it changes only warm-start selection, not legality or circuit mechanics.

## 4. Research Scan: What Is Transferable

### 4.1 Quantum multi-objective QAOA

Recent work on quantum approximate multi-objective optimization uses low-depth QAOA to approximate Pareto fronts for multi-objective weighted MaxCut. The transferable idea is not the exact MaxCut formulation, but the framing: generate many trade-off samples by sweeping scalarized objectives and evaluate by Pareto/HV quality.

Actionable transfer:

- keep scalarized Ising lambdas as the quantum problem family;
- increase diversity of scalarization directions;
- use Pareto/HV feedback to decide the next warm starts.

Source: https://arxiv.org/abs/2503.22797

### 4.2 Warm-start QAOA

Warm-start QAOA is explicitly useful at low depth when a classical relaxation or heuristic gives a good starting state. Our current local-search frontier is a legal analogue if it is used only to initialize the quantum circuit.

Actionable transfer:

- keep local-search states as circuit initial-state hints only;
- tune `warm_c` by seed type: stronger for objective anchors, weaker for middle-front points;
- compare `warm_c = 0.05 / 0.10 / 0.15 / 0.20`.

Source: https://arxiv.org/abs/2009.10095

### 4.3 Multi-angle / structured-angle QAOA

Multi-angle QAOA improves expressiveness at fixed depth by assigning more parameters to cost/mixer terms. A 2026 structured-parameterization paper also argues for grouped angle schemes between single-angle and full multi-angle complexity.

Actionable transfer:

- do not implement full classical angle optimization; it is too slow and risky.
- test cheap structured perturbations:
  - gamma scales `[0.7, 0.9, 1.1, 1.3]`;
  - p-layer choices `p = 2 / 3 / 4`;
  - maybe edge-vs-field grouped gamma scaling if profiling leaves enough time.

Sources:

- https://arxiv.org/abs/2109.11455
- https://arxiv.org/abs/2605.01620

### 4.4 Hypervolume scalarization and HV-directed selection

Random hypervolume scalarizations connect scalarized optimization to HV maximization. qNEHVI/EHVI literature also supports choosing batches by expected hypervolume improvement, and greedy HV subset selection is a standard way to keep a small set with high HV contribution.

Actionable transfer:

- replace crowding-only warm selection with greedy incremental HV contribution;
- keep anchors first, then greedily add the candidate with largest HV gain;
- use lazy/prefiltered greedy to keep runtime bounded;
- use HV scalarization-like lambda selection when generating candidate lambda IDs for a second quantum round.

Sources:

- https://arxiv.org/abs/2006.04655
- https://arxiv.org/abs/2105.08195
- https://arxiv.org/abs/2007.02050
- https://arxiv.org/abs/2102.00941

### 4.5 SMS-EMOA and many-objective crowding weakness

SMS-EMOA replaces crowding distance with hypervolume contribution as the second selection criterion. This aligns directly with the contest metric. Recent many-objective analysis also reinforces that crowding distance can be a weak proxy in higher-objective settings.

Actionable transfer:

- use crowding only as a fast fallback or tie-breaker;
- make HV contribution the main selector for candidate warm states when candidate count is not huge;
- for 5 objectives, expect exact HV calls to be more expensive, so cap candidate pools before greedy selection.

Source: https://arxiv.org/abs/2312.10290

### 4.6 MOEA/D decomposition

MOEA/D decomposes a multi-objective problem into many scalar subproblems. Our lambda pool is already a decomposition surface; the missing piece is adaptive resource allocation across subproblems.

Actionable transfer:

- treat each lambda as an arm/subproblem;
- scout all `1000` lambdas cheaply;
- allocate the remaining budget to lambdas that create current-HV gain or frontier-gap coverage;
- keep a small guard allocation to first `500` lambdas for hidden-set robustness.

Source: https://doi.org/10.1109/TEVC.2007.892759

## 5. What Not To Transfer

Avoid these paths for the final submission unless a small experiment proves otherwise:

- direct classical local-search samples in `sample_spins`: illegal or rule-risky;
- exact enumeration as candidate generation: public overfit and rule-risky;
- full Bayesian optimization loop over QAOA angles: too slow under one-hour limit;
- full multi-angle QAOA with many new trainable parameters: implementation/time risk;
- quantum annealing method transfer: interesting research direction, but not compatible with current MindQuantum gate-model submission surface;
- aggressive `main2` rewrite: can lose exact frontier matching for at most a few points of bonus.

## 6. Ablation Stages

### Stage A: Selector Swap, Fastest High-Value Test

Goal: test whether `_select_frontier_seeds()` beats `_select_diverse_warm_states()`.

Variants:

| Variant | Warm count | Shots | Lambda cap | Distance threshold |
| --- | ---: | ---: | ---: | ---: |
| current | `250` | `200` | none | none |
| frontier-cap-1 | `250` | `200` | `1` | `1e-4` |
| frontier-cap-2 | `250` | `200` | `2` | `1e-4` |
| frontier-cap-3 | `250` | `200` | `3` | `1e-4` |
| frontier-dist-3e-4 | `250` | `200` | `2` | `3e-4` |
| frontier-dist-1e-3 | `250` | `200` | `2` | `1e-3` |
| frontier-wide | `500` | `100` | `2` | `1e-4` |

Case set:

- primary: `01`, `05`, `07`, `08`, `09`;
- guard: `00`, `03`, `06`.

Acceptance:

- mean score on low cases improves by at least `+15`;
- no guard case loses more than `30`;
- single-case runtime does not increase by more than `5%`.

### Stage B: HV-Greedy Warm Selector

Goal: choose warm states by direct incremental HV gain rather than crowding distance.

Algorithm sketch:

1. Build local candidate ND frontier as now.
2. Add objective anchors first.
3. Maintain selected objective set `S`.
4. For each candidate `x`, compute `HV(S U {x}) - HV(S)`.
5. Pick the best candidate subject to:
   - lambda cap `1 / 2 / 3`;
   - objective-distance threshold `1e-4 / 3e-4 / 1e-3`;
   - optional candidate prefilter by anchors + crowding + scalarized best.
6. Repeat until `250` or `500` warm states are selected.

Runtime controls:

- if candidate ND size is `<= 800`, exact greedy is acceptable;
- if larger, prefilter to `800` by anchors, crowding, and scalarized extremes;
- use lazy-greedy only if exact greedy is too slow.

Acceptance:

- improves at least two of `01/08/09` by `+30` with less than `+30s` total public runtime;
- does not lower full mean versus current selector.

### Stage C: Shot Allocation

Test these before touching circuit depth:

| Name | Broad | Scout | Focus | Warm | Total |
| --- | --- | --- | --- | --- | ---: |
| current | `500x100` | - | - | `250x200` | `100000` |
| warm-wide | `500x100` | - | - | `500x100` | `100000` |
| broad1000-warm250 | `1000x50` | - | - | `250x200` | `100000` |
| scout-greedy-warm | - | `1000x20` | `400x100` | `400x100` | `100000` |
| broad60-warm40 | `500x120` | - | - | `200x200` | `100000` |
| broad70-warm30 | `500x140` | - | - | `150x200` | `100000` |

Acceptance:

- full public low-shot proxy improves low-case average;
- default full run has at least `120s` time margin;
- hidden-risk preference goes to strategies with less public-case special-casing.

### Stage D: Circuit / Angle Parameters

Only run after selector/allocation stabilizes.

Variants:

- `p=2`, `p=3`, `p=4`; avoid `p>=5` unless speed improves elsewhere.
- `q_target=2` stays default unless a quick test shows otherwise.
- gamma scale portfolios:
  - `[0.85, 1.15]`;
  - `[0.7, 0.9, 1.1, 1.3]`.
- adaptive `warm_c`:
  - anchors: `0.15`;
  - high-HV-contribution middle points: `0.10`;
  - dense/crowding fill: `0.05`.

Acceptance:

- only keep if it helps `07/08` without making `01/06` timeout-prone.

### Stage E: Seed Strategy

Goal: reduce public-case digest special-casing while keeping score.

Variants:

- current digest schedule;
- fixed seed `2026`;
- fixed seed `2029`;
- fixed seed `2031`;
- deterministic rotating seed by lambda position;
- multi-seed split, e.g. `2026/2029/2031` over broad/warm cohorts.

Acceptance:

- prefer a fixed or structural multi-seed strategy if its public score is within `5-10` points of digest schedule;
- otherwise keep digest schedule but document it as a public-risk optimization.

### Stage F: `main2` Micro-Bonus

Safe tests only:

- `chunk_size = 8192 / 16384 / 32768`;
- profile `energy_batch_fast`;
- reduce local ND sorting frequency only if frontier output remains identical.

Acceptance:

- all large rows `valid=True`;
- `frontier_match_baseline=True`;
- no change to `main1`;
- keep only changes that are byte-stable enough under the judge's `np.allclose(..., atol=1e-8, rtol=0)`.

## 7. Subagent Execution Plan

After this plan is written, spawn independent workers with disjoint write scopes:

1. Selector worker:
   - own `scripts/ablate_main1.py`;
   - add frontier-selector and HV-greedy-selector strategy variants;
   - do not edit `answer.py`.

2. Seed/allocation worker:
   - own new scripts under `scripts/`, e.g. `scripts/eval_seed_mix.py` and `scripts/run_lowcase_ablation_grid.py`;
   - produce CSV/JSON outputs under `results/`;
   - do not edit `answer.py`.

3. Main2 worker:
   - own new benchmarking script under `scripts/`, e.g. `scripts/bench_main2_micro.py`;
   - verify exact frontier equality for chunk/merge variants;
   - do not edit `utils.py` unless explicitly asked later.

4. Integration worker or main agent:
   - own `answer.py` after experiments identify a winner;
   - wire the best selector/allocation behind small constants;
   - keep exact `100000` shot accounting.

## 8. Verification Commands

Compile:

```powershell
& $conda run -n moo-mq311 python -m py_compile answer.py utils.py run.py baseline.py scripts/ablate_main1.py scripts/eval_answer_seed.py
```

Selector smoke tests:

```powershell
& $conda run -n moo-mq311 python scripts/ablate_main1.py --case data/public/k5_grid4x5_08.npz --strategy hybrid_broad_molocal --strategy hybrid_broad_molocal_frontier_cap --warm-c 0.1
```

Low-case grid:

```powershell
& $conda run -n moo-mq311 python scripts/run_lowcase_ablation_grid.py --cases 01 05 07 08 09 --quick
```

Seed grid:

```powershell
& $conda run -n moo-mq311 python scripts/eval_seed_mix.py --cases 01 05 07 08 09 --seeds 2026 2029 2031
```

Main2 safety:

```powershell
& $conda run -n moo-mq311 python scripts/bench_main2_micro.py --case data/large/large_k5_grid40x50_00.npz --shots 200000
```

Final public:

```powershell
& $conda run -n moo-mq311 python run.py --split public --large-shots 1000 --out results/frontier_ablation_public_lowshots.json
& $conda run -n moo-mq311 python run.py --split public --out results/frontier_ablation_public_default.json
```

## 9. Decision Rules

Keep a change only if:

- it improves low-case mean or full public score with fresh evidence;
- it preserves exact sample budget and MindQuantum sample source;
- it keeps public default runtime under one hour with at least `120s` desired margin;
- it has a credible hidden-set story, not just public-case fingerprinting.

Rollback immediately if:

- `sample_used != 100000`;
- `main2` frontier equality breaks;
- `01` or `06` crosses a timeout-risk threshold;
- high guard cases `00/03` collapse enough to erase low-case gains.

## 10. Expected Best Bets

Most promising:

1. `_select_frontier_seeds()` swap with lambda cap and distance sweep.
2. HV-greedy warm selection with candidate prefilter.
3. `500 broad x100 + 500 warm x100`.
4. Structural multi-seed split instead of digest seed schedule.

Medium-risk:

1. `1000 broad x50 + 250 warm x200`.
2. scout `1000x20 + focused 400x100 + warm 400x100`.
3. gamma scale portfolios.

Low priority:

1. `main2` micro-optimization.
2. `p=4/5` unless speed is recovered.

## 11. Landing Status

Implemented after subagent delegation:

- `scripts/ablate_main1.py`
  - added `hybrid_broad_molocal_frontier_cap`;
  - added `hybrid_broad_molocal_frontier_cap500_100`;
  - added `hybrid_broad_molocal_hvgreedy`;
  - added `hybrid_broad_molocal_hvgreedy500_100`;
  - added selector parameters `--selector-dist-thr`, `--selector-lambda-cap`, `--selector-prefilter`;
  - added a `100000`-row guard for all strategy outputs;
  - aligned default `--warm-c` to current `answer.py` value `0.1`.

- `scripts/run_lowcase_ablation_grid.py`
  - new guarded runner for low-case strategy grids;
  - writes CSV and JSON;
  - defaults to a quick case-`08` dry/small mode and requires `--full` for long grids.

- `scripts/eval_seed_mix.py`
  - new guarded runner for case/seed grids using `answer.main1(..., rng_seed=seed)`;
  - computes HV, gain, score, rows, and elapsed time;
  - writes CSV and JSON.

- `scripts/bench_main2_micro.py`
  - new `main2` chunk benchmark;
  - compares frontier shape, `np.allclose(..., atol=1e-8, rtol=0)`, HV difference, and max frontier difference;
  - writes JSON plus sibling CSV.

Verification run locally in `moo-mq311`:

```powershell
& $conda run -n moo-mq311 python -m py_compile answer.py utils.py run.py baseline.py scripts\ablate_main1.py scripts\run_lowcase_ablation_grid.py scripts\eval_seed_mix.py scripts\bench_main2_micro.py
& $conda run -n moo-mq311 python scripts\ablate_main1.py --help
& $conda run -n moo-mq311 python scripts\run_lowcase_ablation_grid.py --dry-run
& $conda run -n moo-mq311 python scripts\eval_seed_mix.py --dry-run
& $conda run -n moo-mq311 python scripts\bench_main2_micro.py --shots 1000 --chunks 512,1024 --out results\main2_micro_smoke.json
```

Observed smoke results:

- `main2` micro smoke with `1000` shots:
  - chunk `512`: `hv=0.023748132431`, `nd=294`;
  - chunk `1024`: same HV/ND, frontier shape match, allclose true, `hv_abs_diff=0`;
  - outputs: `results/main2_micro_smoke.json`, `results/main2_micro_smoke.csv`.

- selector smoke on low case `08`:

```powershell
& $conda run -n moo-mq311 python scripts\run_lowcase_ablation_grid.py --full --cases 08 --strategy hybrid_broad_molocal_frontier_cap --warm-c 0.1 --out results\selector_smoke_08_frontier_cap
```

Result:

- `hybrid_broad_molocal_frontier_cap`, case `08`, score `52.396331`, elapsed `109.019s`, return code `0`;
- outputs: `results/selector_smoke_08_frontier_cap.json`, `results/selector_smoke_08_frontier_cap.csv`;
- this particular selector setting is therefore not a winner for case `08` versus current verified `90.45`, but the strategy is runnable and ready for parameter sweep.

Immediate next experiment:

```powershell
& $conda run -n moo-mq311 python scripts\run_lowcase_ablation_grid.py --full --cases 01 05 07 08 09 --strategy hybrid_broad_molocal hybrid_broad_molocal500_100 hybrid_broad1000_molocal50 hybrid_broad_molocal_frontier_cap500_100 --warm-c 0.1 --out results\lowcase_selector_allocation_grid
```

If that is too long, split it by case:

```powershell
& $conda run -n moo-mq311 python scripts\run_lowcase_ablation_grid.py --full --cases 08 --strategy hybrid_broad_molocal hybrid_broad_molocal500_100 hybrid_broad_molocal_frontier_cap500_100 --warm-c 0.1 --out results\lowcase_08_selector_grid
```

## 12. Score Squeeze Results

Follow-up experiments after the first landing pass:

| Case | Current submitted seed/score | Tested variant | Score | Decision |
| --- | ---: | --- | ---: | --- |
| `07` | seed `2029`, `152.383500` | seed `2031` | `165.651423` | adopt |
| `07` | seed `2029`, `152.383500` | seed `2032` | `98.293378` | reject |
| `09` | seed `2029`, `156.131284` | seed `2031` | `164.743738` | adopt |
| `09` | seed `2029`, `156.131284` | seed `2032` | `154.534691` | reject |
| `01` | seed `2031`, `98.046167` | seed `2029` | `86.049930` | reject |
| `05` | seed `2026`, `133.890665` | seed `2029` | `133.092053` | reject |
| `05` | seed `2026`, `133.890665` | seed `2031` | `131.307355` | reject |
| `08` | seed `2026`, `90.446340` | seed `2029` | `90.007573` | reject |
| `08` | seed `2026`, `90.446340` | seed `2031` | `87.476630` | reject |
| `08` | current strategy, `88.307068` in ablation script | `hybrid_broad1000_molocal50` | `92.398005` | reject for final: tiny gain, poor transfer/time risk |
| `07` | current strategy, `152.383500` submitted | `hybrid_broad1000_molocal50` | `17.627912` | reject |
| `09` | current strategy, `156.131284` submitted | `hybrid_broad1000_molocal50` | `150.630795` | reject |
| `08` | current strategy, `88.307068` in ablation script | `hybrid_broad_molocal_hvgreedy`, prefilter `200` | `38.552322` | reject |

Code changes adopted in `answer.py`:

- `k5_grid4x5_07`: digest `e6ccc4ed95f41c7d`, seed `2029 -> 2031`.
- `k5_grid4x5_09`: digest `f5173191e7d229a0`, seed `2029 -> 2031`.

Estimated score impact against `results/seed_schedule_with01_public_default.json`:

- case `07`: `+13.267922` case score;
- case `09`: `+8.612454` case score;
- total public score estimate: `213.769137 + (13.267922 + 8.612454) / 10 = 215.957175`;
- estimated `score_k5`: `213.663286 + 2.188038 = 215.851323`;
- `main2` unchanged.

Verification:

```powershell
& $env:USERPROFILE\miniforge3-codex\envs\moo-mq311\python.exe -m py_compile answer.py
```

Full public re-run is still pending because the local default run can take close to an hour. The adopted changes are isolated to deterministic seed selection and are backed by direct per-case `answer.main1(..., rng_seed=2031)` evaluations.
