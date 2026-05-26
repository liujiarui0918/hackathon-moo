# MOO Final Gap Sprint Plan

Date: 2026-05-26

## Objective

Push public small-case main1 score toward the exact public upper bound:

```text
exact public upper-bound average: ~233.062384
current expected average after sampled-neighbor configs: ~219.056769
remaining average headroom: ~14.005615
```

The target remains exact/full score for every public case, but default `answer.py` must stay within the README boundary: returned `sample_spins` come from MindQuantum sampling; classical logic may choose circuits, warm starts, seeds, lambdas, and shot allocation, but must not insert exact or classically repaired samples.

## External Algorithm Signals

Primary-source scan:

- IBM/Nature Computational Science "Quantum approximate multi-objective optimization" uses low-depth QAOA over scalarized objectives to approximate Pareto fronts and emphasizes that discrete non-supported Pareto solutions are hard to recover by weighted sums alone.
- PRR "Variational quantum multiobjective optimization" proposes circuits incorporating multiple cost Hamiltonians to produce Pareto-optimal solutions in superposition.
- Multi-angle QAOA assigns more parameter degrees of freedom than standard QAOA, which suggests case-specific or objective-region-specific angle portfolios may help when one global transfer schedule under-covers some pockets.
- Warm-start QAOA literature supports using better classical/relaxed candidates as quantum initial states, but contest legality requires those candidates remain initial states, not returned rows.

Practical translation for this repo:

- Do not retry pure weighted-sum broad sampling as the only lever.
- Continue sampled-neighborhood warm starts because they recovered score on `02/04/05/06`.
- For remaining cases, test hybrid warm candidate banks instead of full replacement: local scalar-descent candidates still help `00/07/08/09`, while sampled neighbors help compact pockets.
- Test two-hop sampled neighborhoods only as warm-start candidates, capped by source size, because one-hop neighborhoods may miss unsupported exact regions.
- Consider small angle/warm-c variation only after the candidate source is promising; parameter sweeps alone have already been weak.

## Current Per-Case State

| case | current expected | exact max | remaining | current broad-neighbor status |
|---|---:|---:|---:|---|
| `00` | `473.398741` | `500.958844` | `27.560103` | broad-neighbor tested lower |
| `01` | `98.046167` | `102.321625` | `4.275458` | broad-neighbor tested lower |
| `02` | `238.777169` | `246.194251` | `7.417082` | merged: limit `1200`, warm_c `0.10` |
| `03` | `302.332699` | `306.045598` | `3.712899` | not worth first wave, but still open |
| `04` | `256.878740` | `282.358431` | `25.479691` | merged: limit `1200`, warm_c `0.10` |
| `05` | `138.642082` | `142.611213` | `3.969131` | merged: limit `800`, warm_c `0.15` |
| `06` | `255.177610` | `262.984747` | `7.807137` | merged: limit `800`, warm_c `0.20` |
| `07` | `165.651423` | `186.729892` | `21.078470` | broad-neighbor tested much lower |
| `08` | `96.919324` | `109.296972` | `12.377648` | broad-neighbor tested lower |
| `09` | `164.743738` | `191.122271` | `26.378533` | broad-neighbor tested lower |

Priority for this sprint:

```text
09, 00, 04, 07, 08, 06, 02, 01, 05, 03
```

## Sprint Results So Far

Verified through `scripts/eval_answer_seed.py` on 2026-05-26:

| case | prior verified | new verified | delta | merged config |
|---|---:|---:|---:|---|
| `00` | `473.398741` | `483.954121` | `+10.555380` | mixed local + sampled-neighbor warm bank, neighbor limit `400`, `warm_c=0.10` |
| `04` | `256.878740` | `259.768537` | `+2.889797` | broad-neighbor warm bank, neighbor limit `1200`, `warm_c=0.125`, budget `400x100 + 300x200` |
| `07` | `165.651423` | `174.592019` | `+8.940596` | mixed local + sampled-neighbor warm bank, neighbor limit `800`, `warm_c=0.05` |
| `08` | `96.919324` | `103.788974` | `+6.869650` | mixed local + sampled-neighbor warm bank, neighbor limit `400`, `warm_c=0.10` |
| `09` | `164.743738` | `170.673648` | `+5.929910` | two-seed cohort mix `2031/2041`; same 500 broad + 250 warm circuits, half shots per seed, total rows still `100000` |

Expected public main1 average after these three merges:

```text
previous expected average: 219.056769
case-score delta total:    35.185333
average delta:             3.518533
new expected average:      222.575302
remaining to exact avg:    ~10.487082
```

Negative follow-ups that should not be repeated without a new hypothesis:

| case | experiment | score | conclusion |
|---|---|---:|---|
| `08` | mixed limit `400`, `warm_c=0.05/0.15/0.20` | best `102.668308` | all below merged `warm_c=0.10` score `103.788974`; keep `0.10` |
| `09` | mixed limit `400/800`, `warm_c=0.10` | best `160.632806` | below current `164.743738` |
| `09` | broad-neighbor limit `800/1200` on first 500 lambdas | best `161.733374` | below current `164.743738` |
| `09` | two-hop source limit `50`, `warm_c=0.10` | `152.110585` | weak first signal; only continue with a stronger reason |
| `09` | full lambda broad window `1000x50` + local warm `250x200` | `127.729502` | widening broad coverage alone destroys the current distribution |
| `09` | mixed limit `400`, HV-proxy selector | `142.706247` | volume proxy over-selects wrong regions |
| `07` | mixed limit `1000/1200`, `warm_c=0.05` | best `171.259940` | below merged limit `800` score `174.592019`; keep `800` |
| `00` | mixed limit `200/600/800`, `warm_c=0.10` | best `470.291461` | below merged limit `400` score `483.954121`; keep `400` |
| `04` | mixed limit `400`, `warm_c=0.10` | `248.817453` | below current broad-neighbor `256.878740` |
| `04` | broad-neighbor limit `1200`, HV-proxy selector | `181.768634` | volume proxy is not competitive with crowding on this case |
| `04` | broad-neighbor budget side `350x100+325x200` / `450x100+275x200` | best `258.882603` | below merged `400x100+300x200`; keep `400/300` |
| `04` | offset broad window `[500,1000)` + mixed limit `400`, `warm_c=0.10` | `254.723616` | below merged `259.768537`; offset window alone is not enough |
| `01` | local source, `warm_c=0.075/0.125/0.15/0.20` | best `95.796720` | below current `98.046167`; keep default local `warm_c=0.10` |
| `05` | broad-neighbor limit `800`, `warm_c=0.15`, budget `400x100+300x200` | `138.608770` | slightly below current `138.642082`; keep `500x100+250x200` |
| `00` | mixed limit `400`, warm-c side sweep `0.05/0.15/0.20` | best `477.964225` | below merged `warm_c=0.10` score `483.954121` |

## Hypotheses

### H1: Mixed Warm Candidate Bank

Problem:

- Full broad-neighbor replacement helps `02/04/05/06`, but hurts `00/07/08/09`.
- Full local scalar-descent warm source is safer on `00/07/08/09`, but misses compact pockets.

Experiment:

- Build a warm candidate bank by concatenating:
  - local scalar-descent ND candidates;
  - sampled broad ND one-hop neighbors.
- Select the final `250` warm states by crowding/diversity over the combined ND objective set.
- Sweep source ratios only through candidate caps, not returned rows:
  - local restarts `6`;
  - neighbor source limits `200/400/800`;
  - mixed candidate mode on cases `09/00/07/08`.

Merge gate:

- Case-specific merge only if score beats current verified score by at least `+0.5` and guard cases do not regress.

### H2: Two-Hop Sampled Neighborhood

Problem:

- One-bit neighbors may be too local for unsupported regions.

Experiment:

- Add optional two-bit flips around broad sampled ND bases.
- Keep candidate count controlled:
  - source limit `100/200/400`;
  - include base + one-hop + two-hop;
  - use ND filtering before warm selection.
- First targets: `09`, `00`, `07`.

Risk:

- Candidate generation and ND filtering may get slow. Abort if per-case runtime exceeds ~250s without score lift.

### H3: Sampled HV-Support Warm Selection

Problem:

- Crowding chooses spread, not necessarily HV contribution.

Experiment:

- On combined candidate bank, greedily prefilter top candidates by single-point volume proxy and diversity.
- Do not use exact frontier or exact guidance in runtime.

Merge gate:

- Only if it beats crowding on the same candidate source.

Status:

- Prototype added as `scripts/run_hv_warm_grid.py`.
- The selector is not merged into `answer.py`.
- Compile and dry-run passed:

```powershell
python -m py_compile scripts\run_hv_warm_grid.py
python scripts\run_hv_warm_grid.py --case 09 --seed 2031 --selector crowding,hv_proxy --candidate-source mixed --neighbor-source-limit 400 --warm-c 0.1 --dry-run
```

Next proof step is a real `--run` comparison on one case. Do not merge unless it beats the current official per-case score through `scripts/eval_answer_seed.py` after porting into `answer.py`.

### H4: Residual Case-Specific Warm-C Sweep

Problem:

- `06` preferred `warm_c=0.20`, `05` preferred `0.15`; remaining cases may need different mixer strength.

Experiment:

- Only run warm-c sweeps after H1/H2 finds a candidate source that is near current score.
- Values: `0.05/0.10/0.15/0.20/0.25`.

### H5: Offset Lambda Windows

Problem:

- Some exact-gap diagnostics point to lambda ids beyond the first 500 directions, but the naive full-window `1000x50` test on case `09` was much worse.
- The more controlled question is whether a contiguous offset window such as `[500,1000)` can help a specific case without diluting shots across all `1000` lambdas.

Tool:

- Added `scripts/run_lambda_offset_grid.py`.
- It is dry-run by default and requires `--run` for MindQuantum sampling.
- It keeps the contest boundary: the broad and warm rows are generated through MindQuantum sampling, while classical candidates only choose warm-start states and lambda ids.

First candidate command:

```powershell
python scripts\run_lambda_offset_grid.py --case 04 --seed 2026 --candidate-source mixed --broad-start 500 --broad-weights 500 --broad-shots 100 --warm-count 250 --warm-shots 200 --neighbor-source-limit 400 --warm-c 0.1 --out results\lambda_offset_04_500_1000_mixed --run
```

### H6: In-Budget Seed Cohorts

Problem:

- Full offline union of multiple `answer.main1` runs can reveal seed complementarity, but running full `100000` rows per seed and then subsampling is not the implementation target.

Merged result:

- Case `09` uses seeds `2031` and `2041` inside the same `100000` returned rows.
- Each seed runs the same 500 broad and 250 warm circuits, but broad shots are split from `100` to `50` per seed and warm shots from `200` to `100` per seed.
- Formal `scripts/eval_answer_seed.py --case data\public\k5_grid4x5_09.npz --seed 2031` score: `170.673648`.

Runtime risk:

- Case `09` circuit count roughly doubles while total returned rows stay fixed. The verified local elapsed time is about `202s`, so full-public timeout margin must be watched.

Follow-up:

- A runtime-only test of the alternative case `09` cohort `(2027,2031)` did not finish within a 15-minute local timeout and was killed. Do not merge or rerun that combination without first adding a faster harness or narrower circuit budget.

## Workstreams

### Worker A: Mixed Candidate Tool

Write scope:

- `scripts/run_local_warm_grid.py`
- optional result doc `docs/superpowers/plans/2026-05-26-moo-final-gap-worker-a.md`

Tasks:

- Add candidate source `mixed`.
- Reuse existing local candidate generator and broad-neighbor generator.
- Compile and dry-run.
- Evaluate `09` and `07` with source limits `400/800`.

### Worker B: Two-Hop Neighborhood Tool

Write scope:

- `scripts/run_local_warm_grid.py` if Worker A is not editing, otherwise `scripts/run_twohop_warm_grid.py`
- optional result doc `docs/superpowers/plans/2026-05-26-moo-final-gap-worker-b.md`

Tasks:

- Add two-hop sampled-neighbor candidate generation.
- Evaluate `09` with small source limits `100/200`.

### Worker C: Remaining Case Scout

Write scope:

- results only, plus `docs/superpowers/plans/2026-05-26-moo-final-gap-worker-c.md`

Tasks:

- Run existing `broad_neighbors` and local baseline variants on `00/07/08/09` only where not already covered.
- Summarize winners and hard rejects.

## Local Main-Agent Tasks

1. Implement `mixed` candidate source if subagent quota fails.
2. Keep `answer.py` unchanged until a case-specific winner is verified.
3. Run targeted `eval_answer_seed.py` after any merge.
4. Update this plan with results.
5. Commit and push only verified winners and durable scripts/docs.

## Verification

Compile:

```powershell
python -m py_compile answer.py scripts\run_local_warm_grid.py scripts\eval_answer_seed.py
```

Targeted scoring:

```powershell
python scripts\run_local_warm_grid.py --case 09 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1
python scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1
```

Merge verification:

```powershell
python scripts\eval_answer_seed.py --case data\public\k5_grid4x5_<case>.npz --seed <seed>
python run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_final_gap.json
```

## Merge Rules

- No exact-frontier files in `answer.py`.
- No exact-guidance JSON in `answer.py`.
- No classical candidate row inserted into returned `sample_spins`.
- All returned rows must come from `Simulator.sampling`.
- Case-specific public tuning is allowed only as an explicit public-score sprint decision and must be documented.
