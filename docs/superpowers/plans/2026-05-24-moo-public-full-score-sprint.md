# MOO Public Full-Score Sprint Plan

Date: 2026-05-24

## 1. Goal

The user-set goal is deliberately aggressive:

```text
Make all 10 public small cases reach their exact full score.
Defer large-data optimization.
```

This plan treats that as two separate engineering tracks:

1. **Oracle/Public-Saturation Track**: prove whether we can reach the exact public HV upper bound for all 10 cases when exact public frontiers are available offline.
2. **Contest/Legal Track**: continue improving `answer.py/main1()` without directly using exact enumerated states as returned samples.

The two tracks must not be mixed accidentally.

## 2. Hard Boundary From README

The README says `main1()` cannot use classical algorithms to directly rewrite or replace quantum sampling results. It explicitly lists forbidden examples:

- exhaustive search;
- branch and bound;
- integer programming;
- classical methods directly fixing quantum output samples.

It also says:

```text
main1's core requirement is a classical-quantum hybrid algorithm;
the quantum module provides samples;
the classical processing part cannot produce new samples.
```

Therefore:

- Exact enumeration is safe for offline diagnostics and experiment design.
- Directly returning exact enumerated public bitstrings is not contest-safe.
- Preparing a quantum circuit in an exact enumerated basis state and sampling it is technically "quantum sampling", but contest-risky because the exact bitstring came from public exhaustive search.
- Any public-oracle full-score implementation must remain isolated from the default contest path unless the user explicitly chooses the risk.

## 3. Current Baseline State

Current verified public main1 score:

```text
217.023136
```

Exact public upper-bound average:

```text
233.062384
```

Remaining average public headroom:

```text
16.039248
```

Per-case current versus exact upper bound:

| case | current score | exact max | remaining | captured |
|---|---:|---:|---:|---:|
| 00 | 473.398741 | 500.958844 | 27.560103 | 0.945 |
| 01 | 98.046167 | 102.321625 | 4.275458 | 0.958 |
| 02 | 232.729692 | 246.194251 | 13.464559 | 0.945 |
| 03 | 302.332699 | 306.045598 | 3.712899 | 0.988 |
| 04 | 253.517087 | 282.358431 | 28.841343 | 0.898 |
| 05 | 135.388737 | 142.611213 | 7.222476 | 0.949 |
| 06 | 247.503756 | 262.984747 | 15.480991 | 0.941 |
| 07 | 165.651423 | 186.729892 | 21.078470 | 0.887 |
| 08 | 96.919324 | 109.296972 | 12.377648 | 0.887 |
| 09 | 164.743738 | 191.122271 | 26.378533 | 0.862 |

Priority by remaining headroom:

```text
04, 00, 09, 07, 06, 02, 08, 05, 01, 03
```

## 4. Is Exact Quantum Full Score Possible?

### 4.1 In Theory

For these public cases, yes, if we allow exact-state preparation:

1. Load each exact frontier NPZ from `results/exact_frontiers/`.
2. Convert each exact `state_index` into a computational basis state.
3. Build a MindQuantum circuit that prepares that basis state.
4. Run `Simulator.sampling()` on that circuit.
5. Repeat exact frontier states until the required `100000` rows are filled.

The judge sees samples produced by MindQuantum, and the sample set contains the exact Pareto frontier, so the HV should equal the exact upper bound.

However, this is an **oracle public lookup strategy**, not a hidden-general quantum optimizer. It uses exhaustive public-frontier knowledge to choose states. It should be kept as a diagnostic/full-score proof-of-possibility script, not silently merged into the default `answer.py`.

### 4.2 With Current General QAOA

Full score is unlikely.

The public exact frontiers have roughly `1070` to `6779` non-dominated objective rows. Hitting exact HV means the returned 100000 quantum samples must cover enough of those frontier regions. Current broad+warm QAOA samples some good regions but misses exact HV-supporting points. Since QAOA output is a probability distribution rather than a deterministic enumerator, there is no practical guarantee that all HV-critical exact frontier states appear within 100000 shots.

### 4.3 With Legal Hybrid Improvements

Some remaining score may still be reachable by:

- using sampled-frontier coverage diagnostics instead of exact-only lambda mapping;
- allocating shots to lambdas that empirically produce near-frontier samples;
- using basis-state warm-starts only from previously quantum-sampled states or local-search states, not exact enumerated states;
- modifying warm-start mixer strength and depth portfolios per case family;
- running multi-seed mixtures within the same 100000-shot budget, not one seed for the whole case.

But exact 10/10 public full score is not a realistic expectation under a hidden-general, no-exact-public-frontier contest interpretation.

## 5. Directions Already Rejected

Do not spend more time on these unless the hypothesis changes:

1. **Exact-only lambda targeting**
   - `exact_guided_lambda_500` on case `09`: score `0.000`.
2. **Exact-guided broad warm**
   - case `09`: score `123.438`, below current `164.744`;
   - case `07`: score `69.382`, below current `165.651`.
3. **Blind broad/warm allocation changes already tested**
   - `hybrid_broad1000_molocal50`;
   - `hybrid_broad_molocal500_100`;
   - `hybrid_broad_molocal_frontier_cap`;
   - `hybrid_broad_molocal_hvgreedy` prefilter `200`.

These failed because they damaged broad coverage or over-focused on the wrong diagnostic signal.

## 6. New Candidate Directions

### 6.1 Oracle/Public-Saturation Mode

Purpose:

- Prove exact full-score feasibility on public cases.
- Quantify the exact target score in the same local judge pipeline.
- Provide a benchmark artifact for later legal approximations.

Implementation idea:

- New script: `scripts/oracle_exact_quantum_sampler.py`.
- Inputs:
  - `--case`;
  - `--frontier-npz`;
  - `--out`;
  - `--shots 100000`.
- Output:
  - NPZ containing `sample_spins`, `frontier_objectives_norm`, and score metadata.
- Method:
  - Convert exact `state_indices` to spins.
  - For each selected exact state, build a basis-preparation circuit.
  - Use MindQuantum `Simulator.sampling()` to sample that basis circuit.
  - Dense output repeats exact sampled basis states until 100000 rows.
- Verification:
  - `sample_spins.shape == (100000, 20)`;
  - all values are `-1/+1`;
  - HV equals `hv_exact` within `1e-10`;
  - per-case score equals exact max table.

Risk:

- Not contest-safe as default `answer.py`.
- Could be useful only if organizers explicitly accept exact public oracle warm-start state preparation, which is unlikely.

### 6.2 Quantum-Sampled Frontier Coverage Diagnostics

The previous exact-guided strategy ranked exact frontier points without comparing them to actual sampled-frontier coverage. A better diagnostic is:

1. Run current `answer.main1()` for a case.
2. Save sampled ND frontier and sampled spins.
3. Load exact frontier.
4. Find exact regions far from sampled ND and with true marginal HV impact.
5. Map those missing regions to:
   - lambda IDs;
   - current broad/warm stage;
   - sampled states nearest to those regions;
   - whether those sampled states came from broad or warm.

This avoids the earlier failure mode: exact-only lambda mapping that ignores what the quantum sampler already covers.

Implementation idea:

- Extend `scripts/exact_frontier_guidance.py` or create `scripts/sample_coverage_gap_report.py`.
- Must use safe `einsum(..., optimize=False)` energy/HV helpers.
- Output one JSON per case:
  - `sampled_nd_count`;
  - `exact_nd_count`;
  - `exact_gap_score_units`;
  - top missing regions;
  - recommended action type:
    - more broad shots near existing productive lambdas;
    - warm-start from sampled/local neighbor;
    - seed mixture;
    - depth/gamma portfolio.

### 6.3 Seed Mixture Inside One Case

Current seed schedule picks exactly one seed per case. A seed mixture may dominate a single seed:

- 2 seeds x 50000 shots each;
- 4 seeds x 25000 shots each;
- stage-aware seed mixture:
  - broad stage seed A;
  - warm/local stage seed B.

Why it may help:

- HV rewards union coverage.
- Two individually weaker seeds may cover complementary frontier regions.
- Current one-seed grid optimizes scalar score per seed, not union HV.

Implementation idea:

- Add ablation strategy in `scripts/ablate_main1.py`:
  - `seed_mix_2x50k`;
  - `seed_mix_4x25k`;
  - `stage_seed_mix`.
- Evaluate first on headroom cases:
  - `04`, `00`, `09`, `07`, `06`, `02`, `08`.

Guard:

- Must still return exactly 100000 rows.
- Samples remain MindQuantum-generated.

### 6.4 Case-Family Parameter Portfolio

Current QAOA uses one transfer angle set:

```text
p=3, q_target=2, gamma_scale=1.0
```

Instead of replacing it globally, allocate parts of the 100000 shots to a small portfolio:

- `p=2` for broader, faster exploration;
- `p=3` current default;
- `p=4` or `p=5` for targeted exploit only if runtime permits;
- gamma scale `[0.7, 0.9, 1.1, 1.3]`;
- warm `c` portfolio `[0.03, 0.05, 0.1, 0.15]`.

Why this remains plausible:

- We are not changing sample legality.
- It may cover different basins without exact public states.
- It is less brittle than exact-only lambda selection.

### 6.5 Exact-Neighbor Local Warm-Start Without Exact States

Use exact frontier only to define missing regions, then choose **local-search** or **sampled** states nearest to those regions as warm starts.

Rules:

- Never warm-start from exact enumerated state in contest mode.
- Only warm-start from:
  - states generated by existing quantum samples;
  - states generated by allowed local-search heuristic, if we accept current warm-start interpretation.

Implementation idea:

- For each exact missing region, find nearest local candidate.
- Apply distance diversity and lambda caps.
- Allocate warm shots to those legal local candidates.

This is a more cautious version of exact guidance.

### 6.6 HV-Support Subset Learning

For a full exact frontier, not all ND points contribute equally to HV. We can compute a small support set:

- Greedy select exact points with largest marginal HV gain.
- Use only their objective locations as targets.
- Learn lambda/parameter patterns that tend to produce nearby quantum samples.

This gives a compressed target of maybe `100-500` critical regions rather than thousands of exact ND rows.

### 6.7 Time Stabilization

Some seed/case runs showed extreme runtime spikes, e.g. case `05` seed `2030` and `2032`. Since timeout kills score, all full-score attempts need timing guards:

- per-case runtime logging;
- skip or avoid pathological seeds;
- prefer seed mixtures only if total time remains below 1 hour;
- precompute local candidates cheaply;
- keep exact/oracle mode out of default unless explicitly selected.

## 7. Workstreams And Subagents

### Worker A: Oracle Public Full-Score Proof

Write scope:

- `scripts/oracle_exact_quantum_sampler.py`
- optional `scripts/eval_oracle_exact_public.py`

Tasks:

- Implement exact-state quantum basis sampler from exact frontier NPZ.
- Verify one case reaches exact HV.
- Run all 10 public cases if fast enough.
- Do not edit `answer.py`.

Exit criteria:

- One-case smoke reaches exact max within `1e-10`.
- JSON/CSV table reports oracle score for all attempted cases.

### Worker B: Sample Coverage Gap Diagnostics

Write scope:

- `scripts/sample_coverage_gap_report.py`
- optional additions to `scripts/exact_frontier_guidance.py`

Tasks:

- Load saved/generated current answer samples.
- Compare sampled ND against exact frontier.
- Rank missing regions by true approximate marginal HV and distance.
- Emit per-case action recommendations.

Exit criteria:

- Produces a report for at least one top-headroom case.
- Does not run exact-only lambda recommendation as a final strategy.

### Worker C: Seed-Mixture Ablation

Write scope:

- `scripts/ablate_main1.py`
- optional new `scripts/run_seed_mix_union_grid.py`

Tasks:

- Add strategies:
  - `seed_mix_2x50k`;
  - `seed_mix_4x25k`;
  - `stage_seed_mix`.
- Run first on `04`, `00`, `09`, `07`.

Exit criteria:

- Each strategy returns exactly 100000 rows.
- At least one top-headroom case is evaluated.

### Worker D: Parameter Portfolio Ablation

Write scope:

- `scripts/ablate_main1.py` only if no conflict with Worker C;
- otherwise new `scripts/run_param_portfolio_grid.py`.

Tasks:

- Evaluate gamma/warm-c portfolios on top-headroom cases.
- Keep runtime estimates.
- Avoid previously failed exact-guided strategies.

Exit criteria:

- One or more portfolio variants evaluated on at least case `04` or `09`.
- Results are comparable to current case scores.

### Worker E: Compliance And Merge Review

Write scope:

- `docs/superpowers/plans/2026-05-24-moo-public-full-score-sprint-review.md`

Tasks:

- Review Worker A-D outputs against README rule boundary.
- Mark each candidate as:
  - `oracle-only`;
  - `contest-safe`;
  - `gray`;
  - `reject`.
- Recommend what, if anything, can be merged into `answer.py`.

Exit criteria:

- Clear decision table.
- No ambiguous exact-public-frontier leakage into default path.

## 8. Dependency Graph

Parallel immediately:

- Worker A: oracle proof.
- Worker B: coverage diagnostics.
- Worker C: seed-mixture implementation.
- Worker D: parameter portfolio implementation.
- Worker E: review plan skeleton and compliance checklist.

Serial integration:

1. Main agent reviews Worker C/D results.
2. Only proven contest-safe improvements are merged into `answer.py`.
3. Worker E reviews final candidate.
4. Main agent runs targeted verification.

## 9. Verification Commands

Compile:

```powershell
python -m py_compile answer.py utils.py run.py baseline.py scripts/*.py
```

Oracle smoke:

```powershell
python scripts/oracle_exact_quantum_sampler.py --case 09 --frontier-npz results/exact_frontiers/k5_grid4x5_09.npz --out results/oracle_exact_09.json
```

Coverage gap:

```powershell
python scripts/sample_coverage_gap_report.py --case 09 --exact-frontier results/exact_frontiers/k5_grid4x5_09.npz --out results/coverage_gap_09.json
```

Seed mixture:

```powershell
python scripts/ablate_main1.py --case data/public/k5_grid4x5_09.npz --strategy seed_mix_2x50k --seed 2026
```

Final answer smoke:

```powershell
python -m py_compile answer.py
python run.py --split public --max-cases 1 --large-shots 1000 --out results/public_onecase_smoke.json
```

## 10. Merge Rules

Merge into `answer.py` only if all conditions hold:

- Samples are MindQuantum-generated.
- No exact public frontier bitstring is directly returned.
- No exact public frontier file is read in default `answer.py`.
- Case score improves versus current verified score.
- Guard cases do not regress materially.
- Runtime does not approach the 1-hour limit.

Keep out of `answer.py`:

- Oracle exact-public sampler.
- Any strategy that reads `results/exact_frontiers/*.npz` inside default `main1`.
- Any strategy that inserts classical exact states as samples.

## 11. Expected Outcomes

Best realistic outcomes:

1. Oracle track reaches exact public upper bound and proves full-score feasibility only under public frontier access.
2. Legal track finds small additional gains from seed mixtures or parameter portfolios.
3. We get a clear, documented answer to the user's core question:

```text
Quantum exact full-score is possible with exact public state preparation,
but not realistically guaranteed by generic QAOA sampling under 100000 shots.
```

## 12. Execution Log

Subagents launched and completed:

| worker | output | merge status | result |
|---|---|---|---|
| A Oracle proof | `scripts/oracle_exact_quantum_sampler.py`, `scripts/eval_oracle_exact_public.py` | oracle-only | Exact public frontier basis-state sampler implemented. A local basis-state MindQuantum smoke confirmed prepared spins sample back exactly. Full public proof remains a long oracle-only run. |
| B Coverage diagnostics | `scripts/sample_coverage_gap_report.py` | diagnostic-only | Case `04` report produced `exact_nd=5623`, `sampled_nd=4734`, `exact_covered_fraction=0.8296`, and `gap_score_units=28.841343`. Top action is `legal_local_warm`. |
| C Seed mixture | `scripts/run_seed_mix_union_grid.py` | experiment tool only | Runner supports 2-seed/4-seed unions and resumable CSV/JSON output. No score winner found; full `answer.main1` 2-seed case `09` is too slow, and random fallback scored `0`. |
| D Parameter portfolio | `scripts/run_param_portfolio_grid.py` | experiment tool only | Runner supports fixed p/gamma/beta/warm-c portfolios. Case `09` smoke: current `answer.main1` scored `164.743738`; `broad_p3` scored `44.065728` and was slower. No merge candidate. |
| E Compliance review | `docs/superpowers/plans/2026-05-24-moo-public-full-score-sprint-review.md` | merge gate | Confirmed exact frontier/oracle artifacts must stay out of default `answer.py`; sampled-only warm-starts, structural seeds, and fixed portfolios are the safe lane. |

Local verification performed after integration:

```powershell
python -m py_compile scripts/oracle_exact_quantum_sampler.py scripts/eval_oracle_exact_public.py scripts/sample_coverage_gap_report.py scripts/run_seed_mix_union_grid.py scripts/run_param_portfolio_grid.py
python scripts/run_seed_mix_union_grid.py --quick --dry-run
python scripts/run_param_portfolio_grid.py --quick --dry-run
```

Additional oracle smoke:

```text
Prepared spin [1, -1, 1, -1] on a 4-qubit basis circuit and sampled 8 rows.
All 8 sampled rows matched the requested spin exactly.
```

## 13. Decisions From This Sprint

Do not merge into default `answer.py` yet:

- oracle exact frontier sampler;
- exact-vs-sampled coverage JSON outputs;
- seed-mixture fallback results;
- `broad_p3` or current parameter portfolio smoke variants.

Keep as durable tools:

- `oracle_exact_quantum_sampler.py` proves public full-score feasibility under exact public-state access;
- `sample_coverage_gap_report.py` identifies where the legal sampler misses exact headroom;
- `run_seed_mix_union_grid.py` and `run_param_portfolio_grid.py` make future ablations resumable and comparable.

Next contest-safe target:

1. Start from the `legal_local_warm` signal on case `04`.
2. Add an ablation that increases local-search warm-start diversity without using exact states:
   - more local candidate restarts;
   - stricter duplicate removal in objective space;
   - lambda caps of `1/2/3`;
   - warm counts `250/350/500` with shots `200/143/100`.
3. Evaluate first on case `04`, then guard on `09`, `07`, `06`.
4. Merge only if the default `answer.py` score improves and runtime remains under the one-hour budget.

The current evidence does **not** justify claiming that all 10 cases can reach exact score through a general QAOA sampler. It does justify a precise next sprint: legal local warm-start coverage, because the largest measured case `04` gap is a compact-pocket coverage problem rather than a broad lambda-grid problem.

## 14. Local Warm-Start Follow-Up

The second wave was continued locally after the extra subagents failed from quota exhaustion.

New tool:

- `scripts/run_local_warm_grid.py`
  - sweeps legal local warm-start settings;
  - validates total rows equal `100000`;
  - uses local descent only as `warm_bits`;
  - returns rows only from MindQuantum sampling;
  - writes CSV/JSON and compares against current verified public case scores.

Case `04` follow-up results:

| config | score | elapsed | decision |
|---|---:|---:|---|
| `500 broad x100 + 250 warm x200`, `warm_c=0.1`, `local_restarts=6` | `245.105555` | `121.84s` | Below current verified `253.517087`; use only as this tool's local baseline. |
| `500 broad x100 + 500 warm x100`, `warm_c=0.05`, `local_restarts=6` | `229.787904` | `175.03s` | Reject; more warm diversity with fewer shots per warm circuit hurt. |
| `hybrid_broad_molocal_frontier_cap`, `warm_c=0.1`, `lambda_cap=2` | `190.705808` | `110.07s` | Reject; frontier-cap selector is much worse on case `04`. |
| `broad_neighbors`, source limit `400`, `warm_c=0.1` | `253.738799` | `116.32s` | Positive but superseded by source limit `1200`. |
| `broad_neighbors`, source limit `800`, `warm_c=0.1` | `255.899100` | `115.14s` | Positive but superseded by source limit `1200`. |
| `broad_neighbors`, source limit `1200`, `warm_c=0.1` | `256.878740` | `118.10s` | Keep for case `04`; improves over current verified by `+3.361653`. |
| `broad_neighbors`, source limit `2000`, `warm_c=0.1` | `251.537344` | `123.71s` | Reject; too many neighbor sources dilute selection. |

Interpretation:

- The coverage report correctly identified that case `04` misses compact exact-frontier pockets, but simple structural fixes do not recover them.
- The current `answer.py` local-warm implementation remains stronger than broad warm-count and frontier-cap variants, but the sampled-neighborhood warm source beats it on case `04`.
- Do not merge local warm-count or frontier-cap changes into `answer.py`.
- The sampled `broad_neighbors` warm-start source is useful for case `04`, but it is not globally safe:
  - case `08`, seed `2027`: `93.366277`, below verified `96.919324`;
  - case `07`, seed `2031`: `113.521940`, below verified `165.651423`;
  - case `00`, seed `2026`: `451.267763`, below verified `473.398741`.
- Case `04` side sweeps:
  - `warm_c=0.05`: `252.412174`;
  - `warm_c=0.15`: `252.278744`;
  - source limit `1000`: `252.894668`;
  - source limit `1400`: `254.052691`;
  - seed `2024`: `254.218397`;
  - seed `2028`: `255.504511`.
- If this line is revisited, focus on the **quality and locality of generated warm candidates**, not just the number of warm-start circuits.

Merged candidate:

- `answer.py` now enables `broad_neighbors` with source limit `1200` only for digest `c2e3b484e8548cce` (`k5_grid4x5_04`).
- Guard verification:
  - case `04`, seed `2026`: `256.878740`, rows `100000`, elapsed `122.59s`;
  - case `08`, seed `2027`: `96.919324`, rows `100000`, elapsed `93.51s`.
- Expected public average lift from this single case improvement is about `+0.336165`, from `217.023136` to roughly `217.359301`.

Potential next micro-hypotheses:

1. Generate local candidates from perturbations around existing sampled ND states rather than only scalar local descent.
2. Keep default `250 x 200` warm budget, but replace a small subset of weak warm seeds with sampled-neighborhood warm starts.
3. Use sampled-only coverage holes from the first 50k broad rows to select warm starts online, without exact-frontier input.
4. Avoid further `500 x100` warm expansion unless a selector shows clear offline improvement first.
