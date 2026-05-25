# MOO Public Full-Score Sprint Compliance Review

Date: 2026-05-24

Role: Worker E, compliance and merge review.

Write scope for this review: this document only. No `answer.py` or worker script changes.

## 1. Sources Reviewed

- `README.md`, especially sections `4`, `8`, `11`, `14`, and `15`.
- `answer.py` current `main1()` / `main2()` structure.
- `docs/superpowers/plans/2026-05-24-moo-public-full-score-sprint.md`.
- Related context from `docs/superpowers/plans/2026-05-24-moo-exact-headroom-guided-tuning.md`.
- Script surfaces for current diagnostic/ablation work:
  - `scripts/exact_public_headroom.py`
  - `scripts/exact_frontier_guidance.py`
  - `scripts/eval_seed_mix.py`
  - `scripts/run_seed_mix_union_grid.py`
  - relevant strategy names in `scripts/ablate_main1.py`

## 2. Hard Rule Boundary

The README rule that matters most is:

- `main1()` must return exactly `100000` samples.
- Returned `sample_spins` must be produced by the MindQuantum sampling path.
- Classical logic may tune circuit structure, warm-start strategy, rounds, shot allocation, and post-sampling evaluation.
- Classical logic must not directly rewrite, repair, append, or replace quantum samples.
- Explicitly forbidden examples include exhaustive search, branch and bound, integer programming, and classical methods directly fixing quantum output samples.

For this sprint, I recommend treating public exact artifacts as a separate oracle track:

- Offline exact enumeration is acceptable for diagnosis and proof of public headroom.
- Default contest `answer.py/main1()` must not read exact public frontier artifacts, exact public headroom files, oracle sampler outputs, or exact guidance JSON.
- Even if an exact basis state is sampled through MindQuantum, selecting that basis state from a public exhaustive frontier is oracle-public lookup, not a hidden-general optimizer.

## 3. Current `answer.py` Compliance Snapshot

Current `main1()` returns samples from `Simulator.sampling()` in both phases:

- broad QAOA coverage: `500` lambdas x `100` shots;
- multi-objective local warm-start QAOA: `250` warm circuits x `200` shots.

The local-search candidates are not copied into `sample_spins`; they are converted to `warm_bits01` and used only as circuit initial states. That is mergeable under the permissive warm-start reading of the README, as long as no exact-public frontier states are used as those warm starts.

The current `_MAIN1_SEED_BY_DIGEST` table is more sensitive. It does not replace samples and still uses MindQuantum, but it fingerprints known public cases and chooses seeds from public evaluation. I classify that as `gray`, not cleanly contest-safe. It should not be expanded unless the user explicitly accepts the public-overfit risk.

One extra gray note: `answer.py` reaches `objective_extrema(problem)` through local candidate scoring. The judge itself uses exact extrema for normalization, so this is less severe than exact-frontier lookup. Still, if a strict reviewer treats any exhaustive small-case computation inside `main1()` as suspect, this path should be replaced or justified as normalization-only, not sample generation.

## 4. Decision Table

| Direction | Review Status | Default `answer.py`? | Rationale |
|---|---|---:|---|
| `results/exact_frontiers/*.npz` | `oracle-only` | No | Contains exact public Pareto states/objectives from exhaustive enumeration. Safe for diagnosis; not safe as runtime input or embedded constants. |
| `scripts/exact_public_headroom.py` and `results/exact_public_headroom*` | `oracle-only` | No | Proves public exact HV and headroom. Must not steer default samples directly. |
| Oracle exact quantum sampler | `oracle-only`; reject for default | No | Sampling exact enumerated basis states through MindQuantum is still public oracle lookup. Useful only for demo if user explicitly chooses that track. |
| Exact-guided lambda strategies | `oracle-only` or `gray` | No by default | Final samples are quantum, but lambda IDs come from exact public frontier guidance. Only a generalized rule learned from these diagnostics may be considered. |
| `scripts/exact_frontier_guidance.py` outputs | `oracle-only diagnostic` | No | JSON includes `state_index`, exact objectives, and recommended lambdas derived from exact frontier. `answer.py` must not load it. |
| Runtime sampled coverage logic | `contest-safe` | Yes, with checks | If it uses only samples generated in the current run to choose later quantum circuits, it matches the allowed hybrid pattern. |
| Exact-vs-sampled coverage diagnostics | `oracle-only diagnostic` | No | Good for understanding gaps; exact-derived files must not become default runtime inputs. |
| Fixed/global seed schedule | `contest-safe` | Yes | A seed rule independent of public case identity is just sampler configuration. |
| Public digest / per-public-case seed tuning | `gray` | Only with explicit opt-in | Still quantum sampled, but it is public-case fingerprinting and weak hidden-set story. Prefer structural multi-seed alternatives. |
| Warm-start selector from prior quantum samples | `contest-safe` | Yes | Classical selection chooses future quantum initial states; it does not fabricate returned rows. |
| Warm-start selector from local heuristic states | `contest-safe` to `gray` | Yes, with documentation | Acceptable if heuristic states are only initial states and never returned directly. Avoid exact search, B&B, IP, or exact-frontier candidates. |
| Warm-start from exact frontier states | `oracle-only` / reject for default | No | Public exhaustive states directly choose circuit starts; high leakage risk even if samples come from a circuit. |
| Fixed parameter portfolio | `contest-safe` | Yes | Depth/gamma/beta/warm-c portfolios are circuit choices if not keyed to public exact artifacts. |
| Per-public-case parameter portfolio | `gray` | Only with explicit opt-in | Similar overfit risk to per-case seeds. Needs a generalization story. |
| Classical direct repair/appending/exact sample insertion | `reject` | No | Violates README: classical processing cannot produce new samples or fix returned samples. |
| Branch-and-bound / IP / exhaustive-frontier replacement | `reject` | No | Explicitly forbidden for `main1()` default path. |

## 5. Required Special Reviews

### Exact Frontier

Allowed:

- compute exact frontiers offline;
- use them to quantify public headroom;
- use them in reports and oracle-only demonstrations.

Not allowed in default `answer.py`:

- reading `results/exact_frontiers/*.npz`;
- embedding exact `state_indices`, `spins`, `bits01`, exact objectives, or exact-derived public lambdas;
- preparing exact public frontier basis states as the normal solution path.

### Exact Public Headroom

`exact_public_headroom` is diagnostic. It should remain in `scripts/` and `results/`, not in `answer.py`.

Acceptable use:

- prioritize experiments;
- explain why full public score is or is not realistic;
- compare legal strategies against the exact upper bound.

Merge blocker:

- any default code path that loads `results/exact_public_headroom.json`, `.csv`, or per-case NPZ frontier artifacts.

### Oracle Sampler

An oracle sampler can demonstrate that exact public HV is reachable if exact public states are known. It should be labeled `oracle-only` and kept out of the contest submission.

Default merge decision: reject.

Reason: the quantum sampler is being used as a transport for exact public enumeration results.

### Seed Schedule

Clean path:

- fixed seed;
- deterministic seed derived from generic problem data without a public lookup table;
- structural multi-seed split applied to every case.

Gray path:

- `_MAIN1_SEED_BY_DIGEST` and any table mapping known public problem fingerprints to winning seeds.

Recommendation:

- do not expand the digest table in this sprint unless the user explicitly confirms that public-overfit risk is acceptable;
- prefer a structural 2-seed or 4-seed mixture if it is within a small score margin of digest tuning.

### Seed Per-Public-Case Tuning

Status: `gray`.

It does not directly violate the "quantum samples only" rule, but it is tuned on public case identity. It should be merged only after explicit user approval and fresh evidence that:

- it improves the target public score;
- it does not regress guard cases materially;
- hidden performance is not the goal, or the risk has been accepted.

### Warm-Start Selector

Contest-safe merge shape:

- select warm starts from previous MindQuantum samples;
- or select local heuristic candidates, then use them only as `warm_bits01`;
- keep returned rows exclusively from `Simulator.sampling()`.

Reject/default-blocking shape:

- warm starts are exact public frontier states;
- warm starts come from branch-and-bound, IP, exhaustive exact frontier search, or public oracle artifacts;
- warm-start candidate rows are copied into `out_spins`.

### Parameter Portfolio

Contest-safe merge shape:

- fixed portfolio shared across all cases;
- structural portfolio based on `n`, `k`, budget, or stage;
- no dependency on `results/` artifacts.

Gray shape:

- per-public-case portfolio table selected from public scoring.

Reject/default-blocking shape:

- portfolio reads exact guidance JSON or exact frontier files at runtime.

### Sampled Coverage Diagnostics

Contest-safe runtime version:

- compute objectives/frontier of samples generated earlier in the same `main1()` call;
- use that sampled frontier to choose later lambdas/warm starts.

Oracle-only diagnostic version:

- compare sampled frontier to `results/exact_frontiers/*.npz`;
- output missing exact anchors or exact-derived lambda IDs.

Default `answer.py` must not load the oracle diagnostic output.

## 6. Merge Gates For `answer.py`

Before any change enters default `answer.py`, all gates below should pass.

### Source And Data Gates

Allowed default reads:

- `transfer_data.csv`;
- `data/w_pool_k5_n1000_seed2026.json` through `load_weight_pool`;
- the `problem_input` passed by the judge;
- standard library and installed packages.

Default `answer.py` must not read:

- `results/exact_frontiers/`;
- `results/exact_public_headroom*`;
- `results/exact_guidance*`;
- `results/exact_guidance_samples/`;
- `results/oracle*`;
- `results/coverage_gap*`;
- `results/seed_*`, `results/lowcase_*`, `results/selector_*`, or other public tuning result files;
- `results/baseline_cache.json`;
- public case directories by enumeration, file name, or case suffix;
- exact state lists, exact objective anchors, or exact-derived lambda tables embedded as constants.

If normalization extrema are used inside `main1()`, document why they are normalization metadata and not a solution oracle. For the strictest contest interpretation, avoid full-state exact enumeration inside `main1()` and use only sampled/local objective comparisons.

### Sample Provenance Gates

- `sample_used == 100000`.
- `sample_spins.shape == (100000, problem.n)`.
- every value is `-1` or `+1`.
- every returned row is obtained from `Simulator.sampling()`.
- no local-search, exact-frontier, repaired, or cached classical row is inserted into `sample_spins`.
- warm-start candidates are allowed only as circuit initialization.

### Public-Overfit Gates

- Prefer one shared algorithm over public case tables.
- Any digest/filename/case-specific schedule must be called out as `gray` and requires explicit user approval.
- Do not encode exact public frontier information indirectly as constants, lambda IDs, state IDs, seed IDs, or per-case portfolios.
- Keep broad baseline coverage so tuning does not collapse on non-public cases.
- Evaluate guard cases, not just the target headroom cases.

### Verification Gates

Minimum checks before declaring merge-ready:

```powershell
python -m py_compile answer.py utils.py run.py baseline.py
python run.py --split public --max-cases 1 --large-shots 1000 --out results/public_onecase_smoke.json
```

For any scoring change:

```powershell
python run.py --split public --large-shots 1000 --out results/public_lowshots_merge_check.json
```

For a final candidate when time permits:

```powershell
python run.py --split public --out results/public_default_merge_check.json
```

If full public verification times out locally, use resumable per-case evidence, but report that limitation clearly.

## 7. Merge Priority Recommendation

1. Merge only contest-safe sampled-runtime improvements first: sampled-frontier warm-start selection, lambda allocation from already sampled data, and local heuristic warm starts that are not returned directly.
2. Test structural seed mixtures next. A global or stage-level multi-seed split is cleaner than expanding `_MAIN1_SEED_BY_DIGEST`.
3. Test fixed parameter portfolios after seed mixtures. Prioritize simple shared gamma/beta/warm-c portfolios with stable runtime.
4. Keep exact frontier, exact headroom, and exact coverage artifacts as diagnostic/oracle-only assets. Use them to choose experiments, not as runtime inputs.
5. Do not merge oracle exact quantum sampling into default `answer.py`. Only run it as a labeled demonstration if the user explicitly asks for oracle/public-saturation mode.
6. Avoid further per-public-case seed or parameter tables unless the user explicitly accepts public-overfit risk for the public-small sprint.

## 8. Bottom Line

The compliant merge lane is narrow but usable: all returned rows must remain MindQuantum samples, and any classical work may only choose circuits, warm starts, seeds, lambdas, and shot allocation without inserting or repairing samples.

Exact public frontiers and exact headroom artifacts are valuable for diagnosis and for proving the public upper bound, but they must not leak into default `answer.py`. The highest-risk current pattern is public digest seed tuning; the safest next merge candidates are structural seed mixtures, sampled-only warm-start selectors, and fixed parameter portfolios.
