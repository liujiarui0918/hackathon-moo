# MOO Seed-Cohort Sprint Plan

Date: 2026-05-26

## Current Verified Baseline

Latest full public verification:

```powershell
python run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_seedmix09_fullcheck.json
```

Result:

```text
score: 222.575303
score_k5: 222.575303
elapsed: 2051.73s
timeout: False
```

Per-case state:

| case | current | exact upper | remaining | current tactic |
|---|---:|---:|---:|---|
| 00 | 483.954121 | 500.958844 | 17.004723 | mixed warm, limit 400, warm_c 0.10 |
| 01 | 98.046167 | 102.321625 | 4.275458 | local warm, seed 2031 |
| 02 | 238.777169 | 246.194251 | 7.417082 | broad-neighbor, limit 1200, warm_c 0.10 |
| 03 | 302.332699 | 306.045598 | 3.712899 | local warm, seed 2029 |
| 04 | 259.768537 | 282.358431 | 22.589894 | broad-neighbor, limit 1200, warm_c 0.125, 40k/60k budget |
| 05 | 138.642082 | 142.611213 | 3.969131 | broad-neighbor, limit 800, warm_c 0.15 |
| 06 | 255.177610 | 262.984747 | 7.807137 | broad-neighbor, limit 800, warm_c 0.20 |
| 07 | 174.592019 | 186.729892 | 12.137873 | mixed warm, limit 800, warm_c 0.05 |
| 08 | 103.788974 | 109.296972 | 5.507998 | mixed warm, limit 400, warm_c 0.10 |
| 09 | 171.914006 | 191.122271 | 19.208265 | weighted in-budget seed cohort 2031:2041 = 60:40 |

## Core Hypothesis

Case `09` proved that multi-seed sampling can improve under the same returned-row budget:

- Old single seed `09`: `164.743738`.
- Offline full-run union `2031+2041`: `166.059364`.
- Implemented in-budget cohort `2031+2041`: `170.673648`.

This means the improvement is not just from extra rows. Splitting per-circuit shots across complementary seeds can change the sampled support enough to improve HV.

The next highest-value question is whether the same mechanism transfers to `04`, `00`, `07`, `02`, `06`, or small-gap cases `01/03/05/08`.

## Constraints

- Returned `sample_spins` must come from MindQuantum `Simulator.sampling`.
- Seed cohorts must keep `sample_used == 100000`.
- No exact frontier or exact-guidance artifact may be loaded in `answer.py`.
- Experimental scripts may read results for analysis, but `answer.py` must remain self-contained.
- Runtime target remains below 1 hour on public. Latest full public leaves roughly `1548s` margin locally, but `09` already costs `333.8s`, so adding many cohorts can consume margin quickly.

## Known Rejects To Avoid

- `09`: mixed `400/800`, broad-neighbor first-500 `800/1200`, full lambda `1000x50`, HV-proxy mixed `400`.
- `09`: runtime-only `(2027,2031)` cohort did not finish within 15 minutes.
- `04`: mixed `400`, HV-proxy broad-neighbor `1200`, offset `[500,1000)` mixed, budget sides `350/325` and `450/275`.
- `00`: mixed limit `200/600/800`, warm-c side `0.05/0.15/0.20`.
- `07`: mixed limit `1000/1200`.
- `01`: local warm-c side `0.075/0.125/0.15/0.20`.
- `05`: broad-neighbor `800`, warm_c `0.15`, budget `400x100+300x200`.

## Workstreams

### Worker A: Seed-Cohort Candidate Analysis

Scope:

- Read `results`, plans, and `answer.py`.
- Do not write files.
- Rank seed-cohort experiments by expected score gain and runtime risk.

Output:

- 5 concrete commands.
- Comparison baseline for each.
- Risk notes.

### Worker B: Seed-Cohort Grid Harness

Scope:

- Add `scripts/run_seed_cohort_grid.py`.
- Optional worker note document.
- Do not modify `answer.py`.

Requirements:

- Monkeypatch `answer._MAIN1_SEED_MIX_CONFIG[digest] = mix` only inside the experiment process.
- Call `answer.main1(problem, rng_seed=first_seed)`.
- Score through `run._hv_from_spins` and `baseline_hv`.
- Dry-run by default.
- Write CSV/JSON.

### Main Agent

1. Keep heavy experiments serial.
2. First verify the harness with the known `09` mix `2031+2041`; it should reproduce about `170.673648`.
3. Test one non-09 high-headroom case first, likely `04` or `07`, with a two-seed mix whose single-seed members have prior evidence.
4. Merge into `answer.py` only if formal `scripts/eval_answer_seed.py` beats the current case score.
5. Run at least one guard case after touching shared main1 logic.
6. Commit and push only verified winners and durable docs/scripts.

## Candidate Experiments

Initial queue:

```powershell
python scripts\run_seed_cohort_grid.py --case 09 --mixes 2031+2041 --out results\seed_cohort_known_09 --run
python scripts\run_seed_cohort_grid.py --case 07 --mixes 2031+2029 2031+2041 --out results\seed_cohort_07_next --run
python scripts\run_seed_cohort_grid.py --case 04 --mixes 2026+2028 2026+2031 --out results\seed_cohort_04_next --run
python scripts\run_seed_cohort_grid.py --case 00 --mixes 2026+2029 2026+2031 --out results\seed_cohort_00_next --run
python scripts\run_seed_cohort_grid.py --case 02 --mixes 2041+2026 2041+2031 --out results\seed_cohort_02_next --run
```

Run order may change after Worker A reports.

## Results

Tool verification:

```powershell
python -m py_compile scripts\run_seed_cohort_grid.py answer.py scripts\eval_answer_seed.py
python scripts\run_seed_cohort_grid.py --case 09 --mixes 2031+2041 --out results\seed_cohort_known_09 --dry-run
python scripts\run_seed_cohort_grid.py --case 09 --mixes 2031+2041 --out results\seed_cohort_known_09 --run --rerun
```

The known `09` cohort reproduced the merged score:

| case | mix | score | baseline | outcome |
|---|---|---:|---:|---|
| 09 | 2031+2041 | 170.673648 | 170.673648 | harness valid |

Additional cohort/budget tests:

| case | experiment | score | current | outcome |
|---|---|---:|---:|---|
| 07 | in-budget cohort `2031+2033` | 121.220306 | 174.592019 | reject |
| 04 | in-budget cohort `2026+2028` | 251.487590 | 259.768537 | reject |
| 04 | narrow budget `410x100 + 295x200` | 260.136560 | 259.768537 | superseded |
| 04 | fine budget `414x100 + 293x200` | 260.930525 | 259.768537 | merge |
| 04 | right budget `416x100 + 292x200` | 255.905231 | 260.930525 | reject |
| 04 | right budget `418x100 + 291x200` | 256.652010 | 260.930525 | reject |
| 04 | right budget `420x100 + 290x200` | 253.276220 | 260.930525 | reject |
| 04 | right budget `422x100 + 289x200` | 259.452070 | 260.930525 | reject |
| 04 | budget `414x100 + 293x200`, warm-c `0.1125` | 252.509012 | 260.930525 | reject |
| 04 | budget `414x100 + 293x200`, warm-c `0.1375` | 245.361267 | 260.930525 | reject |
| 00 | mixed seed side `2024` | 483.796882 | 483.954121 | reject |
| 00 | mixed seed side `2028` | 457.216912 | 483.954121 | reject |
| 00 | mixed seed side `2031` | 481.530488 | 483.954121 | reject |
| 00 | mixed limit `400`, `warm_c=0.10`, budget `400x100+300x200` | 482.073793 | 483.954121 | reject |
| 07 | mixed budget `450x100 + 275x200` | 176.725481 | 174.592019 | merge |
| 07 | mixed budget `470x100 + 265x200` | 176.689509 | 174.592019 | reject |
| 07 | mixed budget `490x100 + 255x200` | 116.402230 | 174.592019 | reject |
| 07 | mixed budget `510x100 + 245x200` | 174.589428 | 174.592019 | reject |
| 07 | mixed budget `530x100 + 235x200` | 118.545080 | 174.592019 | reject |
| 07 | mixed budget `550x100 + 225x200` | 161.964882 | 174.592019 | reject |
| 07 | finer budget `420x100 + 290x200` | 111.165193 | 176.725481 | reject |
| 07 | finer budget `430x100 + 285x200` | 112.480203 | 176.725481 | reject |
| 07 | finer budget `440x100 + 280x200` | 112.480203 | 176.725481 | reject |
| 07 | finer budget `460x100 + 270x200` | 175.658939 | 176.725481 | reject |
| 07 | finer budget `480x100 + 260x200` | 174.173827 | 176.725481 | reject |
| 08 | mixed budget `450x100 + 275x200` | 100.273463 | 103.788974 | reject |
| 08 | mixed budget `470x100 + 265x200` | 102.620177 | 103.788974 | reject |
| 08 | mixed budget `530x100 + 235x200` | 96.606418 | 103.788974 | reject |
| 08 | mixed budget `550x100 + 225x200` | 97.671153 | 103.788974 | reject |
| 05 | broad-neighbor budget `450x100 + 275x200` | 132.076755 | 138.642082 | reject |
| 05 | broad-neighbor budget `470x100 + 265x200` | 139.463996 | 138.642082 | reject |
| 05 | broad-neighbor budget `530x100 + 235x200` | 138.601648 | 138.642082 | reject |
| 05 | broad-neighbor budget `550x100 + 225x200` | 140.818609 | 138.642082 | merge |
| 05 | right budget `560x100 + 220x200` | 140.116360 | 140.818609 | reject |
| 05 | right budget `570x100 + 215x200` | 134.743161 | 140.818609 | reject |
| 05 | right budget `580x100 + 210x200` | 137.526793 | 140.818609 | reject |
| 05 | right budget `590x100 + 205x200` | 126.534882 | 140.818609 | reject |
| 05 | right budget `600x100 + 200x200` | 138.623111 | 140.818609 | reject |
| 06 | in-budget cohort `2028+2033` | 249.905959 | 255.177610 | reject |
| 06 | warm-c fine `0.175` | 252.812732 | 255.177610 | reject |
| 06 | warm-c fine `0.225` | 253.982764 | 255.177610 | reject |
| 02 | in-budget cohort `2041+2026` | 229.247547 | 238.777169 | reject |
| 02 | in-budget cohort `2041+2031` | 227.392065 | 238.777169 | reject |
| 02 | warm-c fine `0.075` | 229.427119 | 238.777169 | reject |
| 02 | warm-c fine `0.125` | 237.169689 | 238.777169 | reject |
| 09 | in-budget cohort `2031+2041+2027+2025` | 168.117225 | 170.673648 | reject |
| 09 | in-budget cohort `2031+2041+2025+2027+2030` | 137.937268 | 170.673648 | reject |
| 09 | weighted cohort `2031:7+2041:3` | 170.597596 | 170.673648 | reject |
| 09 | weighted cohort `2031:3+2041:2` | 171.914006 | 170.673648 | merge |
| 09 | weighted cohort `2031:4+2041:1` | 170.184076 | 171.914006 | reject |
| 09 | weighted cohort `2031:2+2041:3` | 170.099675 | 171.914006 | reject |
| 09 | weighted cohort `2031:11+2041:9` | 172.714670 | 171.914006 | merge |
| 09 | weighted cohort `2031:13+2041:7` | 171.250137 | 171.914006 | reject |
| 09 | weighted cohort `2031:14+2041:11` | 171.939125 | 171.914006 | reject |
| 09 | weighted cohort `2031:16+2041:9` | 171.590600 | 171.914006 | reject |

Conclusion:

- Seed cohorts are not generally beneficial.
- The useful cohort signal is currently case-specific to `09`.
- The best verified `09` split is weighted `2031:2041 = 60:40`, implemented as per-circuit shots `60/40` for broad circuits and `120/80` for warm circuits.
- More seeds and over-biasing to `2031` both hurt.
- `02/04/06/07` cohorts tested so far all lose score, so seed-cohort work should not be generalized without direct evidence.
- Side checks around the old `09` split rejected both `80/20` and `40/60`; a finer denominator-20 scan found `2031:11+2041:9` as the best tested weighted cohort.

2-core / 4GB runner note:

| mode | case | score | elapsed | outcome |
|---|---|---:|---:|---|
| serial default | 08 | 103.788974 | 93.906s | keep |
| `MOO_MAIN1_WORKERS=2`, process backend | 08 | 103.788974 | 229.531s | reject |
| `MOO_MAIN1_WORKERS=2`, thread backend | 08 | 103.788974 | 95.409s | reject as default |

Conclusion: keep main1 sampling serial by default on the 2-core/4GB target. Process parallelism is too slow; thread parallelism does not buy measurable time on the tested case.

Formal merge verification:

```powershell
python scripts\eval_answer_seed.py --case data\public\k5_grid4x5_09.npz --seed 2031
python scripts\eval_answer_seed.py --case data\public\k5_grid4x5_04.npz --seed 2026
```

Results:

```text
09 weighted 55/45: score=172.714670, rows=100000, elapsed=257.456s
04 guard:          score=260.930525, rows=100000, elapsed=132.620s
```

Full-public verification after merging case `04/05/07/09` winners:

```powershell
python run.py --split public --max-cases 0 --large-shots 1000 --out results\public_after_budget_seed_sprint.json
```

```text
score: 223.326602
score_k5: 223.326602
elapsed: 1506.65s
timeout: False
```

## Merge Gate

For any case:

1. Experiment score must beat current score by at least `+0.25` case-score, or by any amount for small-gap cases if runtime does not increase.
2. The same configuration must be ported to `answer.py`.
3. `scripts/eval_answer_seed.py --case ... --seed <first_seed>` must reproduce the win.
4. If the change adds cohorts to a slow case, run `run.py --split public --max-cases 0 --large-shots 1000` before finalizing.
