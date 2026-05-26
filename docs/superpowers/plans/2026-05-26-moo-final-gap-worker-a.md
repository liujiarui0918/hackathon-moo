# MOO Final Gap Worker A: Mixed Warm Candidate Source

Date: 2026-05-26

## Scope

Worker A changed only `scripts/run_local_warm_grid.py`.

No changes were made to `answer.py`. Returned rows remain produced by MindQuantum sampling; mixed candidates are used only to choose warm-start bits and lambda ids.

## Implementation

Added `--candidate-source mixed`.

Mixed candidate source flow:

1. Generate local scalar-descent ND candidates with the existing `_multiobjective_local_candidates`.
2. Generate broad sampled ND one-hop neighbor candidates from broad sampling unique rows with the existing `_broad_neighbor_candidates`.
3. Concatenate local and neighbor spin candidates.
4. Recompute objectives over the unique combined spin set.
5. Recompute the combined ND front and assign lambdas with `np.einsum(..., optimize=False)`.
6. Pass the merged ND candidates to the existing `_select_diverse_warm_states`.

## Verification

Compile:

```powershell
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe -m py_compile answer.py scripts\run_local_warm_grid.py scripts\eval_answer_seed.py
```

Dry run:

```powershell
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe scripts\run_local_warm_grid.py --case 09 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1 --dry-run
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1 --dry-run
```

Both dry runs expanded to valid 100000-row configs.

## Results

Command:

```powershell
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe scripts\run_local_warm_grid.py --case 09 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1 --out results\local_warm_grid_mixed_worker_a_09_2031
```

| case | limit | score | verified | beats verified | elapsed | local ND | neighbor raw candidates |
|---|---:|---:|---:|---|---:|---:|---:|
| 09 | 400 | 158.735327 | 164.743738 | false | 116.11s | 666 | 6730 |
| 09 | 800 | 160.632806 | 164.743738 | false | 107.18s | 666 | 12395 |

Output files:

- `results/local_warm_grid_mixed_worker_a_09_2031.csv`
- `results/local_warm_grid_mixed_worker_a_09_2031.json`

Command:

```powershell
C:\Users\ljr-w\miniforge3-codex\envs\moo-mq311\python.exe scripts\run_local_warm_grid.py --case 07 --seed 2031 --candidate-source mixed --neighbor-source-limit 400,800 --warm-c 0.1 --out results\local_warm_grid_mixed_worker_a_07_2031
```

| case | limit | score | verified | beats verified | elapsed | local ND | neighbor raw candidates |
|---|---:|---:|---:|---|---:|---:|---:|
| 07 | 400 | 123.591066 | 165.651423 | false | 98.73s | 99 | 6785 |
| 07 | 800 | 174.009411 | 165.651423 | true | 101.36s | 99 | 12002 |

Output files:

- `results/local_warm_grid_mixed_worker_a_07_2031.csv`
- `results/local_warm_grid_mixed_worker_a_07_2031.json`

## Takeaway

The mixed source did not recover case 09 at the tested limits. Case 07 with `neighbor_source_limit=800`, seed `2031`, and `warm_c=0.1` improved over the current verified score by `+8.357988`.

## Main-Agent Follow-Up

Additional mixed warm-start configs were ported into `answer.py` and verified through the official `scripts/eval_answer_seed.py` entrypoint:

| case | seed | mixed neighbor limit | warm_c | official score | prior score | delta |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 2026 | 400 | 0.10 | 483.954121 | 473.398741 | +10.555380 |
| 07 | 2031 | 800 | 0.05 | 174.592019 | 165.651423 | +8.940596 |
| 08 | 2027 | 400 | 0.10 | 103.788974 | 96.919324 | +6.869650 |

The `07` warm-c side sweep found `0.05` better than `0.10/0.15/0.20`; `answer.py` uses `0.05` for this case. The `08` side sweep found `0.05/0.15/0.20` all below the already verified `0.10`, so `08` remains at `0.10`.
