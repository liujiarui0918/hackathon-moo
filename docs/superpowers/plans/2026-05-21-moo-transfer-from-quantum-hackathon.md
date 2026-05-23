# MOO Hackathon Transfer Development Plan

Date: 2026-05-21

## 1. Goal

Adapt lessons from the previous `liujiarui0918/quantum-hackathon` project to this multi-objective Ising competition without copying MIQP-specific logic that does not fit the current judge.

Primary deliverable remains `answer.py` only. Supporting docs may stay in the workspace but should not be required by the judge.

## 2. Current Competition Surface

The local judge calls:

```python
main1(problem_input, sample_budget=100000, rng_seed=None) -> dict
main2(problem_input, shots=200000, rng_seed=None, chunk_size=4096) -> dict
```

`main1` requirements:

- Return exactly `sample_used == 100000`.
- Return `sample_spins` with shape `[100000, n_qubits]`.
- Values must be only `-1/+1`.
- Score is `100000 * max(HV_submit - HV_base, 0)` averaged over small cases.
- Official baseline uses the first 100 lambdas from `data/w_pool_k5_n1000_seed2026.json`, 1000 shots per lambda, QAOA depth `p=3`.

`main2` requirements:

- Return `hv`, `frontier_objectives_norm`, and `nd_count`.
- Local judge requires exact match to baseline frontier and HV, then awards only a small speed bonus.
- Any change to sampling order, RNG stream, normalization, sorting, or frontier merge can invalidate the bonus.

Important restrictions:

- `main1` cannot use classical algorithms to directly generate or repair quantum samples.
- Allowed: change quantum circuit structure, warm-start strategy, shot allocation, and post-sampling evaluation/selection.
- `main2` can optimize data processing but must not cheat by returning precomputed hidden outputs.

## 3. Current Repository Findings

Files reviewed:

- `README.md`
- `run.py`
- `baseline.py`
- `answer.py`
- `utils.py`
- `transfer_data.csv`
- public/large `.npz` headers

Data shape:

- Small public cases: `a=4`, `b=5`, `n=20`, `k=5`, `m=31`.
- Large cases: `a=40`, `b=50`, `n=2000`, `m=3910`, observed arrays have `k=6` in `.npz` even though large filenames say `k5`; code uses `problem.k`.

Current `answer.py` already implements a warm-start strategy:

- 100 lambdas per round.
- Shot split `600 / 200 / 200`.
- Round 1: standard QAOA.
- Later rounds: exact ND extraction from sampled states, crowding-distance seed selection, warm-start QAOA with `warm_c=0.4`.
- `main2` delegates to `large_random_frontier_hv`.

Current local environment:

- System Python 3.12 cannot install `mindquantum` on Windows.
- Miniforge is installed at `C:\Users\ljr-w\miniforge3-codex`.
- Working judge env is `moo-mq311` with Python 3.11.15, NumPy 2.4.6, `pygmo 2.19.8`, and `mindquantum 0.12.0`.
- A separate Python 3.12 env (`moo-test`) can import `pygmo`, but cannot install `mindquantum`.

## 4. Reference Project Review

Reference repo cloned to:

```text
C:\Users\ljr-w\Downloads\hackathon-moo-团队名称\quantum-hackathon-reference
```

Important reviewed files:

- `README.md`
- `submission/README_源代码包.md`
- `submission/混合整数优化问题赛道-平步青云-求解思路速读.md`
- `src/quantum_hackathon/miqp/route7.py`
- `src/quantum_hackathon/miqp_cli.py`
- `src/quantum_hackathon/solvers/qaoa/{runner,hamiltonian,backends}.py`
- `src/quantum_hackathon/solvers/simulated_annealing.py`
- `tests/test_miqp_route7.py`
- `tests/test_qaoa_route.py`

Reference project core idea:

- The previous competition was MIQP, not this MOO Ising task.
- Its winning logic did not solve a full huge QUBO directly.
- It selected high-value binary blocks, generated candidates with a solver portfolio, repaired constraints, solved continuous LP exactly, cached evaluations, and polished.
- QAOA was used as a small-block candidate generator, not as the whole solver.

Non-transferable parts:

- Continuous LP subproblem.
- Binary constraint repair.
- Benders/cut advice.
- MIQP block objective and QUBO penalty construction.
- Submission `.npz` result generation.

Transferable principles:

- Use structure-aware search surfaces.
- Avoid spending all budget on one scalarization policy.
- Use a portfolio of strategies, but keep final outputs within the legal quantum sampling channel.
- Use warm-starts only as circuit initialization, not as classical sample fabrication.
- Use post-sampling analysis to decide future quantum calls.
- Keep deterministic seeds and exact budget accounting.

## 5. Development Strategy

This task should be a conservative `answer.py` improvement, not a rewrite of the whole repo.

### 5.1 `main1` Strategy

Keep the legal MindQuantum QAOA sampling path. Improve the way lambdas and warm seeds are scheduled.

Planned enhancements:

1. Preserve a baseline anchor:
   - Always include early broad coverage from the first official lambdas so the result does not collapse on cases where warm-start hurts.

2. Add route7-style portfolio scheduling:
   - Use multiple lambda cohorts:
     - canonical first-100 lambdas,
     - deterministic sparse/extreme-biased lambdas from the 1000-pool,
     - frontier-following lambdas chosen from prior sampled ND points.
   - This maps old "block pool" to new "lambda pool".

3. Improve frontier seed selection:
   - Keep objective extremes as anchors.
   - Use crowding distance/diversity in normalized objective space.
   - Limit duplicates from the same lambda.
   - Add light hypervolume-contribution style priority when candidate count is small enough.

4. Use adaptive warm strength:
   - Round 1: no warm-start.
   - Later rounds: choose `warm_c` based on round and seed confidence.
   - Avoid over-concentrating by capping repeated warm seeds.

5. Keep exact sample budget:
   - Final `sample_spins` must have exactly 100000 rows.
   - All sampled rows must come from `Simulator.sampling`.

Candidate allocation to test:

```text
Round 0: 100 lambdas * 500 shots = 50000
Round 1: 100 lambdas * 300 shots = 30000
Round 2: 100 lambdas * 200 shots = 20000
Total: 100000
```

Fallback allocation if tests regress:

```text
Current stable sample: 600 / 200 / 200
```

### 5.2 `main2` Strategy

Default recommendation: keep `main2` behavior identical to baseline helper for correctness.

Reason:

- Judge compares the full normalized frontier against baseline with `np.allclose(..., atol=1e-8, rtol=0)`.
- Any faster alternative must preserve exact RNG sequence, chunk behavior, ND merge, row sorting, and HV output.
- The bonus weight is only `10`, while `main1` is the main score.

Possible safe micro-optimization:

- None unless we can prove byte-equivalent output and faster wall time.

## 6. Implementation Tasks

1. Add small helper functions in `answer.py`:
   - `_simplex_extreme_lambda_ids`
   - `_select_lambda_portfolio`
   - `_lambda_ids_from_frontier`
   - optional `_hv_priority_scores` or simpler diversity priority

2. Refactor `main1` round loop:
   - Track unique spins/counts/lambda ids as now.
   - After each round, compute exact sampled frontier.
   - Select warm bit bank and active lambda ids for next round using the new portfolio.

3. Keep all `utils.py` interfaces unchanged.

4. Avoid adding heavy dependencies or copying reference repo modules.

5. Review for:
   - exact shot count,
   - no classical-generated final samples,
   - no hidden-data assumptions,
   - deterministic seed behavior.

## 7. Verification Plan

Working Windows setup:

```powershell
$conda = "$env:USERPROFILE\miniforge3-codex\Scripts\conda.exe"
& $conda create -y -n moo-mq311 python=3.11 numpy pygmo pip
& $conda run -n moo-mq311 python -m pip install mindquantum
```

Verified commands:

```powershell
& $conda run -n moo-mq311 python -m py_compile answer.py utils.py baseline.py run.py
& $conda run -n moo-mq311 python run.py --split public --max-cases 1 --large-shots 1000
& $conda run -n moo-mq311 python run.py --split public --max-cases 3 --large-shots 1000
& $conda run -n moo-mq311 python run.py --split public --large-shots 1000
& $conda run -n moo-mq311 python run.py --split public --max-cases 1
```

Observed results:

- `py_compile` passed.
- `--max-cases 1 --large-shots 1000`: score `184.893373`, no timeout.
- `--max-cases 3 --large-shots 1000`: score `64.889250`, all large cases valid.
- full public small set with `--large-shots 1000`: score `77.163397`, all 10 large low-shot checks valid.
- `--max-cases 1` with default `large-shots=200000`: score `185.079932`, large frontier match valid.

Risk: MindQuantum wheels support Windows Python 3.9-3.11 for the tested 0.12.0 release, but not Python 3.12 in this environment.

## 8. Risk Notes

- Over-adaptive warm-start may reduce diversity and lower HV. Keep first-round broad coverage and objective extremes.
- Selecting lambdas from the full 1000 pool may improve hidden generality, but it may also underperform the official first-100 baseline if transfer parameters favor the original pool order.
- Exact frontier extraction after sampling is allowed because it only chooses future quantum circuit warm starts, not final classical samples.
- Any `main2` change is high risk because the judge demands exact frontier equality.

## 9. Acceptance Criteria

- `answer.py` imports successfully.
- `main1` returns exactly 100000 legal quantum-sampled spins in the judge environment.
- Public score does not regress on at least 1-3 cases; if it does, revert allocation to the current `600/200/200` split while keeping only safe seed selection improvements.
- `main2` remains valid against baseline frontier matching.
