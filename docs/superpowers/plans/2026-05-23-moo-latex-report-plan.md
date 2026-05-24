# Quantum MOO Solver LaTeX Report Plan

## 1. Goal

Write a detailed Chinese LaTeX report for the current quantum multi-objective Ising hackathon project.

The report must explain the work in language that a primary-school student or complete outsider can follow. It should still be technically complete enough to describe the problem, data, algorithms, model construction, innovations, tuning process, compliance caveats, and final results.

## 2. Deliverables

- `reports/moo_solver_report.tex`: main Chinese LaTeX source.
- `reports/moo_solver_report.pdf`: compiled PDF if LaTeX compilation succeeds.
- Optional helper assets are reused from existing `fig/` images:
  - `fig/grid_topology.png`
  - `fig/pareto_hv.png`

## 3. Source Material

Use only project-local sources unless a cited external reference is already present in `README.md`.

Primary local sources:

- `README.md`
  - problem statement;
  - task interface;
  - scoring formula;
  - fairness and banned actions;
  - dataset descriptions.
- `answer.py`
  - final algorithm;
  - broad QAOA sampling;
  - multi-objective local-search warm-start;
  - deterministic seed schedule;
  - `main2` implementation.
- `utils.py`
  - Ising data structure;
  - QAOA circuit builder;
  - energy calculation;
  - normalization;
  - hypervolume calculation;
  - random large-case frontier.
- `run.py`
  - local judge;
  - score formulas;
  - validation logic.
- `docs/superpowers/plans/2026-05-21-moo-transfer-from-quantum-hackathon.md`
  - transfer analysis from the previous competition.
- `docs/superpowers/plans/2026-05-22-moo-score-tuning.md`
  - detailed tuning history and final scores.
- `results/seed_schedule_with01_public_default.json`
  - final verified result, although this file is ignored by git.

## 4. Report Structure

Recommended sections:

1. Title page and abstract
   - short story of the task;
   - final score summary.
2. Problem introduction
   - explain multi-objective optimization with everyday analogies;
   - explain Ising spins as small switches;
   - explain Pareto frontier and HV without assuming prior knowledge.
3. Data description
   - public small cases: 10 files, `4 x 5` grid, 20 spins, 5 objectives;
   - large cases: 10 files, `40 x 50` grid, 2000 spins, 5 objectives;
   - explain `edges`, `weights`, and `h` arrays.
4. Contest interfaces and scoring
   - `main1`;
   - `main2`;
   - `100000` shots budget;
   - score formula in simple words and LaTeX.
5. Baseline method
   - `100` lambda weights, `1000` shots each;
   - standard QAOA.
6. Our algorithm
   - broad scalarization coverage;
   - QAOA circuit construction;
   - local-search warm-start as initial-state chooser only;
   - diverse frontier selection;
   - deterministic seed schedule and its risk note;
   - `main2` data-processing path.
7. Model building details
   - turn 5 objectives into one scalarized Ising objective with lambda;
   - build cost Hamiltonian from projected `J` and `h`;
   - initialize circuit;
   - cost layer and mixer layer;
   - convert sampling bitstrings to spins.
8. Innovation points
   - stop inheriting old `100`-lambda constraint;
   - spend shots on coverage rather than repeated sampling;
   - local-search states guide warm-start but are not returned;
   - weak warm-start strength preserves quantum exploration;
   - empirical seed sensitivity analysis.
9. Tuning process
   - original score around `77.112309`;
   - broad lambda strategy improved coverage;
   - hybrid broad + multi-objective warm-start reached around `205.154696`;
   - seed schedule pushed final default public score to `213.769137`;
   - explain failed or rejected attempts.
10. Final results
   - compile final public default table;
   - include all relevant scores and timeout status.
11. Compliance and limitations
   - final samples come from MindQuantum;
   - classical search is only a warm-start guide;
   - public-case seed schedule and exact normalization are potential gray areas;
   - recommend safer production submission options if needed.
12. Conclusion
   - what worked;
   - what to improve next.

## 5. Writing Style Rules

- Use simple Chinese.
- Prefer short sentences.
- Explain technical terms the first time they appear.
- Use analogies:
  - spin = small switch;
  - lambda = taste knob or weighing scale;
  - Pareto frontier = menu of best trade-offs;
  - HV = area/volume covered by good choices.
- Avoid inflated claims.
- State limitations honestly.
- Reduce "AI味":
  - use concrete numbers;
  - use first-person plural where natural;
  - describe actual trial-and-error;
  - avoid vague slogans.

## 6. Subagent Work Split

Spawn 4 subagents:

1. Problem and data explainer
   - Read `README.md`, `utils.py`, and data filenames.
   - Produce simple Chinese notes for problem, data, interfaces, scoring.
2. Algorithm implementation explainer
   - Read `answer.py` and `utils.py`.
   - Produce notes explaining final `main1` and `main2` algorithms line-by-line at a conceptual level.
3. Tuning and results explainer
   - Read both plan docs and final result JSON.
   - Produce a clean tuning timeline and result tables.
4. Plain-language editor and compliance reviewer
   - Review README banned actions and answer implementation.
   - Produce a compliance section and wording suggestions that are honest but not alarmist.

## 7. Implementation Steps

1. Create `reports/` if missing.
2. Collect subagent notes.
3. Write `reports/moo_solver_report.tex`.
4. Use `ctexart` with XeLaTeX for Chinese.
5. Include existing figures with relative paths.
6. Compile with:

   ```powershell
   xelatex -interaction=nonstopmode -halt-on-error moo_solver_report.tex
   ```

   from the `reports/` directory.
7. If compilation fails, fix LaTeX issues and recompile.
8. Review PDF/log for missing images, overfull boxes, and broken references.

## 8. Acceptance Criteria

- Report source exists at `reports/moo_solver_report.tex`.
- PDF compiles at `reports/moo_solver_report.pdf`.
- Report includes:
  - problem introduction;
  - data description;
  - algorithm and model construction;
  - innovation points;
  - tuning process;
  - final result;
  - compliance and limitations.
- Tone is understandable to non-specialists.
- Final report uses concrete numbers from the project.
