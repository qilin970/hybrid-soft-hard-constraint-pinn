# SMOKE_TEST_REPORT

Date: 2026-03-29
Repository: `biomass-gasification-two-stage-reviewer`

## 1) Checked items

### Existence and readability check
Verified all required paths exist and are readable:
- `README.md`
- `requirements.txt`
- `.gitignore`
- `configs/seed_best_config.json`
- `configs/starter_config_fixedsplit.json`
- `data/experiment_data.xlsx`
- `data/Rand_lable.mat`
- `src/run_reviewer_main.py`
- `src/run_reviewer_sensitivity.py`
- `src/export_reviewer_figures.py`
- `results/final_main/best_model`
- `results/final_main/eval`
- `results/confirmed_sensitivity/summary`
- `results/confirmed_sensitivity/standard_figures`

Result: **PASS** (all present).

### Absolute-path scan (.py/.json/.txt/.md)
Scan keywords:
- absolute Windows drive path prefix
- old workspace path fragments
- old source/result directory fragments
- two known historical run-id fragments

Result:
- Runtime-related files: **no residual absolute paths found**.
- Remaining hits: only in `CLEANUP_REPORT.md` explanatory text.

Conclusion: **PASS**.

### Reviewer entry help checks
Executed from repository root:
- `python src/run_reviewer_main.py --help`
- `python src/run_reviewer_sensitivity.py --help`
- `python src/export_reviewer_figures.py --help`

Result: all three commands exited with code 0.

Conclusion: **PASS**.

### py_compile + minimal import checks
Checked scripts:
- `src/core_model.py`
- `src/fixedsplit_utils.py`
- `src/joint_physics_branch.py`
- `src/train_two_stage_correction.py`
- `src/evaluate_constraints_and_monotonicity.py`
- `src/run_confirmed_sensitivity.py`
- `src/export_scatter_plots.py`
- `src/export_standard_sensitivity_figures.py`
- `src/run_reviewer_main.py`
- `src/run_reviewer_sensitivity.py`
- `src/export_reviewer_figures.py`

Result:
- py_compile: 11/11 passed
- importlib minimal import: 11/11 passed

Conclusion: **PASS**.

## 2) Documentation consistency check
Compared:
- `README.md`
- `docs/reproduction.md`
- `docs/repository_scope.md`

Required consistency points:
1. single official mainline
2. final main reproduction command
3. confirmed sensitivity command
4. figure export command
5. output locations

Findings before fix:
- `docs/repository_scope.md` described scope correctly but did not explicitly list commands/output locations.

Action taken:
- Added explicit command section and default output-location section to `docs/repository_scope.md`.

Result after fix: the three documents are now consistent on mainline, commands, and output paths.

## 3) Fixes made during smoke test
Only light documentation fix was applied:
- Updated `docs/repository_scope.md`:
  - added `Official Mainline and Commands`
  - added `Default Output Locations`

No mathematical logic, training algorithm, model formulas, or architecture were changed.

## 4) Runtime-related absolute-path status
- **No runtime-related absolute paths remain** in `.py/.json/.txt/.md`.
- `CLEANUP_REPORT.md` still contains historical path examples in prose; this is acceptable and non-runtime.

## 5) Final decision
- Repository is now suitable for reviewer-facing usage and can be committed.
- Recommended next step: proceed with git initialization/commit/push.

