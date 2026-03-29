# CLEANUP_REPORT

## 1) Files copied into reviewer repository

### Configs
- `configs/seed_best_config.json` (from `源/代码/configs/seed_best_config.json`)
- `configs/starter_config_fixedsplit.json` (from `源/代码/configs/starter_config_fixedsplit.json`)

### Data (public)
- `data/experiment_data.xlsx`
- `data/Rand_lable.mat`

### Core source files kept and renamed
- `00_core_hardbc_softmono_pipeline.py` -> `src/core_model.py`
- `01_train_final_model_fixedsplit.py` -> `src/fixedsplit_utils.py`
- `03_constraint_and_monotonic_figures.py` -> `src/evaluate_constraints_and_monotonicity.py`
- `06_train_final_model_fixedsplit_joint_physics_branch.py` -> `src/joint_physics_branch.py`
- `尝试修正.py` -> `src/train_two_stage_correction.py`
- `17_export_scatter_plots.py` -> `src/export_scatter_plots.py`
- `18_export_standard_sensitivity_figures.py` -> `src/export_standard_sensitivity_figures.py`
- `25_confirmed_sensitivity_analysis.py` -> `src/run_confirmed_sensitivity.py`

### Reviewer wrapper entry scripts added
- `src/run_reviewer_main.py`
- `src/run_reviewer_sensitivity.py`
- `src/export_reviewer_figures.py`

### Result artifacts copied (minimal required)
- `results/final_main/best_model`
- `results/final_main/eval`
- `results/final_main/run_config.xlsx`
- `results/final_main/fixed_split_indices.xlsx`
- `results/confirmed_sensitivity/summary`
- `results/confirmed_sensitivity/standard_figures`
- `results/scatter_plots`

## 2) Files/directories explicitly excluded

### Historical experiment scripts excluded
- `18_sensitivity_loss_weights.py`
- `19_export_loss_curves_from_sensitivity.py`
- `21_phase1_2d_sensitivity.py`
- `23_sensitivity_try1_final_losses.py`
- `24_sensitivity_try2_same_weights.py`
- `尝试修正_无回归补偿.py`

### Historical/non-reviewer result folders excluded
- `结果/尝试修正-无回归补偿`
- `结果/敏感性分析试跑`
- `结果/敏感性分析试跑-阶段1二维`
- `结果/敏感性分析尝试1：用特解的权重对应最后的损失`
- `结果/敏感性分析尝试2：两组权重一模一样`
- `结果/确定的敏感性分析方法/run_20260328_175657`
- `结果/确定的敏感性分析方法/*/grid_runs`
- other historical intermediate runs not required by reviewer mainline

## 3) Shell/wrapper scripts removed from reviewer mainline

These were not copied because the reviewer pipeline now directly uses renamed core scripts:
- `09_train_final_model_fixedsplit_jointphysics_powerD.py`
- `01_train_final_model_fixedsplit_partnet_trainable_exp1.py`
- `run_main_model.py`
- `00_core_hardbc_softmono_pipeline_powerD.py`
- `00_core_hardbc_softmono_pipeline_powerD_product_exp1.py`

## 4) Dependency/path fixes applied

### Script dependency rewiring (old -> new names)
- Updated all mainline references (`importlib`, default script paths, subprocess targets) to:
  - `core_model.py`
  - `fixedsplit_utils.py`
  - `joint_physics_branch.py`
  - `evaluate_constraints_and_monotonicity.py`
  - `train_two_stage_correction.py`

### Absolute path cleanup
- Removed hard-coded `D:\Projects\...` style defaults from reviewer source scripts.
- Updated path defaults to repository-relative resolution via `Path(__file__).resolve()`.
- Updated:
  - `configs/seed_best_config.json` (`split.rand_label_mat` -> `data/Rand_lable.mat`)
  - `results/final_main/best_model/best_config.json` (`split.rand_label_mat` -> `data/Rand_lable.mat`)
  - `results/final_main/eval/run_info.json` to relative metadata paths.

### Result folder normalization
- Flattened nested copy artifacts into required layout:
  - `results/confirmed_sensitivity/summary`
  - `results/confirmed_sensitivity/standard_figures`
  - `results/scatter_plots`

## 5) Documentation and packaging files added
- `README.md`
- `docs/reproduction.md`
- `docs/repository_scope.md`
- `requirements.txt`
- `.gitignore`
- `CLEANUP_REPORT.md`

## 6) Public data inclusion
- Yes. Included:
  - `data/experiment_data.xlsx`
  - `data/Rand_lable.mat`

## 7) Residual risks / non-automated items
- Some `.xlsx` files may still contain historical text/path cells from original runs. These are preserved as result artifacts; they were not edited because xlsx is binary and these fields are not required by the cleaned runtime path logic.
- The training/sensitivity scripts are CUDA-oriented for practical runtime; CPU mode is supported in wrapper arguments where applicable but may be significantly slower.

## 8) Single official reviewer mainline
- **Official final mainline**: `two-stage correction`
- Entry: `python src/run_reviewer_main.py`
- Confirmed sensitivity (secondary official analysis): `python src/run_reviewer_sensitivity.py`

