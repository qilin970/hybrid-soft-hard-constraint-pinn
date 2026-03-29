# Reproduction Guide

## 1) Final Main Model
Command:
```bash
python src/run_reviewer_main.py --device cuda --overwrite
```

Inputs (default):
- `data/experiment_data.xlsx`
- `data/Rand_lable.mat`
- baseline template from `results/final_main`

Outputs:
- synchronized final output directory: `results/final_main`
- archived timestamp runs: `results/two_stage_correction_runs/run_YYYYMMDD_HHMMSS`

## 2) Confirmed Sensitivity Analysis
Command:
```bash
python src/run_reviewer_sensitivity.py --device cuda
```

Inputs (default):
- `results/final_main/best_model/best_config.json`
- `results/final_main/run_config.xlsx`

Outputs:
- `results/confirmed_sensitivity/run_YYYYMMDD_HHMMSS/`
  - `grid_runs/`
  - `summary/`

## 3) Figure Export
Command:
```bash
python src/export_reviewer_figures.py
```

Exports:
- scatter plots from final main prediction files
- standard sensitivity figures from confirmed sensitivity summary

Default output locations:
- `results/scatter_plots`
- `results/confirmed_sensitivity/standard_figures`

## CUDA Requirement
- Main training and sensitivity scripts are designed for CUDA usage.
- If CUDA is unavailable, set `--device cpu` where supported, noting runtime may be much longer.

## Output Location Summary
- Final main model: `results/final_main`
- Confirmed sensitivity runs: `results/confirmed_sensitivity`
- Exported figures: `results/scatter_plots`, `results/confirmed_sensitivity/standard_figures`
