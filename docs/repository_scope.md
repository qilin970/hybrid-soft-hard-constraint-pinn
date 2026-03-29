# Repository Scope

## Why this subset exists
This repository is a reviewer-facing cleanup of a larger local workspace.
The goal is to provide one clear, reproducible pipeline without historical branching noise.

## Official scope retained
- two-stage correction main pipeline
- required fixed-split utilities and joint-physics module
- evaluation pipeline for constraints and monotonicity
- confirmed sensitivity analysis and figure export scripts
- public data files and minimal required results

## Official Mainline and Commands
- Single official mainline: **two-stage correction**
- Final main reproduction:
  - `python src/run_reviewer_main.py --device cuda --overwrite`
- Confirmed sensitivity reproduction:
  - `python src/run_reviewer_sensitivity.py --device cuda`
- Figure export:
  - `python src/export_reviewer_figures.py`

## Default Output Locations
- Final main outputs: `results/final_main`
- Timestamped archived runs: `results/two_stage_correction_runs/run_YYYYMMDD_HHMMSS`
- Confirmed sensitivity runs: `results/confirmed_sensitivity/run_YYYYMMDD_HHMMSS`
- Exported figures:
  - `results/scatter_plots`
  - `results/confirmed_sensitivity/standard_figures`

## Excluded historical content
Excluded on purpose:
- historical sensitivity trial scripts
- historical no-compensation correction variant
- wrapper shell scripts for legacy multi-entry chains
- non-essential historical run folders and intermediate artifacts

## Relationship to full workspace
The full local workspace includes many exploratory and historical files.
This reviewer package keeps only the minimal reproducible subset and standardizes names/paths for readability and portability.

## Minimal required results included
- `results/final_main/best_model`
- `results/final_main/eval`
- `results/confirmed_sensitivity/summary`
- `results/confirmed_sensitivity/standard_figures`
- `results/scatter_plots`
