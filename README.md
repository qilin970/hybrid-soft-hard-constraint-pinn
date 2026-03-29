# Biomass Gasification Two-Stage Reviewer Package

## Overview
This repository is a cleaned, reviewer-facing subset of a larger biomass gasification modeling workspace. It is organized to provide a single reproducible pipeline for review.

The **only official pipeline in this repository** is the **two-stage correction pipeline**. Historical exploratory scripts and alternative intermediate pipelines are intentionally excluded from the reviewer-facing workflow. The mathematical model form and loss terms are preserved, while the training workflow in this package is organized as a single official two-stage correction procedure.

## Quick Start
After setting up the environment, the full reviewer workflow is:

```bash
python src/run_reviewer_main.py --device cuda --overwrite
python src/run_reviewer_sensitivity.py --device cuda
python src/export_reviewer_figures.py
```

These commands reproduce the final main model, run the confirmed sensitivity analysis, and export the reviewer-facing figures.

## Final Reviewer-Facing Pipeline

The reviewer workflow is intentionally limited to three entry points:

1. `src/run_reviewer_main.py`
   Reproduces the final main model using the official two-stage correction pipeline.

2. `src/run_reviewer_sensitivity.py`
   Runs the confirmed sensitivity analysis based on the same reviewer-facing pipeline.

3. `src/export_reviewer_figures.py`
   Exports scatter plots, standard sensitivity figures, and related reviewer-facing figures.

## Repository Structure

```text
biomass-gasification-two-stage-reviewer/
  configs/
  data/
  src/
  results/
    final_main/
    confirmed_sensitivity/
      summary/
      standard_figures/
    scatter_plots/
  docs/
  CLEANUP_REPORT.md
```

## Data Files

This package includes the public data files required by the fixed-split workflow:

* `data/experiment_data.xlsx`
* `data/Rand_lable.mat`

## Environment Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the Final Main Model

```bash
python src/run_reviewer_main.py --device cuda --overwrite
```

Default output directory:

* `results/final_main/`

This step reproduces the final reviewer-facing main model and its associated evaluation outputs.

## Run the Confirmed Sensitivity Analysis

```bash
python src/run_reviewer_sensitivity.py --device cuda
```

Default output directory:

* `results/confirmed_sensitivity/run_YYYYMMDD_HHMMSS/`

This step runs the confirmed sensitivity analysis used in the reviewer-facing package. The repository also includes a minimal archived sensitivity summary and standard exported figures.

## Export Figures

```bash
python src/export_reviewer_figures.py
```

Default output directories:

* Scatter plots: `results/scatter_plots/`
* Standard sensitivity figures: `results/confirmed_sensitivity/standard_figures/`

## Included Result Files

This repository includes the minimal result artifacts needed for inspection and figure reproduction:

* `results/final_main/best_model`
* `results/final_main/eval`
* `results/confirmed_sensitivity/summary`
* `results/confirmed_sensitivity/standard_figures`
* `results/scatter_plots`

Some result files are not only archived outputs but are also reused by downstream evaluation or figure-export scripts, so they are intentionally retained in this cleaned package.

## Scope Note

This repository is intentionally cleaned for review submission.

* The **two-stage correction pipeline** is the only official main pipeline in this package.
* Historical exploratory scripts, discarded trial pipelines, and large intermediate result folders are excluded.
* The retained code, data, and results are limited to the minimum reviewer-facing subset needed for reproducibility and inspection.

For more detail, see:

* `docs/reproduction.md`
* `docs/repository_scope.md`
* `CLEANUP_REPORT.md`
