from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified figure export entry for reviewer package.')
    parser.add_argument('--best-dir', type=Path, default=None, help='Default: <repo>/results/final_main/best_model')
    parser.add_argument('--scatter-out-dir', type=Path, default=None, help='Default: <repo>/results/scatter_plots')
    parser.add_argument('--sensitivity-run-dir', type=Path, default=None, help='Default: <repo>/results/confirmed_sensitivity')
    parser.add_argument('--sensitivity-out-dir', type=Path, default=None, help='Default: <repo>/results/confirmed_sensitivity/standard_figures')
    parser.add_argument('--skip-scatter', action='store_true')
    parser.add_argument('--skip-sensitivity', action='store_true')
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent

    best_dir = args.best_dir if args.best_dir is not None else (repo_root / 'results' / 'final_main' / 'best_model')
    scatter_out = args.scatter_out_dir if args.scatter_out_dir is not None else (repo_root / 'results' / 'scatter_plots')
    sens_run = args.sensitivity_run_dir if args.sensitivity_run_dir is not None else (repo_root / 'results' / 'confirmed_sensitivity')
    sens_out = args.sensitivity_out_dir if args.sensitivity_out_dir is not None else (repo_root / 'results' / 'confirmed_sensitivity' / 'standard_figures')

    if not args.skip_scatter:
        subprocess.run(
            [
                sys.executable,
                str(src_dir / 'export_scatter_plots.py'),
                '--best-dir', str(best_dir),
                '--out-dir', str(scatter_out),
            ],
            check=True,
        )

    if not args.skip_sensitivity:
        subprocess.run(
            [
                sys.executable,
                str(src_dir / 'export_standard_sensitivity_figures.py'),
                '--run-dir', str(sens_run),
                '--out-dir', str(sens_out),
            ],
            check=True,
        )


if __name__ == '__main__':
    main()
