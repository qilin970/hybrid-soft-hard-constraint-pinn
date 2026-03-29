from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def latest_run(run_root: Path) -> Path:
    runs = [p for p in run_root.glob('run_*') if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f'No run_* directory found under: {run_root}')
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description='Reviewer main entry: run two-stage correction and sync outputs to results/final_main.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data', type=Path, default=None, help='Default: <repo>/data/experiment_data.xlsx')
    parser.add_argument('--rand-label-mat', type=Path, default=None, help='Default: <repo>/data/Rand_lable.mat')
    parser.add_argument('--baseline-dir', type=Path, default=None, help='Default: <repo>/results/final_main')
    parser.add_argument('--runs-root', type=Path, default=None, help='Default: <repo>/results/two_stage_correction_runs')
    parser.add_argument('--out-dir', type=Path, default=None, help='Default: <repo>/results/final_main')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite out-dir with the latest completed run.')
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent

    data_xlsx = args.data if args.data is not None else (repo_root / 'data' / 'experiment_data.xlsx')
    rand_label_mat = args.rand_label_mat if args.rand_label_mat is not None else (repo_root / 'data' / 'Rand_lable.mat')
    baseline_dir = args.baseline_dir if args.baseline_dir is not None else (repo_root / 'results' / 'final_main')
    runs_root = args.runs_root if args.runs_root is not None else (repo_root / 'results' / 'two_stage_correction_runs')
    out_dir = args.out_dir if args.out_dir is not None else (repo_root / 'results' / 'final_main')

    train_py = src_dir / 'train_two_stage_correction.py'
    cmd = [
        sys.executable,
        str(train_py),
        '--device', args.device,
        '--data', str(data_xlsx),
        '--rand-label-mat', str(rand_label_mat),
        '--baseline-dir', str(baseline_dir),
        '--out-root', str(runs_root),
    ]
    subprocess.run(cmd, check=True)

    newest = latest_run(runs_root)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{out_dir} already exists. Re-run with --overwrite to replace it.')
        shutil.rmtree(out_dir)
    shutil.copytree(newest, out_dir)

    print(f'latest_run={newest}')
    print(f'final_main_synced={out_dir}')


if __name__ == '__main__':
    main()
