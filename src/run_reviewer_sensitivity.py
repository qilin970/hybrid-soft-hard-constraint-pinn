from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Reviewer sensitivity entry: run confirmed sensitivity analysis from final_main baseline.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--base-dir', type=Path, default=None, help='Default: <repo>/results/final_main')
    parser.add_argument('--out-root', type=Path, default=None, help='Default: <repo>/results/confirmed_sensitivity')
    parser.add_argument('--run-root', type=Path, default=None, help='Optional explicit run dir.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lambda-m-values', type=str, default='0,0.01,0.1,1,10,50,100,500,1000')
    parser.add_argument('--lambda-s-values', type=str, default='0,0.001,0.01,0.1')
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent

    base_dir = args.base_dir if args.base_dir is not None else (repo_root / 'results' / 'final_main')
    out_root = args.out_root if args.out_root is not None else (repo_root / 'results' / 'confirmed_sensitivity')

    cmd = [
        sys.executable,
        str(src_dir / 'run_confirmed_sensitivity.py'),
        '--device', args.device,
        '--base-dir', str(base_dir),
        '--out-root', str(out_root),
        '--lambda-m-values', args.lambda_m_values,
        '--lambda-s-values', args.lambda_s_values,
    ]
    if args.run_root is not None:
        cmd += ['--run-root', str(args.run_root)]
    if args.resume:
        cmd.append('--resume')

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
