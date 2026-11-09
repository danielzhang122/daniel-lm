"""Start or resume the full English Wikipedia stage with fixed settings."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=550550)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / 'runs' / 'full'
    command = [sys.executable, str(root / 'train.py'), '--data', 'E:/wiki-llm/data/full',
               '--out', str(output), '--steps', str(args.steps), '--schedule-steps', '550550',
               '--warmup', '200', '--lr', '0.0001', '--min-lr', '0.00001', '--eval-every', '500',
               '--batch', '16', '--accum', '1', '--optimizer-backend', 'fused']
    if (output / 'latest.pt').exists():
        command += ['--resume']
    else:
        command += ['--init-from', str(root / 'runs' / 'clean-100k' / 'latest.pt')]
    try:
        return subprocess.call(command, cwd=root)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
