"""Run a standalone submission package using the active Python environment."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from package_files import runtime_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'submissions/submit14_src')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/submission.csv')
    args = parser.parse_args()
    source, data_dir, output = (p.resolve() for p in (args.source, args.data_dir, args.output))
    try:
        package = runtime_files(source)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    required = [data_dir / 'test.csv']
    for path in required:
        if not path.is_file():
            parser.error(f'Required file is missing: {path}')
    if output.is_relative_to(source) or output in required or output == data_dir / 'sample_submission.csv':
        parser.error('--output must not overwrite an input file.')

    with tempfile.TemporaryDirectory(prefix='lgaimers-inference-') as directory:
        stage = Path(directory)
        (stage / 'data').mkdir()
        for path in package:
            destination = stage / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        shutil.copy2(data_dir / 'test.csv', stage / 'data/test.csv')
        sample = data_dir / 'sample_submission.csv'
        if sample.is_file():
            shutil.copy2(sample, stage / 'data/sample_submission.csv')
        subprocess.run([sys.executable, '-X', 'utf8', 'script.py'], cwd=stage, check=True)
        result = stage / 'output/submission.csv'
        if not result.is_file():
            raise RuntimeError('Inference completed without producing output/submission.csv.')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, output)
    print(f'Prediction file: {output}')


if __name__ == '__main__':
    main()
