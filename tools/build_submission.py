"""Build a competition-format ZIP from the selected submission package."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from package_files import runtime_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'submissions/submit14_src')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/submit14.zip')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    try:
        package = runtime_files(source)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if output.is_relative_to(source):
        parser.error('--output must be outside the package source directory.')
    members = [path.relative_to(source).as_posix() for path in package]
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(source / member, member)
    with ZipFile(output) as archive:
        if sorted(archive.namelist()) != sorted(members) or archive.testzip() is not None:
            raise RuntimeError('Submission archive verification failed.')
    print(f'Verified submission archive: {output}')


if __name__ == '__main__':
    main()
