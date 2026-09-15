"""Build a competition-format ZIP from the selected submission package."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ('script.py', 'requirements.txt', 'model/model.pkl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'submissions/submit14_src')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/submit14.zip')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    for member in MEMBERS:
        path = source / member
        if not path.is_file():
            parser.error(f'Required file is missing: {path}')
        if output == path:
            parser.error('--output must not overwrite a package input.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for member in MEMBERS:
            archive.write(source / member, member)
    with ZipFile(output) as archive:
        if sorted(archive.namelist()) != sorted(MEMBERS) or archive.testzip() is not None:
            raise RuntimeError('Submission archive verification failed.')
    print(f'Verified submission archive: {output}')


if __name__ == '__main__':
    main()
