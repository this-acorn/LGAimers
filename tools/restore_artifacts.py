"""Fill missing artifact files from an original or organized local workspace."""

import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.from_dir.resolve()
    if not source.is_dir():
        parser.error('--from-dir must be an existing local workspace directory.')
    records = json.loads((ROOT / 'artifacts/manifest.json').read_text(encoding='utf-8'))['files']
    restored, unavailable = 0, []
    for item in records:
        destination = (ROOT / item['destination']).resolve()
        if not destination.is_relative_to(ROOT):
            raise ValueError('Artifact destination is outside the repository.')
        if destination.exists() or item['publish']:
            continue
        candidates = [source / item['destination']]
        if 'archive' not in item:
            candidates.append(source / item['path'])
        original = next((p for p in candidates if p.is_file()), None)
        if original:
            if original.stat().st_size != item['size']:
                unavailable.append(item['destination'])
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            restored += 1
        elif 'archive' in item:
            archive = next((p for p in [source / item['archive'], source / Path(item['archive']).name] if p.is_file()), None)
            if archive:
                with ZipFile(archive) as bundle:
                    data = bundle.read(item['member'])
                if len(data) != item['size']:
                    raise ValueError(f'Unexpected archive member size: {item["member"]}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                restored += 1
            else:
                unavailable.append(item['destination'])
        else:
            unavailable.append(item['destination'])
    print(f'Restored: {restored}; unavailable: {len(unavailable)}')
    for name in unavailable[:10]:
        print(f'  Missing: {name}')
    if unavailable:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
