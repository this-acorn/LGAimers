"""Discover runtime files in the historical standalone package layouts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runtime_files(source: Path) -> list[Path]:
    for name in ('script.py', 'requirements.txt'):
        if not (source / name).is_file():
            raise FileNotFoundError(f'Required package file is missing: {source / name}')
    manifest = ROOT / 'artifacts/manifest.json'
    if manifest.is_file() and source.is_relative_to(ROOT):
        prefix = source.relative_to(ROOT).as_posix() + '/'
        missing = [item['destination'] for item in json.loads(manifest.read_text(encoding='utf-8'))['files']
                   if item['destination'].startswith(prefix)
                   and not (ROOT / item['destination']).is_file()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} package assets are missing, including {missing[0]}. '
                'See artifacts/README.md to restore local artifacts.')
    result = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {'__pycache__', '.git', 'data', 'output'} for part in relative.parts):
            continue
        if path.suffix in {'.pyc', '.pyo'}:
            continue
        if relative.parts[0] in {'model', 'src'} or (len(relative.parts) == 1 and (path.suffix == '.py' or path.name == 'requirements.txt')):
            result.append(path)
    if not any(p.relative_to(source).parts[0] == 'model' for p in result):
        raise FileNotFoundError(f'No model assets found in {source / "model"}')
    return result
