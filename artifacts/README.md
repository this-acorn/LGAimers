# Model artifacts

[manifest.json](manifest.json) lists inference package files, their sizes, and
local weight/archive locations. `last.zip!/…` denotes an extracted member.

- `artifacts/candidates/`: candidate ZIPs, including the 70% variants and `last.zip`.
- `artifacts/probes/`: calibration and blend probe ZIPs.
- `artifacts/submissions/`: earlier submission ZIPs.
- `candidates/*/model/` and `probes/*/model/`: fitted weights beside inference code.

The included `submit14` checkpoint runs after installing its requirements.
Late-stage packages additionally require their local model assets. To fill
missing assets from an original or organized local workspace:

```bash
python tools/restore_artifacts.py --from-dir /path/to/local/workspace
```

Existing files are preserved. Missing assets are reported; this command does
not download weights. For `candidates/last`, it can read a local `last.zip`.
Install the selected package's requirements before running inference.
