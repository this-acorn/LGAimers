# Local artifacts

[manifest.json](manifest.json) records each inspected local file, its original path, its organized destination, size, and whether it is versioned. `last.zip!/…` denotes a member extracted from that archive. Dataset roots, virtual environments, caches, and linked directories are listed separately.

| Location | Contents |
| --- | --- |
| `artifacts/candidates/` | Candidate ZIPs, including the full/compact 70% variants and `last.zip`. |
| `artifacts/probes/` | Probe ZIPs. |
| `artifacts/handoffs/` | Team handoff ZIPs. |
| `artifacts/submissions/` | Earlier submission ZIPs. |
| `candidates/*/model/`, `probes/*/model/` | Fitted weights and runtime state beside versioned inference code. |
| `lab/` | Reports plus local prediction arrays and fitted experimental artifacts. |
| `archive/scratch/` | Temporary builds, rehearsal workspaces, and endpoint expansions. |

Large generated artifacts and dataset copies remain local; previously versioned checkpoint files remain available. A fresh clone therefore runs the included `submit14` example, while late-stage packages also require their corresponding local assets. The manifest is an inventory, not a download service.

To restore assets from a copy of the original or organized local workspace:

```bash
python tools/restore_artifacts.py --from-dir /path/to/local/workspace
```

The command fills missing files only and reports unavailable assets. For `candidates/last`, it can extract the required members from a local `last.zip`. Install each selected package's own requirements before running it. Local ignore rules are intentionally not tracked in Git.
