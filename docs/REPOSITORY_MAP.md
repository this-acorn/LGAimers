# Repository map

The layout includes the complete inspected local project workspace, including code that had never been tracked on GitHub.

| Directory | Role |
| --- | --- |
| [exp](../exp/) | Numbered experiments and shared training/validation utilities. |
| [submissions](../submissions/) | Earlier standalone inference packages and preserved checkpoints. |
| [candidates](../candidates/) | Later inference variants, including the 70% and last packages. |
| [probes](../probes/) | Nine calibration and blend probe packages. |
| [handoffs](../handoffs/) | GPU and team experiment handoffs. |
| [lab](../lab/) | Lightweight reports; predictions and fitted artifacts stay local. |
| [tools](../tools/) | Inference, packaging, local artifact restoration, and remote GPU utilities. |
| [artifacts](../artifacts/) | Local archive locations and the complete file relocation inventory. |
| [docs/archive](archive/) | Historical working notes and experiment ledgers. |
| [archive](../archive/) | Earlier automation, rejected work, and local scratch builds. |

The [manifest](../artifacts/manifest.json) maps old paths to new paths. Data, virtual environments, reference material, nested Git metadata, and caches stay local. Historical handoff and archived scripts preserve their original environment assumptions; current experiment path references follow this layout.
