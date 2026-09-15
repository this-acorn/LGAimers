# Probes

Standalone inference variants used to study fixed calibration and ensemble weights.

| Family | Source packages | Purpose |
| --- | --- | --- |
| Affine scale | [0.96](probe_affine_s096_src/), [1.04](probe_affine_s104_src/) | Change probability spread around the fixed center. |
| Probability shift | [negative shift](probe_shift_m0030_src/) | Test a fixed probability offset. |
| Correction scale | [0.85](probe_corr085_src/), [1.15](probe_corr115_src/) | Vary correction strength. |
| Hierarchical correction | [gamma 0.18](probe_v18g018_src/), [gamma 0.42](probe_v18g042_src/) | Change the hierarchical residual contribution. |
| Endpoint mixtures | [JOA 50%](probe_affine_joa_w050_src/), [JY 50%](probe_affine_jy_w050_src/) | Measure complementary endpoint behavior. |

Each folder preserves its original runtime layout. Code and small configuration files are versioned; fitted weights and probe ZIPs remain local. See the [artifact guide](../artifacts/README.md).
