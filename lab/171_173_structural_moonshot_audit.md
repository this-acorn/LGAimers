# Structural moonshot audit (EXP171–173)

Scope: official `train.csv` plus the workspace's own frozen current-anchor OOF
predictions. No public unlicensed source, weight, prediction, test aggregation,
or LB value was used. Candidate selection stopped at 2023; 2024 outcomes were
not evaluated.

## Fast de-duplication audit

- Test-transductive group calibration, group-frequency features, and
  leave-one-test-group pseudo labels are prohibited by the row-independence
  rule. They cannot be a deployment path.
- `row_id`-based date/game reconstruction is also prohibited. The allowed
  fields contain month and weekday, but no game identifier or exact date.
- Among the 492,997 official 2022–2023 rows, duplicated
  `(season, pitcher_id, asof_pitcher_n)` snapshots: **0**; duplicated
  `(season, batter_id, asof_batter_n)` snapshots: **0**. The same counts are
  also zero after removing `season`. Therefore there is no exact duplicate
  label-transfer path through a player/career-count snapshot.
- Existing records already reject player-ID categorical expansion, direct
  pitcher-batter tables/FM, pitcher×inning/base/stint children, batter hidden
  call fingerprints, recent-window denominator residuals, source-season team
  EB, and multiple current-anchor tensor residuals.

## New strict-forward tests

### EXP171 — explicit venue hierarchy

Deterministically derive the home venue from `top_bottom` and the two team IDs,
then learn a shrunk `(game_type, home_team)` residual plus a
`(game_type, home_team, inning_phase)` child on 2022.

- Best 2023 gain (scale 0.25): **−122.836**
- F/R: **−950.416 / −26.469**
- early/late: **−148.771 / −96.689**
- pitcher-cluster 95% interval: **[−153.548, −96.276]**
- Decision: fail; do not inspect 2024.

### EXP172 — persistent pitcher season state

Learn a game-type-centered per-pitcher current-anchor residual on 2022 and
carry it to 2023 with empirical-Bayes shrinkage. This is the cheapest local
proxy for the unresolved pitcher AR(1) idea.

- Best 2023 gain: **−2.461**
- F/R: **+2.908 / −3.087**
- early/late: **+3.542 / −8.519**
- pitcher-cluster 95% interval: **[−11.553, +6.062]**
- Decision: fail; do not inspect 2024.

### EXP173 — decayed season-boundary form

Freeze each pitcher's final-300-pitch 2022 R residual, shrink it, and decay it
by the current row's own `asof_pitcher_n - frozen_n_end`. F rows are an exact
fallback.

- Best discovery setting: tau 500, scale 0.5
- 2023 gain: **+6.485** (R **+7.243**, F **0**)
- early/late: **+11.548 / +1.378**
- pitcher-cluster 95% interval: **[−4.518, +19.632]**
- Decision: positive but below the +8 screen and CI crosses zero; do not
  inspect 2024 and do not build a submission.

## Conclusion

The remaining legal structural-table corrections are single-digit at best and
cannot plausibly bridge the roughly 35-point gap to 1150. A 1150 attempt needs
a genuinely new standalone endpoint/model family outside the current
Champion–EXP021 plane; structural post-processing does not supply it.
