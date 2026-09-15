# INTERNAL ONLY — repository audit (do not commit/share)

This note names a public reference repository and must not be copied into team-facing
documents. Team-facing ledgers should describe only the generic experiment and result.

## Scope and evidence quality

- Audited repository: `iamdbstjd/LGAIMERS`, branch `main`.
- Audited commit: `ad8ac70c2ce4b31c0718a6c1e2a0f825a54ba1dc`.
- Repository history contains one commit.
- `model_test_2.md` explicitly mixes file-confirmed results, user-reported results, and
  follow-up diagnostics. These are not equivalent evidence classes.
- The only recorded Public-LB submissions are E12 Primary `727.7193290843` and Stable
  `686.9897560676`. E13–E18 and the E17 scale-3 package have no recorded Public-LB score.
- Therefore the current `1092.808353586` champion must not be replaced by this backbone.

## Full experiment classification

| Family | What it tests | Relation to our work | Verdict |
| --- | --- | --- | --- |
| E0 | constant/prevalence baseline | diagnostic only | no candidate |
| E1 | count/base-out/inning/team/game-type logistic structure | already covered by CAT5/context and failed interaction probes | closed |
| E2 | pitcher career history, reliability and pitch mix | covered more richly; direct pitch-mix additions failed | closed |
| E3 | recent 1/3/5-game state and trend | recent/workload variants already failed strict transfer | closed |
| E4 | F/R game-type regime | explicit in our model and V18; segment corrections already audited | closed |
| E5 | empirical-Bayes smoothing and cold start | our as-of/CS and rookie/no-ID audits are richer | closed |
| E6 | batter history | weaker than our batter state; worsened there | closed |
| E7 | global gamma/compression calibration | superseded by our LB-validated affine family | closed |
| E8–E9 | HGB and calibrated blending | weaker, highly collinear model family | closed |
| E10 | CatBoost replacement | failed in that repository; our CatBoost is much stronger | closed |
| E11 | LightGBM replacement | our exact 79-feature target5 LGB gate was `-201.30` | closed |
| E12 | three-model blend | only Public-validated model there, score 727.72 | no replacement/blend |
| E13 | recent-window/season weighting | tiny local result there; our season weighting was LB-negative | closed |
| E14 | manual E5/E8/LGB blend | weaker predecessor of later chain | closed |
| E15-A | exact current-season state | mostly overlaps CS79; exact raw component level remains narrowly untested | one cheap screen only |
| E15-B | previous-season residual maps by batter team and pitcher×hand | genuinely distinct time-safe target construction | killed on our predictions |
| E16-H1/M0/P1 | hierarchical hazards and component innovations | target5/reverse-head/M4/LGB tests already give stronger negative evidence | closed |
| E17 | centered log component ratios -> residual correction | genuinely distinct cheap overlay | killed on our predictions |
| E18-T0 | time-safe TrackMan/player crosswalk | mapping technique is sound but is not predictive gain | no reopening |
| E18-T1 | 19 physical TrackMan mean/std features | fails temporal transfer in repository and overlaps our failed audits | closed |
| E18-T2 | F/R repertoire-context specialists | worsens 2024 and fails bootstrap in repository | closed |
| R specialist | reverse-side specialist chain | plans/code/smoke, not a completed performance result | do not spend budget |
| R_FC_C | three binary component heads + ridge on R | clean protocol but no full result; adjacent heads already fail | do not spend budget |
| M0 ablation | direct/sidecar multiclass ablation | cross-year sign flip or neutral | closed |

## Exact transfer checks on our champion space

### E15-B-style residual maps (`exp/124`)

Baseline was the exact two-seed CAT5 prediction plus frozen V18 gamma `0.30`.
For each source season, residuals were centered within game type, then mapped to the
next season. No test data, deployment bundle, submission, or LB was used.

Robust (`batter_team_id alpha=10000`, `pitcher_id×hand alpha=1000`):

- 2022 -> 2023: raw `+23.851`, equal-mean shape `+22.604`.
- 2023 -> 2024: raw `-46.425`, equal-mean shape `-41.899`.
- Mean raw/shape: `-11.287/-9.648`; gate FAIL.
- Team-only and hand-only are both negative on 2023 -> 2024 (`-30.627/-19.698`).

Clean (`alpha=50/500`):

- 2022 -> 2023: raw `+22.214`, shape `+20.254`.
- 2023 -> 2024: raw `-106.470`, shape `-104.731`.
- Gate FAIL.

An independent implementation produced slightly different magnitudes but the same
strong next-season reversal. `exp/104` historical arrays have a known same-hand dtype
caveat; this does not rescue a direction whose team-only term also reverses sharply.

### E17-style component log-ratio overlay

Mapped our CAT5 probabilities as `R=p2+p3`, `M=p1`, `O=p4`, `S=p0`, with source-centered
`log(M/R)` and `log(O/R)` and game-type-centered source residuals.

- Refit strict transition, both ratios, scale 1: `+35.43` on 2022 -> 2023 and `+8.47`
  on 2023 -> 2024, but the latter equal-mean shape is `-1.39`.
- The two fitted coefficients reverse signs between transitions.
- Scale 3 on 2023 -> 2024 is `-59.51`.
- O/R only: `+61.34/+2.25`; 2024 shape only `+1.49`.
- M/R only: `+46.06/+0.37`; 2024 shape `-2.20`.
- Direct application of the public frozen coefficients is likewise only `+5.62`
  (shape `+3.45`) on 2023 -> 2024; scale 3 remains too small/shape-poor.

This is not worth a submission slot or a training run.

## E15-A: the only narrow survivor

The external 17-feature block is:

- `since_cutoff_n`;
- eight exact current-season raw rates: success, reverse, middle, ball, strike,
  fastball, breaking, offspeed;
- the same eight raw-minus-career deltas.

Our CS79 already contains current-season sample size, smoothed/raw success and several
success/ball/strike/middle/pitch-mix deltas for pitcher and batter. Direct reverse and
breaking delta extensions failed in exp/72, and a larger delta block failed in exp/51.
Those failures do not strictly kill the *raw component level* representation.

The only genuinely untested pieces are therefore:

1. exact pre-pitch raw reverse and middle levels (success is a useful control);
2. optionally raw ball/strike/fastball/breaking levels;
3. `exact_raw - current_approx_raw`, which isolates the one-pitch cutoff discrepancy;
4. a reliability gate by `since_cutoff_n` (their own result worsened for `n<=10`).

Offspeed is redundant when fastball + breaking + offspeed = 1 and should be omitted.
Wholesale copying of all 17 features would add duplication and selection risk.

Recommended gate before any CatBoost training:

- construct exact raw success/reverse/middle plus cutoff-difference terms;
- fit/freeze a simple ridge residual screen on 2022 -> 2023, then evaluate unchanged
  on 2023 -> 2024;
- require both transitions to have raw and equal-mean shape gain > 0, mean gain at
  least `+12` to `+15`, and F/R contributions nonnegative;
- only if it passes, train one seed-42 CatBoost arm with the 2–3 surviving raw features;
- require seed-42 raw `>=+15` and shape `>=+10` before seed 7.

Expected value is small (`0–15`, optimistic upside roughly `20–30`), not a `+90` route.

## Reproducibility and deployment audit

- `.gitignore` excludes CSV/CSV.GZ, joblib, zip, and all `precomputed/` directories.
- The E17 full-chain README says `precomputed/` is included, but it is absent from the
  public repository.
- `build_champion_submit.py` directly requires three missing precomputed artifacts,
  so the latest package cannot be reproduced from the public clone alone.
- The submitted-script pattern itself appears row-independent: it uses train-derived
  frozen cutoffs/maps/coefficients, restores row order, and does not aggregate test rows.
- The principal risk is selection overfit, not obvious test leakage: E17 scale 3 is
  explicitly a 2024 follow-up amplitude diagnostic (2024 optimum about 3.6), not a
  locked transition-selected deployment value.

## Operational conclusion

Keep `candidate_v18g030.zip` as champion. Do not copy or submit their package, replace
our backbone, reopen TrackMan, or train their specialist stacks. Preserve only their
strict source-season -> next-season validation discipline. Run at most the cheap exact
current-season raw-state screen above after confirming no CPU collision with the other
session.
