# Baseline Yardstick

- Milestone: `baseline`
- Experiment ID: `20260430_041315`
- Run count: `1`
- Candidate: `baseline_lgbm_or_histgb`
- Dev CV balanced accuracy: `0.9606334176`
- Holdout balanced accuracy: `0.9638706794`
- Runtime seconds: `434.17`
- Improved: `true`
- Candidate SHA256 prefix: `749de48b1301bcd6`
- Source submission: `experiments/artifacts/submission_20260430_041315.csv`
- Best submission snapshot: `experiments/yardsticks/baseline/submission.csv`

## Notes

Strong boring baseline using ordinal categorical handling with LightGBM. The holdout score is slightly above dev CV, so this baseline does not show an obvious overfitting warning.
