# Run 10 Yardstick

- Milestone: `run_10`
- Experiment ID: `20260430_051216`
- Run count: `10`
- Candidate: `lgbm_balanced_tiny_trees`
- Dev CV balanced accuracy: `0.9701541826`
- Holdout balanced accuracy: `0.9724157250`
- Runtime seconds: see `experiments/results.csv`
- Improved: `true`
- Candidate SHA256 prefix: see `experiments/yardsticks/run_10/config.json`
- Best submission snapshot: `experiments/yardsticks/run_10/submission.csv`

## Notes

Best candidate after 10 runs. The largest gains came from adding `class_weight='balanced'` and progressively reducing LightGBM tree complexity. Current setting uses tiny class-weighted trees with `num_leaves=8` and `min_child_samples=500`, improving both dev CV and holdout over the baseline.
