# AGENTS.md

## Repo Purpose

- This is a Kaggle `playground-series-s6e4` workspace for predicting `Irrigation_Need` (`Low`, `Medium`, `High`).
- The competition metric is balanced accuracy; higher is better.
- `program.md` is the autoresearch-style operating manual for overnight experiment agents.

## Autoresearch Loop

- The protected evaluator is `experiments/run_experiment.py`; do not edit it during normal research runs.
- The editable research surface is `experiments/candidate.py`; keep `get_experiment_config()`, `build_features()`, and `fit_predict_cv()` stable.
- Main verification command after candidate changes is `python experiments/run_experiment.py` from an activated `.venv`.
- Equivalent explicit command is `.venv/bin/python experiments/run_experiment.py`.
- Fast harness check: `.venv/bin/python experiments/run_experiment.py --smoke-rows 1500 --smoke-test-rows 1000`.
- Smoke mode prints `smoke=true` and does not update `results.csv` or `best_config.json`, but it still writes a temporary submission under `experiments/artifacts/`.
- The evaluator reserves a fixed 20% stratified holdout from `train.csv`, runs 5-fold CV on the remaining dev split, selects by dev CV balanced accuracy, and reports holdout balanced accuracy as a guardrail.

## Data And Outputs

- Source data lives in ignored `data/`; do not modify, move, commit, or hand-label it.
- Real runs append `experiments/results.csv`, write submissions under `experiments/artifacts/`, and update `experiments/best_config.json` plus `experiments/best_submission.csv` only on CV improvement.
- These run outputs are ignored by Git; inspect them when researching but do not manually edit them.
- Submission format is `id,Irrigation_Need`; labels must stay exactly `Low`, `Medium`, `High`.
- Kaggle `data/test.csv` is only for final prediction generation, not fitting, scoring, feature selection, threshold tuning, calibration, or experiment decisions.

## Environment

- Dependencies are listed in `requirements.txt`: pandas, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, joblib.
- The existing `.venv` is ignored; install with `.venv/bin/pip install -r requirements.txt` if imports fail.
- The default system `python` may not have the ML stack, so prefer activating `.venv` before running project commands.

## Research Constraints

- Do not optimize manually against the public leaderboard or submit automatically.
- Use deterministic seeds where possible; the evaluator uses 5-fold `StratifiedKFold` with seed `42`.
- Update `get_experiment_config()["name"]` and `["notes"]` for each meaningful candidate experiment.
- If an experiment fails, fix only the latest `candidate.py` change or restore the last working candidate; do not weaken the evaluator.
- `candidate.py` must not read experiment logs, best configs, prior submissions, generated artifacts, or holdout labels to choose features, models, thresholds, or predictions.
- Final Kaggle submission must be a notebook, so keep candidate code portable: no local absolute paths, hidden state, or artifact dependencies.
