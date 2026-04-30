# Kaggle S6E4 Autoresearch

Autoresearch-style workspace for Kaggle Playground Series Season 6 Episode 4. The task is to predict `Irrigation_Need` (`Low`, `Medium`, `High`) from tabular irrigation data, optimizing balanced accuracy.

This repo is inspired by Karpathy's `autoresearch`, but adapted for Kaggle tabular modeling: agents make controlled changes to one candidate file, run a stable evaluator, and preserve milestone yardsticks.

## What This Repo Contains

- `program.md` - the autonomous research operating manual.
- `AGENTS.md` - compact repo-specific guidance for future OpenCode sessions.
- `experiments/run_experiment.py` - protected evaluator and logging harness.
- `experiments/candidate.py` - editable experiment surface for feature/model changes.
- `requirements.txt` - Python ML dependencies.
- `notes.md` - competition notes, metric, rules, and source context.
- `playground-series-s6e4.ipynb` - lightweight EDA/reporting notebook starter.

The Kaggle data is intentionally not tracked. Put competition files under `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

## Setup

Create or reuse a virtual environment, then install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

Verify the core stack:

```bash
.venv/bin/python -c "import pandas, sklearn, lightgbm"
```

## Running Experiments

Run one full CV experiment:

```bash
.venv/bin/python experiments/run_experiment.py
```

Run a quick harness smoke test:

```bash
.venv/bin/python experiments/run_experiment.py --smoke-rows 1500 --smoke-test-rows 1000
```

The evaluator prints a final line like:

```text
FINAL_METRIC balanced_accuracy=0.9406666667 holdout_balanced_accuracy=0.9380000000 improved=true experiment_id=20260430_033027 smoke=true
```

The evaluator uses only `data/train.csv` for model selection: a fixed 20% stratified internal holdout is reserved, 5-fold CV runs on the remaining dev split, and the holdout score is reported as an anti-overfitting guardrail. Kaggle `data/test.csv` is used only after evaluation to generate a submission file.

## Autoresearch Workflow

Normal autonomous research should edit only:

```text
experiments/candidate.py
```

The evaluator owns generated outputs:

```text
experiments/results.csv
experiments/best_config.json
experiments/best_submission.csv
experiments/artifacts/
```

Those generated outputs are ignored by Git. Yardstick snapshots under `experiments/yardsticks/` are intentionally trackable for milestone commits (`baseline`, `run_10`, `run_100`, `run_1000`).

See `program.md` for the full overnight loop, protected-file rules, and yardstick commit procedure.

## Notebook Submission

The autonomous loop is script-first for reliable diffs and repeatable scoring, but the final Kaggle submission should be packaged as a notebook. Keep `experiments/candidate.py` portable so the best pipeline can be moved into a notebook cleanly: no local absolute paths, no dependency on ignored artifacts, and no reads from experiment logs or prior submissions.

## Safety Notes

- Do not commit Kaggle data or credentials.
- Do not hand-label test data or infer test labels manually.
- Do not optimize manually against public leaderboard feedback.
- Do not submit to Kaggle automatically.
- Do not use Kaggle test rows for training, scoring, feature selection, threshold tuning, calibration, or experiment decisions.
- Keep `experiments/run_experiment.py` stable during normal research so scores stay comparable.
