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
FINAL_METRIC balanced_accuracy=0.9406666667 improved=true experiment_id=20260430_033027 smoke=true
```

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

## Safety Notes

- Do not commit Kaggle data or credentials.
- Do not hand-label test data or infer test labels manually.
- Do not optimize manually against public leaderboard feedback.
- Do not submit to Kaggle automatically.
- Keep `experiments/run_experiment.py` stable during normal research so scores stay comparable.
