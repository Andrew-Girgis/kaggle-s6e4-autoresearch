# autoresearch-kaggle

This is an experiment to have an LLM autonomously improve a Kaggle tabular prediction pipeline. It is heavily inspired by Karpathy's `autoresearch`, but adapted for end-to-end data science: data checks, cleaning, feature engineering, feature selection, model training, CV evaluation, ensembling, and submission generation.

The human programs this file. The agent runs the research loop.

## Setup

To set up a new autoresearch run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, such as `apr30` or `apr30-night`.
2. **Confirm branch strategy**: use a branch like `autoresearch/<tag>` if the user wants a dedicated run branch.
3. **Read the in-scope files**:
   - `AGENTS.md` - repo-specific operating constraints.
   - `notes.md` - competition metric, labels, rules, and data notes.
   - `experiments/run_experiment.py` - protected evaluator; do not modify during normal research.
   - `experiments/candidate.py` - editable research surface.
   - `requirements.txt` - installed dependency set.
4. **Verify data exists**:
   - `data/train.csv`
   - `data/test.csv`
   - `data/sample_submission.csv`
5. **Verify dependencies**:

```bash
.venv/bin/python -c "import pandas, sklearn, lightgbm"
```

6. **Run a smoke check**:

```bash
.venv/bin/python experiments/run_experiment.py --smoke-rows 1500 --smoke-test-rows 1000
```

Smoke mode prints `smoke=true`. It does not update `experiments/results.csv` or `experiments/best_config.json`, but it still writes a temporary submission under `experiments/artifacts/`.

7. **Confirm setup looks good**, then begin experimentation.

## Experimentation

The goal is simple: get the highest cross-validated balanced accuracy for `Irrigation_Need`.

Labels must stay exactly:

```text
Low
Medium
High
```

The main experiment command is:

```bash
python experiments/run_experiment.py
```

If the virtual environment is not activated, use:

```bash
.venv/bin/python experiments/run_experiment.py
```

The evaluator uses deterministic 5-fold stratified CV with seed `42`. A real run appends to `experiments/results.csv`, writes a submission under `experiments/artifacts/`, and updates `experiments/best_config.json` plus `experiments/best_submission.csv` only when CV improves.

## What You CAN Do

- Modify `experiments/candidate.py` - this is the only file edited during normal autonomous research.
- Change data cleaning, encoders, feature engineering, feature selection, models, hyperparameters, ensembling, calibration, and thresholding inside `candidate.py`.
- Read `experiments/results.csv`, `experiments/best_config.json`, and prior generated artifacts to decide what to try next.
- Create yardstick snapshots and yardstick commits only at the milestones documented below.

Keep this public API stable:

```python
def get_experiment_config():
    ...

def build_features(train_df, test_df):
    ...

def fit_predict_cv(X, y, X_test, metadata):
    ...
```

## What You CANNOT Do

- Do not modify `experiments/run_experiment.py` during normal research; it is the protected evaluator.
- Do not manually edit `experiments/results.csv`, `experiments/best_config.json`, `experiments/best_submission.csv`, or ordinary files under `experiments/artifacts/`.
- Do not modify, move, commit, or hand-label anything under `data/`.
- Do not manually infer test labels.
- Do not optimize against public leaderboard feedback.
- Do not submit to Kaggle automatically.
- Do not weaken the evaluator or change the CV protocol to make a score look better.
- Do not install new dependencies during an overnight run unless the human explicitly approves.

## Data Science Workflow

Each experiment should test a concrete hypothesis. Good experiment categories include:

- Data cleaning and type handling.
- Missing value strategy, even if missingness initially appears absent.
- Categorical encoding choices.
- Numeric transforms, binning, clipping, ratios, and interactions.
- Feature selection before or after feature engineering.
- Model selection across LightGBM, XGBoost, CatBoost, sklearn models, or ensembles.
- Hyperparameter tuning with deterministic seeds.
- Class balancing, sample weighting, or balanced objectives.
- Probability averaging across folds instead of hard voting.
- OOF-based thresholding or calibration.
- Pseudo-labeling only if confidence thresholds are explicit and the method is CV-safe.

Avoid public leaderboard overfitting and avoid large rewrites unless simpler experiments have plateaued.

## First Run

The first real run should establish the baseline. Do not change `experiments/candidate.py` before the baseline unless the baseline is broken.

After the baseline run finishes, create the `baseline` yardstick snapshot and commit it as described below.

## Output Format

The evaluator prints one final machine-readable line:

```text
FINAL_METRIC balanced_accuracy=<score> improved=<true|false> experiment_id=<id> smoke=<true|false>
```

Use `balanced_accuracy` as the primary metric. Higher is better.

## Logging Results

Do not manually edit the experiment ledger. The evaluator owns these files:

```text
experiments/results.csv
experiments/best_config.json
experiments/best_submission.csv
experiments/artifacts/
```

Use `experiments/results.csv` as the run ledger. Use `experiments/best_config.json` to identify the current best score, experiment id, candidate hash, and source submission path.

## Yardstick Snapshots

Yardsticks are permanent progress markers. Preserve both the file snapshot and a Git commit at these milestones:

```text
baseline
run_10
run_100
run_1000
```

Do not commit every experiment. The only automatic commits allowed during autonomous research are yardstick commits at `baseline`, `run_10`, `run_100`, and `run_1000`.

At each yardstick milestone, create:

```text
experiments/yardsticks/<milestone>/candidate.py
experiments/yardsticks/<milestone>/config.json
experiments/yardsticks/<milestone>/submission.csv
experiments/yardsticks/<milestone>/summary.md
```

Snapshot contents:

- `candidate.py` - exact current `experiments/candidate.py` at the milestone.
- `config.json` - copy of `experiments/best_config.json` if it exists; otherwise write a small JSON file explaining that no best config exists yet.
- `submission.csv` - copy of `experiments/best_submission.csv` if it exists; otherwise omit it and note that in `summary.md`.
- `summary.md` - short human-readable summary with milestone name, timestamp, run count, current CV score, best CV score, experiment id, candidate notes, and whether this milestone is the best known candidate.

Recommended commands for a milestone:

```bash
mkdir -p experiments/yardsticks/<milestone>
cp experiments/candidate.py experiments/yardsticks/<milestone>/candidate.py
cp experiments/best_config.json experiments/yardsticks/<milestone>/config.json
cp experiments/best_submission.csv experiments/yardsticks/<milestone>/submission.csv
```

Then write `experiments/yardsticks/<milestone>/summary.md` manually with concise facts from the latest `FINAL_METRIC`, `experiments/results.csv`, and `experiments/best_config.json`.

Commit only the yardstick snapshot and the current candidate:

```bash
git add experiments/candidate.py experiments/yardsticks/<milestone>
git commit -m "yardstick: <milestone> autoresearch run"
```

Use these commit messages:

```text
yardstick: baseline autoresearch run
yardstick: autoresearch run 10
yardstick: autoresearch run 100
yardstick: autoresearch run 1000
```

Do not commit ordinary generated artifacts, `experiments/results.csv`, or every experimental attempt.

## Experiment Loop

The experiment runs on the current branch, ideally a dedicated branch such as `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the current Git state and current run count in `experiments/results.csv`.
2. Inspect `experiments/best_config.json` if it exists.
3. Form one concrete data science hypothesis.
4. Modify `experiments/candidate.py` directly.
5. Update `get_experiment_config()["name"]` and `get_experiment_config()["notes"]` to describe the experiment.
6. Run the experiment with `python experiments/run_experiment.py`.
7. Read the final `FINAL_METRIC` line.
8. If the run crashed, fix obvious bugs such as typos, missing imports, or shape errors and rerun; if the idea is fundamentally broken, discard the idea and move on.
9. If the score improved, keep the candidate and continue from it.
10. If the score is worse, decide whether it is useful foundation for the next experiment; otherwise restore the last better `candidate.py` state.
11. If the completed run hits `baseline`, `run_10`, `run_100`, or `run_1000`, create the yardstick snapshot and Git commit before continuing.
12. Repeat until manually interrupted.

## Simplicity Criterion

All else being equal, simpler is better. A small improvement that adds fragile complexity may not be worth keeping. A small improvement from deleting code is valuable. An equal score with a simpler, more robust pipeline is also a good result.

Prefer reproducible CV gains over lucky one-off changes. Do not add complexity without a measurable reason.

## Timeout And Failures

Use judgment for runtime. Full runs on 630k rows can take longer than Karpathy's fixed 5-minute LLM runs, but an experiment should still be bounded. If a run is clearly hung or far slower than comparable previous runs, kill it, treat it as a failed idea, and move on.

For crashes:

- Fix simple implementation mistakes and rerun.
- Do not keep retrying a fundamentally broken idea.
- Do not modify the protected evaluator to accommodate a candidate bug.

## Notebook Policy

The notebook is not the autonomous experiment runner. The script-based evaluator is the source of truth for comparable experiments.

Use notebooks for EDA, explanation, or final reporting only. If notebook execution is needed later, install notebook execution tooling only with human approval and run it separately from the core loop, for example with `nbconvert` or `papermill`.

## NEVER STOP

Once the experiment loop has begun, do not pause to ask the human whether to continue. Do not ask if this is a good stopping point. The human may be asleep and expects autonomous progress.

If ideas run low, inspect the results ledger, reread the in-scope files, compare strong and weak experiments, try simplifications, try feature interactions, try different model families, try ensembles, or revisit near misses. The loop runs until the human interrupts it.
