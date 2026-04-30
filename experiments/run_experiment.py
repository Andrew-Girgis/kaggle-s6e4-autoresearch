"""Stable evaluator for the Kaggle autoresearch loop.

The overnight agent should not edit this file. It should make research changes in
experiments/candidate.py, then run this script to get one comparable score.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"
DATA_DIR = ROOT / "data"
TARGET = "Irrigation_Need"
ID_COLUMN = "id"
CLASS_ORDER = ["Low", "Medium", "High"]
N_SPLITS = 5
CV_SEED = 42
RESULTS_PATH = EXPERIMENTS_DIR / "results.csv"
BEST_CONFIG_PATH = EXPERIMENTS_DIR / "best_config.json"
BEST_SUBMISSION_PATH = EXPERIMENTS_DIR / "best_submission.csv"
HIGHER_IS_BETTER = True


def _load_candidate() -> Any:
    candidate_path = EXPERIMENTS_DIR / "candidate.py"
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load candidate module from {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _validate_labels(name: str, labels: Any, expected_len: int) -> np.ndarray:
    values = np.asarray(labels)
    if len(values) != expected_len:
        raise ValueError(f"{name} has length {len(values)}, expected {expected_len}")
    unique = set(pd.Series(values).dropna().astype(str).unique())
    allowed = set(CLASS_ORDER)
    unknown = sorted(unique - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown labels: {unknown}; allowed={CLASS_ORDER}")
    return values.astype(str)


def _read_best_score() -> float | None:
    if not BEST_CONFIG_PATH.exists():
        return None
    with BEST_CONFIG_PATH.open() as f:
        data = json.load(f)
    score = data.get("balanced_accuracy")
    return None if score is None else float(score)


def _is_improvement(score: float, best_score: float | None) -> bool:
    if best_score is None:
        return True
    return score > best_score if HIGHER_IS_BETTER else score < best_score


def _append_result(row: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "timestamp_utc",
        "name",
        "balanced_accuracy",
        "improved",
        "duration_seconds",
        "seed",
        "notes",
        "candidate_sha256",
        "submission_path",
        "model_metadata_json",
    ]
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Kaggle autoresearch experiment.")
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="Use a deterministic subset of training rows for a fast harness smoke test.",
    )
    parser.add_argument(
        "--smoke-test-rows",
        type=int,
        default=0,
        help="Use a deterministic subset of test rows for a fast harness smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    smoke_mode = bool(args.smoke_rows or args.smoke_test_rows)
    start = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_id = timestamp
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if args.smoke_rows:
        train_df = train_df.groupby(TARGET, group_keys=False).sample(
            n=max(1, args.smoke_rows // len(CLASS_ORDER)), random_state=CV_SEED
        )
        train_df = train_df.sample(frac=1.0, random_state=CV_SEED).reset_index(drop=True)
    if args.smoke_test_rows:
        test_df = test_df.head(args.smoke_test_rows).copy()
        sample_submission = sample_submission.head(args.smoke_test_rows).copy()

    if TARGET not in train_df.columns:
        raise ValueError(f"Missing target column {TARGET!r} in train.csv")
    if ID_COLUMN not in test_df.columns or ID_COLUMN not in sample_submission.columns:
        raise ValueError("Missing id column in test.csv or sample_submission.csv")

    candidate = _load_candidate()
    config = dict(candidate.get_experiment_config())
    candidate_sha = _file_sha256(EXPERIMENTS_DIR / "candidate.py")

    y = train_df[TARGET].astype(str)
    unknown_targets = sorted(set(y.unique()) - set(CLASS_ORDER))
    if unknown_targets:
        raise ValueError(f"Unknown target labels in training data: {unknown_targets}")

    X, X_test, returned_y, metadata = candidate.build_features(train_df.copy(), test_df.copy())
    if returned_y is not None:
        y = pd.Series(returned_y).astype(str)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    metadata = dict(metadata or {})
    metadata.update(
        {
            "target": TARGET,
            "id_column": ID_COLUMN,
            "class_order": CLASS_ORDER,
            "cv": cv,
            "cv_splits": list(cv.split(X, y)),
            "n_splits": N_SPLITS,
            "cv_seed": CV_SEED,
            "artifacts_dir": ARTIFACTS_DIR,
            "experiment_id": experiment_id,
        }
    )

    oof_pred, test_pred, model_metadata = candidate.fit_predict_cv(X, y, X_test, metadata)
    oof_pred = _validate_labels("oof_pred", oof_pred, len(train_df))
    test_pred = _validate_labels("test_pred", test_pred, len(test_df))

    score = float(balanced_accuracy_score(y, oof_pred))

    submission = sample_submission.copy()
    submission[TARGET] = test_pred
    submission_path = ARTIFACTS_DIR / f"submission_{experiment_id}.csv"
    submission.to_csv(submission_path, index=False)

    best_score = _read_best_score()
    improved = _is_improvement(score, best_score)
    duration = time.time() - start

    model_metadata_json = json.dumps(model_metadata or {}, sort_keys=True, default=_json_default)
    row = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name", "unnamed"),
        "balanced_accuracy": f"{score:.10f}",
        "improved": str(improved).lower(),
        "duration_seconds": f"{duration:.2f}",
        "seed": config.get("seed", ""),
        "notes": config.get("notes", ""),
        "candidate_sha256": candidate_sha,
        "submission_path": str(submission_path.relative_to(ROOT)),
        "model_metadata_json": model_metadata_json,
    }
    if not smoke_mode:
        _append_result(row)

    if improved and not smoke_mode:
        shutil.copyfile(submission_path, BEST_SUBMISSION_PATH)
        best_config = {
            "experiment_id": experiment_id,
            "timestamp_utc": row["timestamp_utc"],
            "balanced_accuracy": score,
            "previous_best_balanced_accuracy": best_score,
            "name": row["name"],
            "notes": row["notes"],
            "seed": row["seed"],
            "candidate_sha256": candidate_sha,
            "submission_path": str(BEST_SUBMISSION_PATH.relative_to(ROOT)),
            "source_submission_path": str(submission_path.relative_to(ROOT)),
            "model_metadata": model_metadata or {},
        }
        with BEST_CONFIG_PATH.open("w") as f:
            json.dump(best_config, f, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")

    print(
        "FINAL_METRIC "
        f"balanced_accuracy={score:.10f} "
        f"improved={str(improved).lower()} "
        f"experiment_id={experiment_id} "
        f"smoke={str(smoke_mode).lower()}"
    )


if __name__ == "__main__":
    main()
