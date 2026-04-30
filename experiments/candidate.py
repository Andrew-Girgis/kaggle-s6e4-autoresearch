"""Editable research candidate for the Kaggle autoresearch loop.

This is the only file the overnight agent should modify. Keep the public API
stable: get_experiment_config(), build_features(), and fit_predict_cv().
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def get_experiment_config() -> dict:
    return {
        "name": "baseline_lgbm_or_histgb",
        "notes": "Strong boring baseline: ordinal categorical encoding with LightGBM if available, otherwise sklearn HistGradientBoosting.",
        "seed": 42,
    }


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y = train_df["Irrigation_Need"].astype(str)
    X = train_df.drop(columns=["Irrigation_Need", "id"])
    X_test = test_df.drop(columns=["id"])

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    metadata = {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }
    return X, X_test, y, metadata


def _majority_vote(predictions: list[np.ndarray], class_order: list[str]) -> np.ndarray:
    stacked = np.vstack(predictions).T
    output = []
    class_rank = {label: idx for idx, label in enumerate(class_order)}
    for row in stacked:
        counts = Counter(row)
        output.append(max(counts, key=lambda label: (counts[label], -class_rank[label])))
    return np.asarray(output, dtype=object)


def _fit_predict_lightgbm(X, y, X_test, metadata):
    from lightgbm import LGBMClassifier

    categorical_columns = metadata["categorical_columns"]
    class_order = metadata["class_order"]
    cv_splits = metadata["cv_splits"]
    seed = get_experiment_config()["seed"]

    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    for col in categorical_columns:
        categories = pd.Index(pd.concat([X_lgb[col], X_test_lgb[col]], ignore_index=True).astype(str).unique())
        X_lgb[col] = pd.Categorical(X_lgb[col].astype(str), categories=categories)
        X_test_lgb[col] = pd.Categorical(X_test_lgb[col].astype(str), categories=categories)

    oof_pred = np.empty(len(X_lgb), dtype=object)
    test_fold_preds = []
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv_splits):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(class_order),
            n_estimators=800,
            learning_rate=0.045,
            num_leaves=96,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.2,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_lgb.iloc[train_idx],
            y.iloc[train_idx],
            categorical_feature=categorical_columns,
        )
        valid_pred = model.predict(X_lgb.iloc[valid_idx])
        oof_pred[valid_idx] = valid_pred
        test_fold_preds.append(model.predict(X_test_lgb))

        from sklearn.metrics import balanced_accuracy_score

        fold_scores.append(float(balanced_accuracy_score(y.iloc[valid_idx], valid_pred)))

    test_pred = _majority_vote(test_fold_preds, class_order)
    return oof_pred, test_pred, {"model": "lightgbm", "fold_scores": fold_scores}


def _fit_predict_histgb(X, y, X_test, metadata):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    categorical_columns = metadata["categorical_columns"]
    numeric_columns = metadata["numeric_columns"]
    class_order = metadata["class_order"]
    cv_splits = metadata["cv_splits"]
    seed = get_experiment_config()["seed"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            ),
            ("num", "passthrough", numeric_columns),
        ]
    )

    oof_pred = np.empty(len(X), dtype=object)
    test_fold_preds = []
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv_splits):
        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=300,
                        max_leaf_nodes=63,
                        l2_regularization=0.05,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        valid_pred = model.predict(X.iloc[valid_idx])
        oof_pred[valid_idx] = valid_pred
        test_fold_preds.append(model.predict(X_test))
        fold_scores.append(float(balanced_accuracy_score(y.iloc[valid_idx], valid_pred)))

    test_pred = _majority_vote(test_fold_preds, class_order)
    return oof_pred, test_pred, {"model": "hist_gradient_boosting", "fold_scores": fold_scores}


def fit_predict_cv(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, metadata: dict):
    try:
        return _fit_predict_lightgbm(X, y, X_test, metadata)
    except ImportError:
        return _fit_predict_histgb(X, y, X_test, metadata)
