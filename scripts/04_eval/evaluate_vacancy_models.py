# -*- coding: utf-8 -*-
"""
Evaluate two vacancy-risk models from dataset_vacancy.csv.

Model 1: classify whether a listing is high vacancy risk.
Model 2: estimate vacancy rate as a continuous value.

Target definition:
  Y_vacancy = availability_365 / 365
  high_risk = Y_vacancy > 0.70
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "../../dataset_vacancy.csv"
RESULT_PATH = "../../model_evaluation_results.json"
HIGH_RISK_THRESHOLD = 0.70
RANDOM_STATE = 42


def round_float(value, digits=4):
    return None if value is None else round(float(value), digits)


def regression_metrics(y_true, pred):
    return {
        "mae": round_float(mean_absolute_error(y_true, pred)),
        "mae_percentage_points": round_float(mean_absolute_error(y_true, pred) * 100, 2),
        "rmse": round_float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": round_float(r2_score(y_true, pred)),
    }


def classification_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": round_float(threshold, 2),
        "roc_auc": round_float(roc_auc_score(y_true, prob)),
        "pr_auc": round_float(average_precision_score(y_true, prob)),
        "brier": round_float(brier_score_loss(y_true, prob)),
        "recall": round_float(recall_score(y_true, pred, zero_division=0)),
        "precision": round_float(precision_score(y_true, pred, zero_division=0)),
        "f1": round_float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred).ravel().astype(int).tolist(),
    }


def select_threshold_for_recall(y_true, prob, min_recall=0.80):
    best = None
    for threshold in np.round(np.arange(0.05, 0.95, 0.01), 2):
        pred = (prob >= threshold).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        if recall >= min_recall and (best is None or precision > best["precision"]):
            best = {"threshold": float(threshold), "recall": float(recall), "precision": float(precision)}
    if best is None:
        best = {"threshold": 0.50, "recall": None, "precision": None}
    return best


def main():
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    df = pd.read_csv(DATA_PATH)
    y_reg = df["Y_vacancy"].astype(float).to_numpy()
    y_cls = (df["Y_vacancy"].astype(float) > HIGH_RISK_THRESHOLD).astype(int).to_numpy()
    X = df.drop(columns=["listing_id", "Y_vacancy"])
    feature_names = X.columns.tolist()

    results = {
        "source_dataset": DATA_PATH,
        "row_count": int(len(df)),
        "feature_count": int(len(feature_names)),
        "target": {
            "vacancy_definition": "availability_365 / 365",
            "high_risk_definition": f"Y_vacancy > {HIGH_RISK_THRESHOLD:.2f}",
            "vacancy_mean": round_float(df["Y_vacancy"].mean()),
            "vacancy_median": round_float(df["Y_vacancy"].median()),
            "high_risk_rate": round_float(y_cls.mean()),
        },
    }

    # Regression model: estimate vacancy rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.20, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    lin_reg = LinearRegression().fit(scaler.transform(X_train), y_train)
    pred_linear = np.clip(lin_reg.predict(scaler.transform(X_test)), 0, 1)

    vacancy_regressor = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        l2_regularization=1.0,
        loss="squared_error",
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    pred_hgb = np.clip(vacancy_regressor.predict(X_test), 0, 1)

    reg_importance = permutation_importance(
        vacancy_regressor,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    reg_order = np.argsort(reg_importance.importances_mean)[::-1][:12]

    results["regression_model"] = {
        "task": "estimate vacancy rate",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_linear_regression": regression_metrics(y_test, pred_linear),
        "hist_gradient_boosting": regression_metrics(y_test, pred_hgb),
        "top_features_by_mae_increase": [
            {
                "feature": feature_names[i],
                "mae_increase_when_shuffled": round_float(reg_importance.importances_mean[i]),
            }
            for i in reg_order
        ],
        "sample_predictions": [
            {"predicted_vacancy": round_float(p), "actual_vacancy": round_float(a)}
            for p, a in zip(pred_hgb[:5], y_test[:5])
        ],
    }

    # Classification model: predict high vacancy risk.
    X_train_all, X_test_cls, y_train_all, y_test_cls = train_test_split(
        X, y_cls, test_size=0.20, stratify=y_cls, random_state=RANDOM_STATE
    )
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.20,
        stratify=y_train_all,
        random_state=RANDOM_STATE,
    )

    cls_scaler = StandardScaler().fit(X_train_cls)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logistic.fit(cls_scaler.transform(X_train_cls), y_train_cls)
    prob_logistic = logistic.predict_proba(cls_scaler.transform(X_test_cls))[:, 1]

    risk_classifier = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ).fit(X_train_cls, y_train_cls)
    prob_raw = risk_classifier.predict_proba(X_test_cls)[:, 1]

    calibrated = CalibratedClassifierCV(FrozenEstimator(risk_classifier), method="isotonic")
    calibrated.fit(X_val_cls, y_val_cls)
    prob_val = calibrated.predict_proba(X_val_cls)[:, 1]
    prob_calibrated = calibrated.predict_proba(X_test_cls)[:, 1]

    selected = select_threshold_for_recall(y_val_cls, prob_val, min_recall=0.80)
    chosen_threshold = selected["threshold"]

    cls_importance = permutation_importance(
        risk_classifier,
        X_test_cls,
        y_test_cls,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    cls_order = np.argsort(cls_importance.importances_mean)[::-1][:12]

    results["classification_model"] = {
        "task": "classify high vacancy risk",
        "train_rows": int(len(X_train_cls)),
        "validation_rows": int(len(X_val_cls)),
        "test_rows": int(len(X_test_cls)),
        "test_high_risk_rate": round_float(y_test_cls.mean()),
        "baseline_logistic_regression_at_0_50": classification_metrics(y_test_cls, prob_logistic, 0.50),
        "hist_gradient_boosting_raw_at_0_50": classification_metrics(y_test_cls, prob_raw, 0.50),
        "calibrated_hist_gradient_boosting_at_0_50": classification_metrics(
            y_test_cls, prob_calibrated, 0.50
        ),
        "validation_threshold_rule": {
            "goal": "choose highest precision threshold with validation recall >= 0.80",
            "selected_threshold": round_float(chosen_threshold, 2),
            "validation_recall": round_float(selected["recall"]),
            "validation_precision": round_float(selected["precision"]),
        },
        "final_calibrated_model_at_selected_threshold": classification_metrics(
            y_test_cls, prob_calibrated, chosen_threshold
        ),
        "top_features_by_auc_increase": [
            {
                "feature": feature_names[i],
                "auc_increase_when_shuffled": round_float(cls_importance.importances_mean[i]),
            }
            for i in cls_order
        ],
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as output:
        json.dump(results, output, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
