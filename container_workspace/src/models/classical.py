from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=42)


def get_svm_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]
    )


def get_xgboost_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
