"""Classification pipeline that adapts to whatever columns the data has.

Numeric and categorical columns are detected by dtype, so the same code trains
on the small synthetic frame and on the wider real UCI dataset without listing
20 column names by hand. Categoricals are one-hot encoded with
``handle_unknown="ignore"`` so a category unseen at fit time does not crash
prediction.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import TARGET


def split_columns(df: pd.DataFrame, target: str = TARGET) -> Tuple[List[str], List[str]]:
    features = [c for c in df.columns if c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_pipeline(numeric: List[str], categorical: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def train(df: pd.DataFrame, target: str = TARGET) -> Tuple[Pipeline, dict]:
    numeric, categorical = split_columns(df, target)
    X, y = df.drop(columns=[target]), df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    pipe = build_pipeline(numeric, categorical)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1_poisonous": round(float(f1_score(y_te, pred, pos_label="p")), 4),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
    }
    return pipe, metrics
