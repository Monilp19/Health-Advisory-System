"""
Health Advisory System - Live Model Training

The training data is read directly from data/disease_symptom.csv.
The symptom severity dataset is used as an additional model feature.
The saved artifacts are compatible with app.py.

Run manually when needed:
    python train_model.py
"""

from pathlib import Path
import warnings
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "disease_symptom.csv"
SEVERITY_FILE = DATA_DIR / "symptom_severity.csv"

MODEL_FILE = BASE_DIR / "disease_model.pkl"
ENCODER_FILE = BASE_DIR / "label_encoder.pkl"
FEATURE_FILE = BASE_DIR / "feature_cols.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 300


def normalize(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def strict_normalize(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize(value))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def find_target_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if normalize(column) in {"disease", "prognosis"}:
            return column
    raise ValueError("disease_symptom.csv must contain Disease or Prognosis.")


def load_severity() -> dict[str, float]:
    df = read_csv(SEVERITY_FILE)

    symptom_col = next(
        (c for c in df.columns if normalize(c) == "symptom"),
        None,
    )
    weight_col = next(
        (c for c in df.columns if normalize(c) in {"weight", "severity", "severity weight"}),
        None,
    )

    if symptom_col is None or weight_col is None:
        raise ValueError(
            "symptom_severity.csv must contain 'symptom' and 'weight' columns."
        )

    severity = {}

    for _, row in df.iterrows():
        symptom = normalize(row[symptom_col])
        if not symptom:
            continue

        try:
            weight = float(row[weight_col])
        except (TypeError, ValueError):
            continue

        if weight > 0:
            severity[strict_normalize(symptom)] = weight

    return severity


def build_training_matrix(
    df: pd.DataFrame,
    target_col: str,
    severity: dict[str, float],
):
    symptom_columns = [c for c in df.columns if c != target_col]
    symptom_data = df[symptom_columns].map(normalize)

    all_symptoms = sorted(
        value
        for value in pd.unique(symptom_data.to_numpy().ravel())
        if value
    )

    X = pd.DataFrame(
        0.0,
        index=df.index,
        columns=all_symptoms,
        dtype=np.float32,
    )

    for column in symptom_columns:
        values = symptom_data[column]
        values = values[values != ""]

        for symptom, indices in values.groupby(values).groups.items():
            X.loc[indices, symptom] = severity.get(strict_normalize(symptom), 1.0)

    # Total symptom severity is a derived feature from the supplied
    # symptom_severity.csv and is also used during live prediction.
    X["total_symptom_severity"] = X.sum(axis=1)

    y = df[target_col].fillna("Unknown").astype(str).str.strip()

    return X, y


def train_model(verbose: bool = True):
    """Train and return the model plus metadata."""

    train_df = read_csv(TRAIN_FILE)
    severity = load_severity()

    target_col = find_target_column(train_df)

    train_df[target_col] = (
        train_df[target_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    X, y = build_training_matrix(
        train_df,
        target_col,
        severity,
    )

    if X.empty or X.shape[1] <= 1:
        raise ValueError("No usable symptom features were found.")

    if y.nunique() < 2:
        raise ValueError("At least two disease classes are required.")

    class_counts = y.value_counts()
    min_class = int(class_counts.min())

    if min_class < 2:
        raise ValueError(
            "Every disease needs at least 2 training rows."
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    cv_folds = min(5, min_class)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        model,
        X,
        y_encoded,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    # Final training on all current data.
    model.fit(X, y_encoded)

    if verbose:
        print("\n" + "=" * 70)
        print("HEALTH ADVISORY SYSTEM - MODEL TRAINING")
        print("=" * 70)
        print(f"Rows:                {len(X)}")
        print(f"Symptoms/features:   {X.shape[1] - 1}")
        print(f"Diseases/classes:    {y.nunique()}")
        print(f"Test accuracy:       {test_accuracy:.4f}")
        print(
            f"{cv_folds}-fold CV accuracy: "
            f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
        )

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=encoder.classes_,
                zero_division=0,
            )
        )

    metadata = {
        "rows": len(X),
        "symptoms": X.shape[1] - 1,
        "diseases": y.nunique(),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "severity": severity,
    }

    return model, encoder, list(X.columns), metadata


def save_artifacts(model, encoder, feature_cols):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(feature_cols, FEATURE_FILE)


def main():
    model, encoder, feature_cols, metadata = train_model(verbose=True)
    save_artifacts(model, encoder, feature_cols)

    print("\nSaved:")
    print(" -", MODEL_FILE)
    print(" -", ENCODER_FILE)
    print(" -", FEATURE_FILE)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

