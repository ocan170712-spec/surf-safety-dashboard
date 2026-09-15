"""Train a severity classifier on the cleaned NEISS data.

Predicts Mild / Moderate / Severe outcome from circumstance features that
would be known right when a case comes in (age, sex, season, body part,
injury type) rather than from the disposition itself (that would be leakage,
since severity is derived from disposition).

Output: data/processed/severity_model.joblib, data/processed/model_metrics.json
"""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/processed/neiss_clean.csv"
MODEL_PATH = "data/processed/severity_model.joblib"
METRICS_PATH = "data/processed/model_metrics.json"

# Collapses ~24 NEISS body-part codes into broad regions so the model isn't
# fit against dozens of sparse one-hot columns.
BODY_PART_GROUP = {
    "Head": "Head/Face/Neck", "Face": "Head/Face/Neck", "Neck": "Head/Face/Neck",
    "Eyeball": "Head/Face/Neck", "Mouth": "Head/Face/Neck", "Ear": "Head/Face/Neck",
    "Upper Trunk": "Trunk", "Lower Trunk": "Trunk", "Pubic Region": "Trunk",
    "All Parts Body": "Trunk", "Internal": "Trunk",
    "Shoulder": "Arm/Hand", "Upper Arm": "Arm/Hand", "Lower Arm": "Arm/Hand",
    "Elbow": "Arm/Hand", "Wrist": "Arm/Hand", "Hand": "Arm/Hand", "Finger": "Arm/Hand",
    "Upper Leg": "Leg/Foot", "Lower Leg": "Leg/Foot", "Knee": "Leg/Foot",
    "Ankle": "Leg/Foot", "Foot": "Leg/Foot", "Toe": "Leg/Foot",
}

FEATURES = ["age_years", "sex", "season", "body_part_group", "diagnosis_group"]
TARGET = "severity"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["body_part_group"] = df["body_part"].map(BODY_PART_GROUP).fillna("Other")
    return df


def build_pipeline() -> Pipeline:
    categorical = ["sex", "season", "body_part_group", "diagnosis_group"]
    numeric = ["age_years"]
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def main():
    df = load_features().dropna(subset=FEATURES + [TARGET])
    X, y = df[FEATURES], df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
    print(classification_report(y_test, pipeline.predict(X_test)))

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"classification_report": report, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
