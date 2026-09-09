"""
Model Training Script
-----------------------
Trains a Random Forest classifier that predicts the best-fit career path
for a student based on their academic and skill profile.

This also saves the label encoder and the list of feature columns so the
web application can reuse them at prediction time.
"""

import json
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/student_career_data.csv"
MODEL_PATH = "model/career_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
COLUMNS_PATH = "model/feature_columns.json"

# Skills needed for each career path (used for the skill-gap report)
REQUIRED_SKILLS = {
    "Data Science": {"stats_skill": 8, "math_score": 80, "programming_score": 70},
    "AI / Machine Learning Engineering": {"ml_skill": 8, "programming_score": 80, "math_score": 75},
    "Web Development": {"web_skill": 8, "programming_score": 75},
    "Cyber Security": {"security_skill": 8, "programming_score": 65},
    "Cloud / DevOps Engineering": {"cloud_skill": 8, "programming_score": 65},
}


def main():
    df = pd.read_csv(DATA_PATH)

    df_encoded = pd.get_dummies(df, columns=["interest_area"], prefix="interest")

    label_encoder = LabelEncoder()
    df_encoded["career_path_encoded"] = label_encoder.fit_transform(df_encoded["career_path"])

    feature_columns = [c for c in df_encoded.columns if c not in ["career_path", "career_path_encoded"]]

    X = df_encoded[feature_columns]
    y = df_encoded["career_path_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f)

    print("Model, encoder, and feature columns saved in the model/ folder.")


if __name__ == "__main__":
    main()
