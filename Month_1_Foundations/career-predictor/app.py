"""
AI Career Path & Skill-Gap Predictor
--------------------------------------
A Flask web application. A student fills a form describing their academic
scores, self-rated skills, and interest area. The trained machine learning
model predicts the best-fit career path, along with the confidence for
every path and a personalized skill-gap report showing which skills to
improve for the recommended path.
"""

import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model/career_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("model/feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

REQUIRED_SKILLS = {
    "Data Science": {"stats_skill": 8, "math_score": 80, "programming_score": 70},
    "AI / Machine Learning Engineering": {"ml_skill": 8, "programming_score": 80, "math_score": 75},
    "Web Development": {"web_skill": 8, "programming_score": 75},
    "Cyber Security": {"security_skill": 8, "programming_score": 65},
    "Cloud / DevOps Engineering": {"cloud_skill": 8, "programming_score": 65},
}

SKILL_LABELS = {
    "ml_skill": "Machine Learning",
    "web_skill": "Web Development",
    "security_skill": "Cyber Security",
    "cloud_skill": "Cloud Platforms",
    "stats_skill": "Statistics",
    "math_score": "Mathematics",
    "programming_score": "Programming",
}


def build_feature_row(form):
    raw = {
        "cgpa": float(form["cgpa"]),
        "math_score": int(form["math_score"]),
        "programming_score": int(form["programming_score"]),
        "communication_score": int(form["communication_score"]),
        "ml_skill": int(form["ml_skill"]),
        "web_skill": int(form["web_skill"]),
        "security_skill": int(form["security_skill"]),
        "cloud_skill": int(form["cloud_skill"]),
        "stats_skill": int(form["stats_skill"]),
        "internship_count": int(form["internship_count"]),
        "project_count": int(form["project_count"]),
        "interest_area": form["interest_area"],
    }

    row = {col: 0 for col in FEATURE_COLUMNS}
    for key, value in raw.items():
        if key == "interest_area":
            dummy_col = f"interest_{value}"
            if dummy_col in row:
                row[dummy_col] = 1
        else:
            if key in row:
                row[key] = value

    return pd.DataFrame([row])[FEATURE_COLUMNS], raw


def build_skill_gap(predicted_path, raw):
    required = REQUIRED_SKILLS.get(predicted_path, {})
    gaps = []
    for skill_key, target_value in required.items():
        current_value = raw.get(skill_key, 0)
        if current_value < target_value:
            gaps.append(
                {
                    "skill": SKILL_LABELS.get(skill_key, skill_key),
                    "current": current_value,
                    "target": target_value,
                }
            )
    return gaps


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    features_df, raw = build_feature_row(request.form)

    probabilities = model.predict_proba(features_df)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_path = label_encoder.inverse_transform([predicted_index])[0]

    ranked = sorted(
        zip(label_encoder.classes_, probabilities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    ranked_results = [{"path": path, "confidence": round(prob * 100, 1)} for path, prob in ranked]

    skill_gaps = build_skill_gap(predicted_path, raw)

    result = {
        "predicted_path": predicted_path,
        "confidence": ranked_results[0]["confidence"],
        "ranked_results": ranked_results,
        "skill_gaps": skill_gaps,
    }

    return render_template("index.html", result=result, form_data=request.form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
