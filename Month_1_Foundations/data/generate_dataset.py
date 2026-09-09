"""
Dataset Generator
------------------
This script creates a synthetic but realistic dataset of student profiles.
Each student has academic scores, self-rated skills, interest areas, and
project/internship experience. The target column is the best-fit career
path for that student.

The dataset is generated using clear rules mixed with random noise, so the
patterns are learnable by a machine learning model but not perfectly
predictable (just like real data).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

CAREER_PATHS = [
    "Data Science",
    "AI / Machine Learning Engineering",
    "Web Development",
    "Cyber Security",
    "Cloud / DevOps Engineering",
]

N_SAMPLES = 1200


def generate_row():
    cgpa = round(np.random.normal(3.2, 0.4), 2)
    cgpa = min(max(cgpa, 1.5), 4.0)

    math_score = np.random.randint(40, 100)
    programming_score = np.random.randint(40, 100)
    communication_score = np.random.randint(40, 100)

    ml_skill = np.random.randint(0, 10)
    web_skill = np.random.randint(0, 10)
    security_skill = np.random.randint(0, 10)
    cloud_skill = np.random.randint(0, 10)
    stats_skill = np.random.randint(0, 10)

    internship_count = np.random.randint(0, 4)
    project_count = np.random.randint(0, 6)

    interest = np.random.choice(
        ["data", "ai", "web", "security", "cloud"], p=[0.2, 0.2, 0.25, 0.15, 0.2]
    )

    # scoring system: each career path gets a score based on relevant skills
    scores = {
        "Data Science": stats_skill * 1.5 + math_score / 20 + (interest == "data") * 5,
        "AI / Machine Learning Engineering": ml_skill * 1.6 + programming_score / 20
        + (interest == "ai") * 5,
        "Web Development": web_skill * 1.6 + programming_score / 25
        + (interest == "web") * 5,
        "Cyber Security": security_skill * 1.7 + (interest == "security") * 6,
        "Cloud / DevOps Engineering": cloud_skill * 1.6 + (interest == "cloud") * 5,
    }

    # add noise so the label is not 100% deterministic from the score
    for k in scores:
        scores[k] += np.random.normal(0, 2.0)

    label = max(scores, key=scores.get)

    return {
        "cgpa": cgpa,
        "math_score": math_score,
        "programming_score": programming_score,
        "communication_score": communication_score,
        "ml_skill": ml_skill,
        "web_skill": web_skill,
        "security_skill": security_skill,
        "cloud_skill": cloud_skill,
        "stats_skill": stats_skill,
        "internship_count": internship_count,
        "project_count": project_count,
        "interest_area": interest,
        "career_path": label,
    }


def main():
    rows = [generate_row() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)
    df.to_csv("data/student_career_data.csv", index=False)
    print(f"Dataset created with {len(df)} rows.")
    print(df["career_path"].value_counts())


if __name__ == "__main__":
    main()
