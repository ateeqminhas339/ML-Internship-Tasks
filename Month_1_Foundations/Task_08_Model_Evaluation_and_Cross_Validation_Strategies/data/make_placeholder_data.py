"""Generates a synthetic stand-in for heart.csv (same schema/shape as the
UCI Heart Disease dataset) so the notebook is runnable before the real file
is downloaded. DELETE this placeholder once the real dataset is in place.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

age = rng.integers(29, 78, size=n)
sex = rng.integers(0, 2, size=n)  # 1 = male, 0 = female
cp = rng.integers(0, 4, size=n)  # chest pain type
trestbps = rng.integers(94, 201, size=n)  # resting blood pressure
chol = rng.integers(126, 565, size=n)  # cholesterol
fbs = rng.integers(0, 2, size=n)  # fasting blood sugar > 120
restecg = rng.integers(0, 3, size=n)
thalach = rng.integers(71, 203, size=n)  # max heart rate
exang = rng.integers(0, 2, size=n)  # exercise-induced angina
oldpeak = np.round(rng.uniform(0, 6.2, size=n), 1)
slope = rng.integers(0, 3, size=n)
ca = rng.integers(0, 4, size=n)
thal = rng.integers(0, 4, size=n)

# target correlated with a few risk factors, plus noise, so the models have
# real signal to find rather than pure randomness
risk = (
    0.03 * (age - 50) + 0.6 * exang + 0.02 * (chol - 240)
    - 0.015 * (thalach - 150) + 0.4 * oldpeak + rng.normal(0, 1.5, size=n)
)
target = (risk > np.median(risk)).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target,
})
df.to_csv("data/heart.csv", index=False)
print("Wrote placeholder data/heart.csv:", df.shape)
