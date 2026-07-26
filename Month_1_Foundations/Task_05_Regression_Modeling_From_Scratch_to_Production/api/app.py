"""Production FastAPI app serving the trained insurance-charges regression
model. Loads api/model.joblib (produced by notebook.ipynb) and exposes a
single /predict endpoint.
"""

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

app = FastAPI(
    title="Insurance Charges Prediction API",
    description="Predicts medical insurance charges from customer attributes (Task 05).",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    sex: str = Field(..., example="male")
    bmi: float = Field(..., ge=10, le=60, example=28.5)
    children: int = Field(..., ge=0, le=10, example=2)
    smoker: str = Field(..., example="no")
    region: str = Field(..., example="southeast")


class PredictionResponse(BaseModel):
    predicted_charges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model = get_model()

    row = pd.DataFrame([{
        "age": request.age,
        "sex": request.sex.lower(),
        "bmi": request.bmi,
        "children": request.children,
        "smoker": request.smoker.lower(),
        "region": request.region.lower(),
    }])
    row["smoker_bmi_interaction"] = (row["smoker"] == "yes").astype(int) * row["bmi"]

    prediction = model.predict(row)[0]
    return PredictionResponse(predicted_charges=round(float(prediction), 2))
