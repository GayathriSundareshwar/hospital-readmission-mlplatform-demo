from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "NoShowPredictionModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

THRESHOLD = 0.30


app = FastAPI(
    title="Healthcare No-Show Prediction API",
    version="1.0.0"
)

model = mlflow.sklearn.load_model(MODEL_URI)


class PredictionRequest(BaseModel):
    Gender: str
    Age: int
    age_group: str
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    sms_received_flag: int
    days_until_appointment: int
    chronic_disease_count: int


@app.get("/")
def home():
    return {
        "message": "Healthcare No-Show Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    input_df = pd.DataFrame(
        [{
            "Gender": request.Gender,
            "Age": request.Age,
            "age_group": request.age_group,
            "Scholarship": request.Scholarship,
            "Hipertension": request.Hipertension,
            "Diabetes": request.Diabetes,
            "Alcoholism": request.Alcoholism,
            "Handcap": request.Handcap,
            "sms_received_flag": request.sms_received_flag,
            "days_until_appointment": request.days_until_appointment,
            "chronic_disease_count": request.chronic_disease_count
        }]
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    label = (
        "Likely No-show"
        if prediction == 1
        else "Likely Show"
    )

    return {
        "no_show_probability": round(probability, 4),
        "predicted_no_show": prediction,
        "prediction_label": label
    }