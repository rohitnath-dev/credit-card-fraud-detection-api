from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# load artifacts
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")
default_input = joblib.load("models/default_input.pkl")

app = FastAPI()


class InputData(BaseModel):
    amount: float
    time: float


def prepare_input(amount: float, time: float):
    data = default_input.copy()

    data["Amount"] = amount
    data["Time"] = time

    df = pd.DataFrame([data])

    # ensure column order consistency
    df = df[default_input.index]

    return df


@app.post("/predict")
def predict(data: InputData):
    input_df = prepare_input(data.amount, data.time)

    input_df[["Amount", "Time"]] = scaler.transform(
        input_df[["Amount", "Time"]]
    )

    prob = model.predict_proba(input_df.values)[0][1]

    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob > threshold)
    }


@app.get("/")
def home():
    return {"status": "API running"}
