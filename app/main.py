from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# load things
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")
default_input = joblib.load("models/default_input.pkl")

app = FastAPI()

# request schema
class InputData(BaseModel):
    amount: float
    time: float

def prepare_input(amount, time):
    input_data = default_input.copy()
    
    input_data['Amount'] = amount
    input_data['Time'] = time
    
    return pd.DataFrame([input_data])

@app.post("/predict")
def predict(data: InputData):
    input_df = prepare_input(data.amount, data.time)
    
    # scale only these columns
    input_df[['Amount','Time']] = scaler.transform(input_df[['Amount','Time']])
    
    prob = model.predict_proba(input_df)[0][1]
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": prob > 0.5
    }

@app.get("/")
def home():
    return {"status": "API running"}
