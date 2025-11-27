from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API Prevenção de AVC")

model = joblib.load("../models/stroke_model.pkl")

class PatientData(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    work_type: int
    Residence_type: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int

@app.post("/predict")
def predict_stroke(data: PatientData):
    features = np.array([[v for v in data.dict().values()]])
    pred = model.predict(features)[0]
    return {"stroke_risk": int(pred)}
