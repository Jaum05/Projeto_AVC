from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from typing import Optional
import pandas as pd

import os

from src.preprocess import prepare_input_dataframe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "./models/stroke_model.pkl")
model = joblib.load(MODEL_PATH)
DATA_PATH = os.getenv("DATA_PATH", "./data/stroke_clean.csv")


reference_df = pd.read_csv(DATA_PATH)

class PredictInput(BaseModel):
    gender: str                   
    age: float
    hypertension: int              
    heart_disease: int             
    ever_married: Optional[str] = None  
    work_type: str                 
    Residence_type: str            
    avg_glucose_level: float
    bmi: float
    smoking_status: str           


@app.post("/predict")
async def predict(data: PredictInput):
    try:
        raw = data.model_dump()
        
        raw.pop("ever_married", None)
        
        input_df = prepare_input_dataframe(raw, reference_df)
        
        proba = model.predict_proba(input_df)[0][1]
        
        return {
            "prediction": int(proba >= 0.5),
            "probability": round(float(proba * 100), 2)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}