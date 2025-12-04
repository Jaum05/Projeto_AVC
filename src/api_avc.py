from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from typing import Optional
import pandas as pd
import os

# helpers do projeto
from src.preprocess import prepare_input_dataframe
from src.utils import load_artifact  # usa a função defensiva que criamos

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# caminhos / configs
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "./models/artifact_stroke.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/stroke_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "./data/stroke_clean.csv")
# fallback threshold (será sobrescrito se artifact trouxer threshold)
DEFAULT_THRESHOLD = float(os.getenv("PRED_THRESHOLD", 0.492))


# Carregar modelo/artifact com segurança
_model = None
_model_threshold = DEFAULT_THRESHOLD

# Tenta artifact primeiro (pipeline + threshold)
try:
    if os.path.exists(ARTIFACT_PATH):
        artifact = load_artifact(ARTIFACT_PATH)  # pode lançar FileNotFoundError ou retornar dict/pipeline
        if isinstance(artifact, dict) and "pipeline" in artifact:
            _model = artifact["pipeline"]
            _model_threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))
            print(f"✔ Carregado artifact: {ARTIFACT_PATH} (threshold={_model_threshold})")
        else:
            # se o artifact for um pipeline direto
            _model = artifact
            print(f"✔ Carregado pipeline direto de artifact: {ARTIFACT_PATH}")
    else:
        # fallback para MODEL_PATH (compatibilidade)
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            print(f"✔ Carregado modelo isolado: {MODEL_PATH} (sem threshold salvo)")
        else:
            print("⚠ Nenhum modelo encontrado. Configure ARTIFACT_PATH ou MODEL_PATH corretamente.")
except Exception as e:
    print("Erro ao carregar modelo/artifact:", e)
    _model = None

model = _model
MODEL_THRESHOLD = _model_threshold

# carregar reference_df de forma segura
reference_df = None
try:
    if os.path.exists(DATA_PATH):
        reference_df = pd.read_csv(DATA_PATH)
        print(f"✔ reference_df carregado: {DATA_PATH}")
    else:
        print(f"⚠ DATA_PATH não encontrado ({DATA_PATH}). Alguns endpoints exigirão reference_df.")
except Exception as e:
    print("Erro ao carregar reference_df:", e)
    reference_df = None


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
        if model is None:
            raise RuntimeError("Modelo não carregado. Verifique ARTIFACT_PATH / MODEL_PATH.")

        if reference_df is None:
            raise RuntimeError("reference_df não disponível. Verifique DATA_PATH.")

        raw = data.model_dump()
        raw.pop("ever_married", None)  # mantém compatibilidade com sua lógica atual

        input_df = prepare_input_dataframe(raw, reference_df)

        # tenta obter probabilidade; se não suportar, usa predict como fallback
        try:
            proba = model.predict_proba(input_df)[0][1]
        except Exception as e_proba:
            # fallback: tentar usar predict (0/1)
            try:
                pred_val = int(model.predict(input_df)[0])
                return {
                    "prediction": pred_val,
                    "probability": None,
                    "threshold_used": None,
                    "warning": f"predict_proba não disponível: {e_proba}"
                }
            except Exception as e_pred:
                raise RuntimeError(f"predict_proba falhou: {e_proba}; fallback predict também falhou: {e_pred}")

        return {
            "prediction": int(proba >= MODEL_THRESHOLD),
            "probability": round(float(proba * 100), 2),
            "threshold_used": MODEL_THRESHOLD
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "threshold": MODEL_THRESHOLD}
