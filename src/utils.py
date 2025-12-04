
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from typing import Tuple, Optional, Dict, Any


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega CSV em um DataFrame.
    """
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Limpeza simples:
      - remove linhas completamente nulas
      - remove coluna 'id' se existir (opcional)
      - retorna cópia limpa
    NOTA: não faz encoding/escalonamento — isso deve ser feito pelo pipeline salvo.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("basic_clean espera um pandas.DataFrame")
    df_clean = df.dropna(how="all").copy()
    if drop_id and "id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id"])
    return df_clean


def get_features_target(df: pd.DataFrame, target: str = "stroke") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y. Lança erro se target não existir.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("get_features_target espera um pandas.DataFrame")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no DataFrame.")
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return X, y


def split_data(df: pd.DataFrame, target: str = "stroke", test_size: float = 0.2,
               random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Faz train_test_split retornando X_train, X_test, y_train, y_test.
    Usa stratify por padrão (recomendado para problemas desbalanceados).
    """
    X, y = get_features_target(df, target=target)
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula métricas básicas e imprime relatório.
    Retorna dicionário com as métricas chaves.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"🎯 Acurácia: {acc:.4f}")
    print(f"Precision (classe positiva): {prec:.4f}")
    print(f"Recall (classe positiva): {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\n📋 Relatório de classificação:")
    print(classification_report(y_true, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_artifact(pipeline_obj, threshold: float = 0.492, path: str = "./models/artifact_stroke.pkl",
                  metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Salva um dict contendo:
      {"pipeline": pipeline_obj, "threshold": <float>, "metadata": <dict opcional>}
    Cria diretório se necessário.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact = {"pipeline": pipeline_obj, "threshold": float(threshold)}
    if metadata is not None:
        artifact["metadata"] = metadata
    joblib.dump(artifact, path)
    print(f"✔ Artifact salvo em: {path}")


def load_artifact(path: str = "./models/artifact_stroke.pkl"):
    """
    Carrega e retorna o conteúdo do artifact (dict ou pipeline direto).
    Lança FileNotFoundError se não existir.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact não encontrado em: {path}")
    return joblib.load(path)


def predict_from_raw(artifact_or_model, payload_json: dict, reference_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Faz predição a partir de um payload bruto (dicionário), usando prepare_input_dataframe
    para alinhar as colunas com reference_df.

    artifact_or_model: pode ser:
      - dict com 'pipeline' e opcional 'threshold' (artifact salvo)
      - pipeline/estimator diretamente (ex.: carregado com joblib.load)
    payload_json: dicionário com campos brutos (igual ao payload da API)
    reference_df: DataFrame de referência (mesma estrutura usada no treino) — necessário para alinhar colunas

    Retorna:
      {"prediction": int(0|1), "probability": float|None, "threshold_used": float|None}
    """
    from src.preprocess import prepare_input_dataframe

    if not isinstance(payload_json, dict):
        raise ValueError("payload_json deve ser um dicionário (JSON-like).")
    if not isinstance(reference_df, pd.DataFrame):
        raise ValueError("reference_df deve ser um pandas.DataFrame (o dataframe de referência do treino).")

    if isinstance(artifact_or_model, dict):
        if "pipeline" not in artifact_or_model:
            raise ValueError("artifact dict deve conter a chave 'pipeline'.")
        model = artifact_or_model["pipeline"]
        threshold = artifact_or_model.get("threshold", None)
        threshold = float(threshold) if threshold is not None else None
    elif hasattr(artifact_or_model, "predict"):
        model = artifact_or_model
        threshold = None
    else:
        raise ValueError("artifact_or_model inválido: deve ser um dict com 'pipeline' ou um model com método predict.")

    
    input_df = prepare_input_dataframe(payload_json, reference_df)

    result = {"prediction": None, "probability": None, "threshold_used": threshold}

    try:
        proba_arr = model.predict_proba(input_df)
        
        if proba_arr is None:
            raise RuntimeError("model.predict_proba retornou None")
        if proba_arr.ndim != 2:
            raise RuntimeError(f"predict_proba retornou array com shape inesperado: {proba_arr.shape}")
        if proba_arr.shape[1] == 1:
            prob_pos = float(proba_arr[0][0])
        else:
            prob_pos = float(proba_arr[0][1])
        result["probability"] = prob_pos
        if threshold is not None:
            result["prediction"] = int(prob_pos >= float(threshold))
        else:
            result["prediction"] = int(model.predict(input_df)[0])
    except Exception as e_proba:
        
        try:
            pred = int(model.predict(input_df)[0])
            result["prediction"] = pred
            result["probability"] = None
            result["threshold_used"] = threshold
            result["_warning"] = f"predict_proba falhou: {e_proba}; usado predict() como fallback."
        except Exception as e_pred:
            raise RuntimeError(f"Falha ao predizer: predict_proba erro: {e_proba}; fallback predict erro: {e_pred}")

    return result
