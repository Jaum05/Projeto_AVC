# src/train_pipeline.py
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from dotenv import load_dotenv


from preprocess import build_preprocess_pipeline




def main():
    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH", "./data/stroke_clean.csv")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/stroke_model.pkl")


    # carregar dados
    print("📥 Carregando dados de:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)


    # checagens básicas
    if "stroke" not in df.columns:
        raise ValueError("Coluna alvo 'stroke' não encontrada no CSV")


    # separar X/y
    X = df.drop(columns=["stroke"])
    y = df["stroke"]


    # construir preprocessor
    preprocessor, num_cols, cat_cols = build_preprocess_pipeline(X)


    # montar pipeline completo
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", clf)
    ])


    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    print("🚀 Treinando...")
    pipeline.fit(X_train, y_train)


    # avaliar
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)


    print(f"✅ Treino completo — Acc: {acc:.4f} AUC: {auc:.4f}")


    # salvar pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)


    print("💾 Modelo salvo em:", MODEL_PATH)




if __name__ == "__main__":
    main()