import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier

print("📥 Carregando dados...")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

data_path = os.path.join(data_dir, "stroke_clean.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError("❌ stroke_clean.csv não encontrado")

print(f"🗂️ Usando dataset: {data_path}")

df = pd.read_csv(data_path)

# Separar X e y
X = df.drop("stroke", axis=1)
y = df["stroke"]

print("🧹 Pré-processando os dados...")

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Aplicar transformação preliminar
X_transformed = preprocessor.fit_transform(X)

print("📊 Aplicando SMOTE para balancear classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

print(f"Antes SMOTE: {y.value_counts().to_dict()}")
print(f"Depois SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

print("⚙️ Treinando modelo XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("📈 Predizendo probabilidades...")
y_prob = model.predict_proba(X_test)[:, 1]

# Encontrar o melhor threshold baseado no F1
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🎯 Melhor threshold encontrado: {best_thresh:.3f}")
print(f"🏆 Melhor F1-score: {best_f1:.4f}\n")

# Avaliação final com o melhor threshold
final_preds = (y_prob >= best_thresh).astype(int)

print("📋 Relatório final:")
print(classification_report(y_test, final_preds, zero_division=0))
print(f"✔️ Acurácia: {accuracy_score(y_test, final_preds):.4f}")

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


save_path = os.path.join(model_dir, "stroke_model.pkl")
joblib.dump(final_pipeline, save_path)

print(f"\n💾 Modelo substituído com sucesso: {save_path}")
print("✅ Novo modelo treinado e otimizado!")
