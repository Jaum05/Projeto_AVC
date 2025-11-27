import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_dir, "data", "stroke_augmented.csv")
output_path = os.path.join(base_dir, "data", "stroke_clean.csv")

print("📥 Lendo arquivo original...")
df = pd.read_csv(input_path)

# Colunas desejadas
keep_cols = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
]

print("🧹 Limpando dataset...")
df_clean = df[keep_cols].copy()

df_clean["bmi"] = df_clean["bmi"].replace("N/A", None).astype(float)

print("💾 Salvando CSV limpo...")
df_clean.to_csv(output_path, index=False)

print(f"✅ CSV limpo salvo em: {output_path}")
