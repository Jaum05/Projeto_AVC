import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocess_pipeline(df: pd.DataFrame):

    # detectar colunas automaticamente
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()


    # remover target caso esteja presente
    if "stroke" in numeric_features:
        numeric_features.remove("stroke")
    if "stroke" in categorical_features:
        categorical_features.remove("stroke")


    # Pipelines
    numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])


    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")


    return preprocessor, numeric_features, categorical_features


def prepare_input_dataframe(input_json: dict, reference_df: pd.DataFrame):

    # construir um DataFrame com uma linha
    input_df = pd.DataFrame([input_json])


    # garantir que todas colunas do reference_df (menos target) existam no input
    expected_cols = reference_df.drop(columns=["stroke"], errors="ignore").columns.tolist()


    for c in expected_cols:
        if c not in input_df.columns:
            input_df[c] = pd.NA


    # reordenar colunas como no treino
    input_df = input_df[expected_cols]


    return input_df