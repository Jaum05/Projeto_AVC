import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Função para carregar os dados
def load_data(path):
    df = pd.read_csv(path)
    return df


# Função para pré-processar os dados
def preprocess_data(df):
    df = df.dropna()

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Treinar o modelo
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/stroke_model.pkl')
    return model

# Avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Acurácia: {acc:.2f}")
    print("\n📋 Relatório de classificação:")
    print(classification_report(y_test, y_pred))
