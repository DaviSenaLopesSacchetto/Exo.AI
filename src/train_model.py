from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import os
import numpy as np

# Caminhos dos arquivos
MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/exoplanets_balanced.csv"

# 🔧 Cria pasta se não existir
os.makedirs("models", exist_ok=True)

# 1️⃣ Carregar os dados
df = pd.read_csv(DATA_PATH)

# 2️⃣ Definir features (X) e rótulo (y)
X = df[["koi_prad", "koi_period", "koi_steff", "koi_srad", "koi_depth", "koi_duration", "koi_model_snr"]]
y = df["koi_disposition"]

# 3️⃣ Separar treino e teste (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ Normalizar (Z-score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Criar e treinar o modelo Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 6️⃣ Avaliar modelo (opcional, só pra debug)
y_pred = model.predict(X_test_scaled)
print("✅ Modelo treinado com sucesso!\n")
print("📊 Relatório de classificação:")
print(classification_report(y_test, y_pred))
print("📈 Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# 7️⃣ Salvar modelo e scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Modelo e scaler salvos com sucesso em /models/")
