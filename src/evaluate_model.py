import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar modelo e scaler salvos
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Carregar dataset de teste
df = pd.read_csv("data/processed/exoplanets_balanced.csv")

features = ["koi_prad", "koi_period", "koi_steff", "koi_srad", 
            "koi_depth", "koi_duration", "koi_model_snr"]
target = "koi_disposition"

# Separar X e y
X = df[features]
y = df[target]

# Normalizar usando o scaler já treinado
X_scaled = scaler.transform(X)

# Fazer predição
y_pred = model.predict(X_scaled)

# Relatório de classificação
classes = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
print(classification_report(y, y_pred, target_names=classes))

# Matriz de confusão
cm = confusion_matrix(y, y_pred, labels=classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.show()
