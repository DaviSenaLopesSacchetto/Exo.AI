import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.utils.class_weight import compute_class_weight
import os

# -------------------------
# Carregar dataset processado
# -------------------------
df = pd.read_csv("data/processed/exoplanets_balanced.csv")

features = ["koi_prad", "koi_period", "koi_steff", "koi_srad", 
            "koi_depth", "koi_duration", "koi_model_snr"]
target = "koi_disposition"

# Separar features e target
X = df[features]
y = df[target]

# -------------------------
# Divisão treino/teste
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# Normalização
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Classes do modelo
# -------------------------
classes = np.array(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])

# -------------------------
# Pesos balanceados
# -------------------------
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, class_weights_values))
print("Class weights:", class_weights)

# -------------------------
# Inicializa modelo incremental
# -------------------------
model = SGDClassifier(
    loss='log_loss',          # regressão logística
    max_iter=1,
    warm_start=True,
    class_weight=class_weights,
    random_state=42
)

# -------------------------
# Inicialização justa: 1 amostra de cada classe
# -------------------------
init_batch = pd.concat([
    df[df[target] == "CONFIRMED"].sample(1, random_state=42),
    df[df[target] == "CANDIDATE"].sample(1, random_state=42),
    df[df[target] == "FALSE POSITIVE"].sample(1, random_state=42)
])

X_init = scaler.transform(init_batch[features])
y_init = init_batch[target]

model.partial_fit(X_init, y_init, classes=classes)

# -------------------------
# Treino incremental com todo dataset
# -------------------------
X_full_scaled = scaler.transform(X)
y_full = y
model.partial_fit(X_full_scaled, y_full)

# -------------------------
# Avaliação
# -------------------------
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# -------------------------
# Salvar modelo e scaler
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sgd_incremental.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Treinamento concluído! Modelo salvo em 'models/sgd_incremental.pkl'")
