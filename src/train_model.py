import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.utils.class_weight import compute_class_weight


# Carregar dataset
df = pd.read_csv("data/processed/exoplanets_balanced.csv")

batch = df

# Separar features e target
features = ["koi_prad", "koi_period", "koi_steff", "koi_srad", "koi_depth", "koi_duration", "koi_model_snr", "koi_score"]
target = "koi_disposition"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classes = numpy.array(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])

class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train  # ou qualquer conjunto representativo grande
)

# Criar um dicionário para passar ao SGDClassifier
class_weights = dict(zip(classes, class_weights_values))
print(class_weights)

model = SGDClassifier(
    loss='log_loss',            # regressão logística (multi-classe)
    max_iter=1,            # uma iteração por batch (incremental)
    warm_start=True,       # mantém pesos entre fit
    class_weight=class_weights,  # evita colapso em classes minoritárias
    random_state=42
)

# Primeira chamada para inicializar pesos
model.partial_fit(X_train_scaled, y_train, classes=classes)

# Novo batch
X_new = batch[[features]]  # features do batch
y_new = batch[target] # rótulos do batch

# Normalizar usando o scaler incremental
X_new_scaled = scaler.transform(X_new)  # se já tiver fit; ou use partial_fit no scaler se quiser atualizar média/desvio


# Atualizar modelo
model.partial_fit(X_new_scaled, y_new)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=classes)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

joblib.dump(model, "models/sgd_incremental.pkl")
joblib.dump(scaler, "models/scaler.pkl")