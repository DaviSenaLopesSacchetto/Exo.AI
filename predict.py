from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from utils import normalize_with_scaler, compute_class_weights

# -------------------------
# Caminhos dos arquivos
# -------------------------
MODEL_PATH = "models/sgd_incremental.pkl"
SCALER_PATH = "models/scaler.pkl"

# -------------------------
# Inicializa FastAPI
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Classes do modelo
# -------------------------
classes = np.array(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
label_map_num2str = {i: c for i, c in enumerate(classes)}
label_map_str2num = {v: k for k, v in label_map_num2str.items()}

# -------------------------
# Carrega ou cria modelo e scaler
# -------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    os.makedirs("models", exist_ok=True)
    # Modelo incremental
    model = SGDClassifier(
        loss='log_loss',  # regressão logística
        max_iter=1,
        warm_start=True,
        random_state=42
    )
    # Inicializa scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Salva arquivos vazios inicialmente
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

# -------------------------
# Endpoint principal
# -------------------------
@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas (SGDClassifier incremental) pronta 🚀"}

# -------------------------
# Endpoint de predição
# -------------------------
@app.post("/predict")
def predict(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...),
    koi_depth: float = Body(...),
    koi_duration: float = Body(...),
    koi_model_snr: float = Body(...),
    koi_score: float = Body(...),
    host_name: str = Body(None)
):
    # Valida sistema solar
    if host_name and host_name.lower() in ["sol", "sun"]:
        return {
            "prediction": None,
            "message": "This object is from solar system, not an exoplanet!"
        }

    # Regra de falso positivo baseado no score
    if koi_score < 0.3:
        return {"prediction": "FALSE POSITIVE"}

    # Prepara input para o modelo
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad, koi_depth, koi_duration, koi_model_snr, koi_score]])
    X_new_scaled = scaler.transform(X_new)

    # Predição
    pred = model.predict(X_new_scaled)
    class_name = label_map_num2str.get(str(pred[0]), "UNKNOWN")

    return {"prediction": class_name}

# -------------------------
# Endpoint de feedback (treino incremental)
# -------------------------
@app.post("/feedback")
def feedback(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...),
    koi_depth: float = Body(...),
    koi_duration: float = Body(...),
    koi_model_snr: float = Body(...),
    koi_score: float = Body(...),
    real_label: str = Body(...)
):
    # Prepara batch
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad,
                       koi_depth, koi_duration, koi_model_snr, koi_score]])
    X_new_scaled = normalize_with_scaler(scaler, X_new)
    y_new = np.array([real_label.upper()])

    # Atualiza modelo incrementalmente
    model.partial_fit(X_new_scaled, y_new, classes=classes)

    # Salva modelo atualizado
    joblib.dump(model, MODEL_PATH)

    return {"msg": f"Feedback registrado! Modelo atualizado com {real_label.upper()}."}
