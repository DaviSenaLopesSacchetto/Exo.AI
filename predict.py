from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

# -------------------------
# Caminhos dos arquivos
# -------------------------
MODEL_PATH = "models/random_forest_model.pkl"
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
classes = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

# -------------------------
# Carrega modelo e scaler (obrigatórios)
# -------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError("Modelo e scaler não encontrados. Treine primeiro com train.py.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------
# Endpoint principal
# -------------------------
@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas (Random Forest) pronta 🚀"}

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
    host_name: str = Body(None)
):
    # 🚫 Validação: sistema solar
    if host_name and host_name.lower() in ["sol", "sun"]:
        return {
            "prediction": None,
            "message": "This object is from the Solar System, not an exoplanet!"
        }

    # 🔢 Monta o vetor de entrada
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad,
                       koi_depth, koi_duration, koi_model_snr]])

    # 🔄 Normaliza os dados
    X_new_scaled = scaler.transform(X_new)

    # 🤖 Faz a predição
    pred = model.predict(X_new_scaled)[0]

    # 🔤 Retorna nome da classe
    return {"prediction": str(pred)}

# -------------------------
# Endpoint de feedback (agora só registra, não treina)
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
    real_label: str = Body(...)
):
    # ⚠️ Agora não atualiza o modelo — apenas salva feedback
    feedback_data = {
        "koi_prad": koi_prad,
        "koi_period": koi_period,
        "koi_steff": koi_steff,
        "koi_srad": koi_srad,
        "koi_depth": koi_depth,
        "koi_duration": koi_duration,
        "koi_model_snr": koi_model_snr,
        "real_label": real_label
    }

    # Salva feedback localmente
    os.makedirs("feedbacks", exist_ok=True)
    with open("feedbacks/feedback_log.csv", "a") as f:
        f.write(",".join(map(str, feedback_data.values())) + "\n")

    return {"msg": f"Feedback registrado para {real_label.upper()}. Modelo não atualizado automaticamente."}
