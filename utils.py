import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# Funções de manipulação de dados
# ---------------------------

def load_dataset(path):
    """
    Carrega CSV e retorna DataFrame
    """
    return pd.read_csv(path)


def split_features_target(df, features, target):
    """
    Separa DataFrame em X e y
    """
    X = df[features]
    y = df[target]
    return X, y


# ---------------------------
# Funções de pré-processamento
# ---------------------------

def create_scaler(X):
    """
    Cria e ajusta StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def normalize_with_scaler(scaler, X):
    """
    Normaliza novas features com scaler já treinado
    """
    return scaler.transform(X)


# ---------------------------
# Funções de balanceamento de classes
# ---------------------------

def compute_class_weights(y, classes):
    """
    Calcula class_weight manual para SGDClassifier incremental
    """
    classes_np = np.array(classes)
    weights_values = compute_class_weight(
        class_weight='balanced',
        classes=classes_np,
        y=y
    )
    class_weights = dict(zip(classes_np, weights_values))
    return class_weights
