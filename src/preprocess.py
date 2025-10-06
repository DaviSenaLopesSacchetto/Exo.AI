import pandas as pd
from sklearn.utils import resample
import os

# Caminho do arquivo raw
RAW_PATH = "data/raw/cumulative.csv"
PROCESSED_PATH = "data/processed/exoplanets_balanced.csv"

# Criar pasta processed se não existir
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# Carregar dataset raw
df = pd.read_csv(RAW_PATH)

# Features usadas no modelo
features = [
    "koi_prad", "koi_period", "koi_steff", "koi_srad",
    "koi_depth", "koi_duration", "koi_model_snr"
]
target = "koi_disposition"


# Filtrar colunas e remover linhas com valores faltantes
df = df[features + [target]].dropna()

# Filtrar apenas classes que nos interessam
df = df[df[target].isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])]

# Separar por classe
df_confirmed = df[df[target] == "CONFIRMED"]
df_candidate = df[df[target] == "CANDIDATE"]
df_fp = df[df[target] == "FALSE POSITIVE"]

# Balancear classes pelo tamanho da menor
min_size = min(len(df_confirmed), len(df_candidate), len(df_fp))

df_confirmed_bal = resample(df_confirmed, n_samples=min_size, random_state=42)
df_candidate_bal = resample(df_candidate, n_samples=min_size, random_state=42)
df_fp_bal = resample(df_fp, n_samples=min_size, random_state=42)

# Concatenar e embaralhar
df_balanced = pd.concat([df_confirmed_bal, df_candidate_bal, df_fp_bal]).sample(frac=1, random_state=42)

print(df_balanced[target].value_counts())

# Salvar dataset processado
df_balanced.to_csv(PROCESSED_PATH, index=False)

print("Pré-processamento concluído!")
print("Distribuição das classes:")
print(df_balanced[target].value_counts())
