import os
import pandas
from sklearn.utils import resample
import matplotlib.pyplot as plt

df = pandas.read_csv("data/raw/cumulative.csv")
print(df["koi_disposition"].value_counts())

features = {
    "koi_prad",
    "koi_period",
    "koi_steff",
    "koi_srad",
    "koi_depth",
    "koi_duration",
    "koi_model_snr",
    "koi_score"
}

target = "koi_disposition"

features = list(features)

df = df[features + [target]].dropna()
print(df[target].value_counts())

min_size = df[target].value_counts().min()
df_balanced = pandas.concat([
    resample(
        group,          # subconjunto da classe
        replace=False,  # não repete exemplos
        n_samples=min_size,  # número igual ao da menor classe
        random_state=42
    )
    for _, group in df.groupby(target)
])

print(df_balanced[target].value_counts())



df_balanced.to_csv("data/processed/exoplanets_balanced.csv", index=False)
print("Dataset balanceado salvo com sucesso!")
