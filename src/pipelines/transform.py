import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def transform(df=None):
    if df is None:
        df = pd.read_csv("data/raw/creditcard.csv")

    print("[transform] " + str(len(df)) + " transactions chargees")

    # Nettoyage
    df = df.drop_duplicates()
    df = df.dropna()
    print("[transform] Apres nettoyage : " + str(len(df)) + " transactions")

    # Normalisation Time et Amount
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Time","Amount"])

    # Stats sur les fraudes
    fraudes    = df[df["Class"] == 1]
    normales   = df[df["Class"] == 0]
    print("[transform] Transactions normales : " + str(len(normales)))
    print("[transform] Transactions frauduleuses : " + str(len(fraudes)))
    print("[transform] Montant moyen fraude : " + str(round(fraudes["Amount_scaled"].mean(), 3)))
    print("[transform] Montant moyen normal : " + str(round(normales["Amount_scaled"].mean(), 3)))

    os.makedirs("data/clean", exist_ok=True)
    df.to_csv("data/clean/creditcard_clean.csv", index=False)
    print("[transform] Fichier sauvegarde -> data/clean/creditcard_clean.csv")
    return df

if __name__ == "__main__":
    transform()