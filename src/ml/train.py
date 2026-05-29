import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

def train():
    print("[train] Chargement des donnees...")
    df = pd.read_csv("data/clean/creditcard_clean.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[train] Train : " + str(len(X_train)) + " | Test : " + str(len(X_test)))
    print("[train] Fraudes train : " + str(y_train.sum()) + " | Fraudes test : " + str(y_test.sum()))

    # SMOTE — on rééquilibre les classes
    print("[train] Application SMOTE pour rééquilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print("[train] Apres SMOTE — Fraudes : " + str(y_train_sm.sum()) + " / Total : " + str(len(y_train_sm)))

    # ---- RANDOM FOREST ----
    print("\n[train] Entrainement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sm, y_train_sm)
    y_pred_rf   = rf.predict(X_test)
    y_proba_rf  = rf.predict_proba(X_test)[:, 1]
    auc_rf      = roc_auc_score(y_test, y_proba_rf)
    print("[train] Random Forest — AUC : " + str(round(auc_rf, 4)))
    print(classification_report(y_test, y_pred_rf, target_names=["Normal","Fraude"]))

    # ---- XGBOOST ----
    print("\n[train] Entrainement XGBoost...")
    scale_pos = int(y_train_sm.value_counts()[0] / y_train_sm.value_counts()[1])
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(X_train_sm, y_train_sm)
    y_pred_xgb  = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_xgb     = roc_auc_score(y_test, y_proba_xgb)
    print("[train] XGBoost — AUC : " + str(round(auc_xgb, 4)))
    print(classification_report(y_test, y_pred_xgb, target_names=["Normal","Fraude"]))

    # Meilleur modele
    best_model  = xgb if auc_xgb >= auc_rf else rf
    best_name   = "XGBoost" if auc_xgb >= auc_rf else "RandomForest"
    best_proba  = y_proba_xgb if auc_xgb >= auc_rf else y_proba_rf
    print("\n[train] Meilleur modele : " + best_name + " (AUC=" + str(round(max(auc_xgb, auc_rf), 4)) + ")")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":      best_model,
        "model_name": best_name,
        "features":   list(X.columns),
        "auc":        round(max(auc_xgb, auc_rf), 4),
        "X_test":     X_test,
        "y_test":     y_test,
        "y_proba":    best_proba,
    }, "models/fraud_model.pkl")
    print("[train] Modele sauvegarde -> models/fraud_model.pkl")

    return best_model, X_test, y_test, best_proba

if __name__ == "__main__":
    train()