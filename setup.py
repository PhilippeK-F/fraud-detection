import os

readme = """# Fraud Detection — Détection de Fraude Bancaire

Système de détection de fraude bancaire par Machine Learning,
entraîné sur 284 807 transactions réelles anonymisées.

---

## Pourquoi ce projet ?

La fraude bancaire représente des milliards d euros de pertes chaque année.
Le défi principal : les fraudes représentent seulement 0.17% des transactions.
Un modèle naïf qui dit "tout est normal" aurait 99.83% de précision — et serait
totalement inutile. Ce projet montre comment traiter ce problème de classes
déséquilibrées et construire un système réellement efficace.

---

## Le problème des classes déséquilibrées

Sur 284 807 transactions :
- 283 253 transactions normales (99.83%)
- 492 transactions frauduleuses (0.17%)

Un modèle qui prédit "normal" à chaque fois aurait une accuracy de 99.83%
mais ne détecterait aucune fraude. C'est pourquoi on utilise :

- SMOTE (Synthetic Minority Oversampling Technique) pour rééquilibrer
  artificiellement les classes à l'entraînement
- AUC-ROC comme métrique principale plutôt que l'accuracy
- Le Recall comme métrique critique : mieux vaut une fausse alerte
  qu'une fraude manquée

---

## Résultats

    Modèle retenu      : Random Forest
    AUC                : 0.9805
    Recall fraudes     : 80% (le modèle détecte 8 fraudes sur 10)
    Précision fraudes  : 55% (sur 100 alertes, 55 sont de vraies fraudes)
    F1-score fraudes   : 0.65
    Seuil optimal      : 0.74 (calculé automatiquement par maximisation du F1)

    Comparaison :
    Random Forest  AUC=0.9805  Recall=80%  Précision=55%  F1=0.65  ← retenu
    XGBoost        AUC=0.9739  Recall=81%  Précision=40%  F1=0.54

    Random Forest est retenu car il génère moins de fausses alertes,
    ce qui est critique en production bancaire.

---

## Stack technique

Python · scikit-learn · XGBoost · imbalanced-learn (SMOTE) ·
SHAP · Streamlit · Plotly · ReportLab · Docker

---

## Architecture

    src/pipelines/
        extract.py      Téléchargement dataset Kaggle (284 807 transactions)
        transform.py    Nettoyage + normalisation Amount et Time
        analyze.py      Statistiques descriptives + EDA

    src/ml/
        train.py        SMOTE + Random Forest + XGBoost + évaluation
        predict.py      Scoring de nouvelles transactions

    dashboard/
        app.py          Dashboard Streamlit — 6 onglets

    models/
        fraud_model.pkl Modèle entraîné + métadonnées

---

## Dashboard — 6 onglets

    1. Performance      Courbe ROC, matrice de confusion, seuil optimal automatique
    2. Comparaison      Random Forest vs XGBoost côte à côte
    3. Analyse fraudes  Distribution montants, temporelle, importance features
    4. Simulateur       Prédiction en temps réel + explication SHAP
    5. Alerte temps réel Flux de transactions simulé avec alertes dynamiques
    6. Rapport PDF      Export complet métriques + recommandations

---

## Lancement rapide

    pip install -r requirements.txt
    python src/pipelines/extract.py
    python src/pipelines/transform.py
    python src/ml/train.py
    streamlit run dashboard/app.py

## Avec Docker

    docker compose up --build

## Dataset

Source : Kaggle — ULB Machine Learning Group
URL    : kaggle.com/datasets/mlg-ulb/creditcardfraud
Période: Transactions européennes — septembre 2013
Features: 28 composantes PCA anonymisées (V1-V28) + Time + Amount

---

## Ce que ce projet démontre

- Gestion des classes déséquilibrées (SMOTE, class_weight)
- Évaluation correcte d un modèle ML (AUC, Recall, Précision, F1)
- Interprétabilité des décisions (SHAP values)
- Pipeline ML complet de la donnée brute au dashboard production
- Comparaison et sélection objective de modèles

---

## Auteur

Data Engineer / ML Engineer passionné par la donnée industrielle et financière.
Portfolio : github.com/TON_USERNAME
"""

dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

dockercompose = """version: '3.8'
services:
  fraud-detection:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
"""

gitignore = """.env
venv/
.venv/
__pycache__/
*.pyc
data/raw/*.csv
data/clean/*.csv
models/*.pkl
.DS_Store
*.log
"""

requirements = """pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.0.3
imbalanced-learn==0.12.3
shap==0.45.0
streamlit==1.35.0
plotly==5.22.0
joblib==1.4.2
kaggle==1.6.12
python-dotenv==1.0.1
reportlab==4.2.0
"""

open('README.md',          'w', encoding='utf-8').write(readme)
open('Dockerfile',         'w', encoding='utf-8').write(dockerfile)
open('docker-compose.yml', 'w', encoding='utf-8').write(dockercompose)
open('.gitignore',         'w', encoding='utf-8').write(gitignore)
open('requirements.txt',   'w', encoding='utf-8').write(requirements)
print('Tous les fichiers OK')