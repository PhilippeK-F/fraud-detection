import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, f1_score, roc_auc_score)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
from datetime import datetime
import joblib
import shap
import time
import io
import os

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/clean/creditcard_clean.csv")

@st.cache_resource
def load_model():
    if not os.path.exists("models/fraud_model.pkl"):
        return None
    return joblib.load("models/fraud_model.pkl")

st.title("Détection de Fraude Bancaire")
st.caption("284 807 transactions réelles · Dataset Kaggle ULB · Random Forest + XGBoost")

bundle = load_model()
df     = load_data()

if bundle is None:
    st.warning("Lance d abord : python src/ml/train.py")
    st.stop()

model      = bundle["model"]
model_name = bundle["model_name"]
auc        = bundle["auc"]
X_test     = bundle["X_test"]
y_test     = bundle["y_test"]
y_proba    = bundle["y_proba"]
features   = bundle["features"]

fraudes  = df[df["Class"] == 1]
normales = df[df["Class"] == 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions totales",  f"{len(df):,}")
c2.metric("Fraudes détectées",     f"{len(fraudes):,}")
c3.metric("Taux de fraude",        f"{round(len(fraudes)/len(df)*100, 3)}%")
c4.metric("AUC du modèle",         str(auc))

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Performance", "Comparaison modèles", "Analyse des fraudes",
    "Simulateur", "Alerte temps réel", "Rapport PDF"
])

# ---- TAB 1 : PERFORMANCE ----
with tab1:
    st.subheader("Performance — " + model_name)

    # Seuil optimal automatique
    f1_scores = []
    seuils    = np.arange(0.1, 0.9, 0.01)
    for s in seuils:
        y_pred_s = (y_proba >= s).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_s, zero_division=0))
    seuil_optimal = round(seuils[np.argmax(f1_scores)], 2)
    st.info("Seuil optimal calculé automatiquement (F1 max) : **" + str(seuil_optimal) + "**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name="ROC (AUC=" + str(auc) + ")",
                                     line=dict(color="#4C9BE8", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     name="Aléatoire",
                                     line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(xaxis_title="Taux faux positifs",
                              yaxis_title="Taux vrais positifs",
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown("#### Matrice de confusion")
        seuil  = st.slider("Seuil de détection", 0.1, 0.9, seuil_optimal, 0.05)
        y_pred = (y_proba >= seuil).astype(int)
        cm_mat = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm_mat, text_auto=True,
                           labels=dict(x="Prédit", y="Réel"),
                           x=["Normal","Fraude"], y=["Normal","Fraude"],
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)
        vp = cm_mat[1][1]
        fp = cm_mat[0][1]
        fn = cm_mat[1][0]
        st.markdown("**Vrais positifs (fraudes détectées) :** " + str(vp))
        st.markdown("**Faux positifs (fausses alertes) :** " + str(fp))
        st.markdown("**Faux négatifs (fraudes manquées) :** " + str(fn))

    st.markdown("#### Courbe Précision / Rappel")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                name="Precision-Recall",
                                line=dict(color="#E8724C", width=2)))
    fig_pr.update_layout(xaxis_title="Rappel", yaxis_title="Précision",
                         plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pr, use_container_width=True)

# ---- TAB 2 : COMPARAISON MODELES ----
with tab2:
    st.subheader("Comparaison Random Forest vs XGBoost")
    st.caption("Les deux modèles ont été entraînés sur les mêmes données avec SMOTE.")

    metriques = {
        "Modèle":      ["Random Forest", "XGBoost"],
        "AUC":         [0.9805, 0.9739],
        "Recall fraude":[0.80, 0.81],
        "Précision fraude":[0.55, 0.40],
        "F1 fraude":   [0.65, 0.54],
    }
    df_comp = pd.DataFrame(metriques)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_comp = px.bar(
            df_comp.melt(id_vars="Modèle", var_name="Métrique", value_name="Score"),
            x="Métrique", y="Score", color="Modèle", barmode="group",
            color_discrete_map={"Random Forest":"#4C9BE8","XGBoost":"#E8724C"},
            labels={"Score":"Score","Métrique":"Métrique"}
        )
        fig_comp.update_layout(plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0,1])
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_b:
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
        st.markdown("""
**Pourquoi Random Forest est le meilleur ici :**
- AUC plus élevée (0.9805 vs 0.9739)
- Précision fraude bien supérieure (55% vs 40%)
- Moins de fausses alertes — crucial en production
- F1-score fraude meilleur (0.65 vs 0.54)

XGBoost a un recall légèrement meilleur (81% vs 80%)
mais génère trop de fausses alertes pour être utilisé tel quel.
        """)

# ---- TAB 3 : ANALYSE FRAUDES ----
with tab3:
    st.subheader("Analyse des transactions frauduleuses")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Distribution des montants")
        fig_amt = px.histogram(df, x="Amount_scaled", color="Class",
                               nbins=50, barmode="overlay",
                               color_discrete_map={0:"#4C9BE8", 1:"#E8484C"},
                               labels={"Amount_scaled":"Montant normalisé","Class":"Type"},
                               opacity=0.7)
        fig_amt.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_amt, use_container_width=True)

    with col4:
        st.markdown("#### Distribution temporelle des fraudes")
        fig_time = px.histogram(df[df["Class"]==1], x="Time_scaled", nbins=30,
                                color_discrete_sequence=["#E8484C"],
                                labels={"Time_scaled":"Temps normalisé"})
        fig_time.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("#### Importance des features")
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "feature":    features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)
        fig_feat = px.bar(feat_imp, x="importance", y="feature",
                          orientation="h", color="importance",
                          color_continuous_scale="Blues",
                          labels={"importance":"Importance","feature":"Feature"})
        fig_feat.update_layout(coloraxis_showscale=False,
                               yaxis={"categoryorder":"total ascending"},
                               plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_feat, use_container_width=True)

# ---- TAB 4 : SIMULATEUR + SHAP ----
with tab4:
    st.subheader("Simulateur de transaction")
    st.caption("Entrez les caractéristiques d une transaction pour prédire si elle est frauduleuse.")

    col5, col6 = st.columns(2)
    with col5:
        amount = st.number_input("Montant (€)", min_value=0.0, max_value=50000.0, value=100.0)
        heure  = st.number_input("Heure de la transaction (0-172792)", min_value=0, max_value=172792, value=50000)

    st.caption("Les features V1-V28 sont des composantes PCA anonymisées — valeurs par défaut = transaction normale type.")

    if st.button("Analyser cette transaction"):
        sample = pd.DataFrame([{f: 0.0 for f in features}])
        sample["Amount_scaled"] = (amount - 88.35) / 250.12
        sample["Time_scaled"]   = (heure - 94813) / 47488

        proba = model.predict_proba(sample[features])[0][1]
        pred  = "FRAUDE" if proba >= seuil_optimal else "NORMALE"

        if pred == "FRAUDE":
            st.error("Transaction FRAUDULEUSE détectée — Score : " + str(round(proba*100, 1)) + "%")
        else:
            st.success("Transaction NORMALE — Score de fraude : " + str(round(proba*100, 1)) + "%")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba*100, 1),
            title={"text": "Risque de fraude (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#E8484C" if proba >= seuil_optimal else "#4C9BE8"},
                "steps": [
                    {"range": [0,  30], "color": "#E1F5EE"},
                    {"range": [30, 70], "color": "#FAEEDA"},
                    {"range": [70,100], "color": "#FCEBEB"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP — explication de la décision
        st.markdown("#### Explication de la décision (SHAP)")
        st.caption("Les features en rouge ont poussé vers FRAUDE, en bleu vers NORMAL.")
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample[features])
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
            shap_df = pd.DataFrame({
                "feature": features,
                "shap":    sv
            }).sort_values("shap", key=abs, ascending=False).head(10)
            shap_df["couleur"] = shap_df["shap"].apply(lambda x: "Fraude" if x > 0 else "Normal")
            fig_shap = px.bar(shap_df, x="shap", y="feature",
                              orientation="h", color="couleur",
                              color_discrete_map={"Fraude":"#E8484C","Normal":"#4C9BE8"},
                              labels={"shap":"Impact SHAP","feature":"Feature"})
            fig_shap.update_layout(yaxis={"categoryorder":"total ascending"},
                                   plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info("SHAP non disponible pour cette configuration : " + str(e))

# ---- TAB 5 : ALERTE TEMPS REEL ----
with tab5:
    st.subheader("Simulation d alerte en temps réel")
    st.caption("Simule un flux de transactions arrivant en direct — les fraudes déclenchent une alerte.")

    nb_transactions = st.slider("Nombre de transactions à simuler", 10, 100, 30)
    vitesse         = st.slider("Vitesse (secondes entre chaque transaction)", 0.1, 2.0, 0.3)
    demarrer        = st.button("Démarrer la simulation")

    if demarrer:
        sample_data = df.sample(nb_transactions, random_state=42).reset_index(drop=True)
        X_sim       = sample_data[features]
        probas_sim  = model.predict_proba(X_sim)[:, 1]

        placeholder  = st.empty()
        alertes      = []
        transactions = []

        for i in range(nb_transactions):
            proba_i = probas_sim[i]
            est_fraude = proba_i >= seuil_optimal
            transactions.append({
                "#":        i+1,
                "Montant":  round(float(sample_data.loc[i, "Amount_scaled"]), 3),
                "Score":    round(float(proba_i)*100, 1),
                "Statut":   "FRAUDE" if est_fraude else "Normal",
                "Réel":     "FRAUDE" if sample_data.loc[i, "Class"] == 1 else "Normal"
            })
            if est_fraude:
                alertes.append(i+1)

            with placeholder.container():
                st.markdown("**Transaction " + str(i+1) + "/" + str(nb_transactions) + "**")
                df_live = pd.DataFrame(transactions)
                st.dataframe(df_live, hide_index=True, use_container_width=True)
                if alertes:
                    st.error("ALERTES FRAUDE : transactions " + str(alertes))

            time.sleep(vitesse)

        st.success("Simulation terminée — " + str(len(alertes)) + " fraude(s) détectée(s) sur " + str(nb_transactions) + " transactions.")

# ---- TAB 6 : RAPPORT PDF ----
with tab6:
    st.subheader("Rapport d'audit PDF")
    st.caption("Génère un rapport complet avec les métriques, résultats et recommandations.")

    if st.button("Générer le rapport PDF"):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles  = getSampleStyleSheet()
        s_titre = ParagraphStyle("t", parent=styles["Title"], fontSize=18,
                                 textColor=HexColor("#1A1A2E"), spaceAfter=6)
        s_sous  = ParagraphStyle("s", parent=styles["Normal"], fontSize=10,
                                 textColor=HexColor("#666666"), spaceAfter=16)
        s_h2    = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13,
                                 textColor=HexColor("#1A1A2E"), spaceBefore=14, spaceAfter=6)
        s_n     = ParagraphStyle("n", parent=styles["Normal"], fontSize=9,
                                 leading=14, spaceAfter=4)
        elems   = []

        elems.append(Paragraph("Rapport d Audit — Détection de Fraude Bancaire", s_titre))
        elems.append(Paragraph("Généré le " + datetime.now().strftime("%d/%m/%Y à %H:%M") +
                                " · Modèle : " + model_name, s_sous))

        elems.append(Paragraph("Résumé exécutif", s_h2))
        elems.append(Paragraph(
            "Ce rapport présente les résultats du système de détection de fraude bancaire "
            "basé sur le Machine Learning. Le modèle " + model_name + " a été entraîné sur "
            "283 726 transactions réelles avec un taux de fraude de 0.167%.", s_n))

        elems.append(Paragraph("Métriques clés", s_h2))
        data_metriques = [
            ["Métrique", "Valeur"],
            ["Transactions analysées", f"{len(df):,}"],
            ["Fraudes dans le dataset", str(len(fraudes))],
            ["Taux de fraude", f"{round(len(fraudes)/len(df)*100, 3)}%"],
            ["AUC du modèle", str(auc)],
            ["Seuil optimal (F1 max)", str(seuil_optimal)],
            ["Fraudes détectées (test)", str(cm_mat[1][1])],
            ["Fraudes manquées (test)", str(cm_mat[1][0])],
            ["Fausses alertes (test)", str(cm_mat[0][1])],
        ]
        t = Table(data_metriques, colWidths=[10*cm, 6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  HexColor("#1A1A2E")),
            ("TEXTCOLOR",     (0,0), (-1,0),  white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#F5F5F5"), white]),
            ("GRID",          (0,0), (-1,-1), 0.3, HexColor("#CCCCCC")),
            ("ALIGN",         (1,0), (1,-1),  "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        elems.append(t)
        elems.append(Spacer(1, 0.5*cm))

        elems.append(Paragraph("Comparaison des modèles", s_h2))
        data_comp = [
            ["Modèle", "AUC", "Recall fraude", "Précision fraude", "F1 fraude"],
            ["Random Forest", "0.9805", "80%", "55%", "0.65"],
            ["XGBoost",       "0.9739", "81%", "40%", "0.54"],
        ]
        t2 = Table(data_comp, colWidths=[4*cm, 3*cm, 3*cm, 3.5*cm, 2.5*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  HexColor("#1A1A2E")),
            ("TEXTCOLOR",     (0,0), (-1,0),  white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#F5F5F5"), white]),
            ("GRID",          (0,0), (-1,-1), 0.3, HexColor("#CCCCCC")),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        elems.append(t2)
        elems.append(Spacer(1, 0.5*cm))

        elems.append(Paragraph("Recommandations", s_h2))
        elems.append(Paragraph(
            "1. Déployer Random Forest en production avec seuil de " + str(seuil_optimal) + ".", s_n))
        elems.append(Paragraph(
            "2. Mettre en place un système d alerte en temps réel sur le flux de transactions.", s_n))
        elems.append(Paragraph(
            "3. Réentraîner le modèle tous les 3 mois avec les nouvelles données de fraude.", s_n))
        elems.append(Paragraph(
            "4. Investiguer manuellement les transactions avec score entre 0.3 et " + str(seuil_optimal) + ".", s_n))

        doc.build(elems)
        buf.seek(0)

        st.download_button(
            "Télécharger le rapport PDF",
            data=buf,
            file_name="audit_fraude_" + datetime.now().strftime("%Y%m%d") + ".pdf",
            mime="application/pdf"
        )
        st.success("Rapport généré !")