#!/usr/bin/env python3
"""
app_v2.py - BICICI BankRisk Intelligence Platform
Interface de scoring credit avec SHAP et commentaire LLM Mistral.
Design system : BICICI (https://www.bicici.ci)

Sequence : J3S1 MLflow -> J3S2 Streamlit -> J3S3 Spark
Modele   : BankRisk-RF-Champion/Production (DagsHub Registry)
LLM      : utils/llm_utils.py (Mistral-7B via HF Inference Providers)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import mlflow.sklearn
import shap
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path

# ── Importer generate_shap_explanation depuis J2S4 ──────────
try:
    from utils.llm_utils import generate_shap_explanation
except ImportError:
    def generate_shap_explanation(shap_contributions, decision, rf_proba, seuil, hf_token, **kwargs):
        return None, "utils/llm_utils.py introuvable. Executer J2S4 d'abord."

# ── Configuration secrets (dual env : local .env / Streamlit Cloud) ──
load_dotenv()

def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

MLFLOW_URI    = _secret("MLFLOW_TRACKING_URI")
MLFLOW_USER   = _secret("MLFLOW_TRACKING_USERNAME")
MLFLOW_TOKEN  = _secret("MLFLOW_TRACKING_PASSWORD")
HF_TOKEN      = _secret("HF_TOKEN")
DEFAULT_THRESHOLD = float(_secret("THRESHOLD", "0.42"))

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TOKEN

# ── Charte graphique BICICI (https://www.bicici.ci) ──────
BICICI_GREEN       = "#005E42"   # vert principal (logo, titres, boutons)
BICICI_GREEN_MID   = "#0D8A5F"   # vert secondaire (degrade)
BICICI_GREEN_LIGHT = "#7FF08E"   # vert clair (accents)
BICICI_GREEN_TINT  = "#E8FCEB"   # fond vert tres clair (cartes)
BICICI_TEXT        = "#1E1E1E"
BICICI_GRAY        = "#757575"
BICICI_GRAY_LIGHT  = "#D9D9D9"
BICICI_RED         = "#D64545"   # rouge risque (refus)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH  = ASSETS_DIR / "bicici_logo.png"
ICON_PATH  = ASSETS_DIR / "bicici_icon.png"

LOAN_INTENTS = [
    "PERSONAL", "EDUCATION", "MEDICAL",
    "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
]

# ── Charger le modele depuis le Registry (cache) ──────────
@st.cache_resource
def load_rf_model():
    try:
        return mlflow.sklearn.load_model("models:/BankRisk-RF-Champion/Production")
    except Exception as e:
        st.error(f"Erreur chargement modele : {e}")
        return None

@st.cache_resource
def load_explainer(_model, n_background=200):
    """Initialise le SHAP TreeExplainer (calcule une seule fois).

    shap_background_sample.parquet (echantillon leger, commite sur GitHub) est
    priorise pour fonctionner sur les deploiements distants (Streamlit Cloud) ou
    data/processed/*.parquet (donnees completes) est absent du repo (.gitignore).
    """
    try:
        features = list(_model.feature_names_in_)
        root = Path(__file__).resolve().parent
        for p in [root / "data" / "processed" / "shap_background_sample.parquet",
                  root / "data" / "processed" / "credit_risk_kmeans.parquet",
                  root / "data" / "processed" / "credit_risk_clean.parquet"]:
            if p.exists():
                df = pd.read_parquet(p)
                feats = [f for f in features if f in df.columns]
                X_bg = df[feats].sample(min(n_background, len(df)), random_state=42)
                scaler = _model.named_steps["scaler"]
                rf     = _model.named_steps["model"]
                return shap.TreeExplainer(rf), scaler, feats
    except Exception:
        pass
    return None, None, []

# ── build_features ─────────────────────────────────────────
def build_features(income, loan_amnt, loan_int_rate, home,
                   intent, emp_length, default_hist, age, cred_hist, features):
    """Transforme les inputs sidebar en vecteur compatible avec le pipeline J1S4.

    `features` = model.feature_names_in_ (source de verite) : le reindex
    elimine silencieusement toute colonne construite ici mais absente du fit
    (ex. l'ancien "default_enc", jamais vu a l'entrainement), au lieu de
    lever "feature names unseen at fit time".
    """
    lpi = loan_amnt / income if income > 0 else 0.0
    intent_categories = sorted(LOAN_INTENTS)  # ordre du LabelEncoder (alphabetique) a l'entrainement
    return pd.DataFrame([{
        "person_income":              income,
        "person_emp_length":          emp_length,
        "person_age":                 age,
        "loan_amnt":                  loan_amnt,
        "loan_int_rate":              loan_int_rate,
        "loan_percent_income":        lpi,
        "cb_person_cred_hist_length": cred_hist,
        "home_RENT":     1 if home == "RENT"     else 0,
        "home_MORTGAGE": 1 if home == "MORTGAGE" else 0,
        "home_OWN":      1 if home == "OWN"      else 0,
        "loan_intent_enc":      intent_categories.index(intent) if intent in intent_categories else 0,
        "debt_service_rate":          loan_int_rate * lpi,
        "monthly_payment_proxy":      loan_amnt / (income / 12) if income > 0 else 0.0,
        "log_income":                 np.log1p(income),
        "high_risk_intent":           1 if intent in ("DEBTCONSOLIDATION", "MEDICAL") else 0,
    }]).reindex(columns=features, fill_value=0)

# ── SHAP values (compat multi-versions shap) ───────────────
def get_shap_values(explainer, scaler, X_input):
    """Extrait les valeurs SHAP de la classe 1 (defaut) pour un seul dossier.

    shap.TreeExplainer.shap_values() ne retourne pas le meme format selon
    la version de shap installee :
      - anciennes versions : liste [sv_classe_0, sv_classe_1]
      - shap recent (0.45+) : array 3D (n_samples, n_features, n_classes)
      - sinon : array 2D (n_samples, n_features) deja pour la classe positive
    """
    X_sc = scaler.transform(X_input)
    sv = explainer.shap_values(X_sc)
    if isinstance(sv, list):
        return sv[1][0]
    if sv.ndim == 3:
        return sv[0, :, 1]
    return sv[0]

# ── SHAP waterfall Plotly ───────────────────────────────────
def shap_waterfall(shap_vals, feature_names, n_top=8):
    """Graphique en barres horizontales des contributions SHAP."""
    pairs = sorted(zip(feature_names, shap_vals),
                   key=lambda x: abs(x[1]), reverse=True)[:n_top]
    names = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=[BICICI_RED if v > 0 else BICICI_GREEN for v in vals],
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Facteurs SHAP - Contribution au score de defaut",
        xaxis_title="Impact (rouge = risque hausse, vert = risque baisse)",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=BICICI_TEXT), height=320,
        xaxis=dict(gridcolor=BICICI_GRAY_LIGHT),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160),
    )
    return fig

# ── Stat tile BICICI (barre coloree + valeur + delta inline) ──
def stat_tile(label, value, delta=None, status="neutral"):
    """Composant KPI compact : barre laterale coloree, valeur en gras, delta inline."""
    color = {"good": BICICI_GREEN, "bad": BICICI_RED, "neutral": BICICI_GRAY_LIGHT}[status]
    delta_html = f'<span class="bicici-stat-delta">{delta}</span>' if delta else ""
    st.markdown(
        f'<div class="bicici-stat-tile" style="--tile-color:{color};">'
        f'<div class="bicici-stat-label">{label}</div>'
        f'<div class="bicici-stat-value">{value}{delta_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Grille de lecture du score ──────────────────────────────
def score_legend_table(threshold, proba):
    """Table de lecture du score : bandes de risque relatives au seuil actif."""
    bands = [
        (0.0, threshold * 0.5, "Risque faible", "GO", BICICI_GREEN),
        (threshold * 0.5, threshold, "Risque modere", "GO (sous surveillance)", BICICI_GREEN_MID),
        (threshold, min(threshold + 0.20, 1.0), "Risque eleve", "NO-GO", BICICI_RED),
        (min(threshold + 0.20, 1.0), 1.0, "Risque tres eleve", "NO-GO immediat", BICICI_RED),
    ]
    rows_html = ""
    for lo, hi, label, dec, color in bands:
        active = lo <= proba < hi or (hi >= 1.0 and proba >= hi)
        bg     = f"{color}1F" if active else "transparent"
        weight = "700" if active else "500"
        marker = "→ " if active else "&nbsp;&nbsp;"
        rows_html += (
            f'<tr style="background:{bg}; font-weight:{weight};">'
            f'<td style="padding:6px 12px; border-bottom:1px solid {BICICI_GRAY_LIGHT};">{marker}{lo:.0%} – {hi:.0%}</td>'
            f'<td style="padding:6px 12px; border-bottom:1px solid {BICICI_GRAY_LIGHT}; color:{color};">{label}</td>'
            f'<td style="padding:6px 12px; border-bottom:1px solid {BICICI_GRAY_LIGHT};">{dec}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">'
        f'<thead><tr style="border-bottom:2px solid {BICICI_GREEN};">'
        f'<th style="text-align:left; padding:6px 12px; color:{BICICI_GREEN};">Score de defaut</th>'
        f'<th style="text-align:left; padding:6px 12px; color:{BICICI_GREEN};">Niveau de risque</th>'
        f'<th style="text-align:left; padding:6px 12px; color:{BICICI_GREEN};">Decision</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BICICI — BankRisk Scoring",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🏦",
    layout="wide",
)

# Logo BICICI (coin superieur gauche + sidebar, natif Streamlit)
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), size="large",
            icon_image=str(ICON_PATH) if ICON_PATH.exists() else None)

# CSS — theme BICICI (charte graphique https://www.bicici.ci)
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Poppins', Arial, sans-serif; }}

.stApp {{ background-color: #F4F5F7; }}
[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: none;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}}
.stApp, [data-testid="stSidebar"] {{ color: {BICICI_TEXT}; }}
h1, h2, h3, h4 {{ color: {BICICI_GREEN} !important; font-weight: 700 !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {BICICI_GRAY} !important; }}

/* Sections en cartes blanches arrondies (fond gris de la page en arriere-plan) */
.st-key-kpi_card_0, .st-key-kpi_card_1, .st-key-kpi_card_2, .st-key-kpi_card_3,
.st-key-card_gauge, .st-key-card_shap, .st-key-card_llm {{
    background: #FFFFFF;
    border: 1px solid #ECECEC;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}}

.bicici-stat-tile {{
    border-left: 4px solid var(--tile-color, {BICICI_GRAY_LIGHT});
    padding: 2px 0 2px 14px;
    margin-bottom: 4px;
}}
.bicici-stat-label {{
    font-size: 0.85rem; font-weight: 600; color: {BICICI_TEXT};
}}
.bicici-stat-value {{
    font-size: 1.9rem; font-weight: 700; color: {BICICI_TEXT}; line-height: 1.3;
}}
.bicici-stat-delta {{
    font-size: 0.85rem; font-weight: 500; color: {BICICI_GRAY}; margin-left: 8px;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600; }}

[data-testid="stHorizontalBlock"] {{ gap: 1rem; align-items: stretch; }}
hr {{ border-color: {BICICI_GREEN_LIGHT} !important; opacity: 0.8; }}

[data-testid="stBaseButton-primary"] {{
    background-color: {BICICI_GREEN} !important;
    border-color: {BICICI_GREEN} !important;
    border-radius: 8px !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
    background-color: {BICICI_GREEN_MID} !important;
    border-color: {BICICI_GREEN_MID} !important;
}}
</style>""", unsafe_allow_html=True)

# Chargement
if not MLFLOW_URI:
    st.error(
        "MLFLOW_TRACKING_URI manquant ou vide — mlflow utilise un registre local "
        "vide au lieu de DagsHub, d'ou l'erreur 'Registered Model not found'.\n\n"
        "**Sur Streamlit Community Cloud** : le fichier `.env` local n'est jamais "
        "deploye. Configurer les secrets separement dans "
        "*App settings → Secrets* (memes cles que `.streamlit/secrets.toml` : "
        "`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, "
        "`MLFLOW_TRACKING_PASSWORD`, `HF_TOKEN`), puis **redemarrer l'app** "
        "(menu ⋮ → Reboot app)."
    )
    st.stop()

model = load_rf_model()
if model is None:
    st.error("Modele non disponible. Verifier la connexion DagsHub (J3S1).")
    st.stop()

explainer, scaler_shap, feat_for_shap = load_explainer(model)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.title("BankRisk Scoring")
    st.caption("BICICI · Contexte bancaire ivoirien")
    st.markdown("---")
    st.subheader("Profil emprunteur")

    income        = st.slider("Revenu annuel ($)", 10000, 250000, 60000, 1000)
    loan_amnt     = st.slider("Montant du pret ($)", 500, 35000, 10000, 500)
    loan_int_rate = st.number_input("Taux interet (%)", 5.0, 30.0, 11.5, 0.1)
    home          = st.selectbox("Type de logement", ["RENT", "MORTGAGE", "OWN"])
    intent        = st.selectbox("Objet du pret", LOAN_INTENTS)
    emp_length    = st.slider("Duree emploi (ans)", 0.0, 20.0, 4.0, 0.5)
    default_hist  = st.selectbox("Defaut passe", ["N", "Y"],
                                 format_func=lambda x: "Non" if x == "N" else "Oui")
    age           = st.slider("Age", 20, 65, 30)
    cred_hist     = st.slider("Historique credit (ans)", 0, 30, 5)

    st.markdown("---")
    st.subheader("Politique de risque")
    threshold = st.slider("Seuil de decision t", 0.30, 0.65, DEFAULT_THRESHOLD, 0.01,
                          help="t=0.42 retenu (BCEAO 026-2016)")

# ── SCORING ──────────────────────────────────────────────
X_input  = build_features(income, loan_amnt, loan_int_rate, home,
                           intent, emp_length, default_hist, age, cred_hist,
                           features=list(model.feature_names_in_))
proba    = model.predict_proba(X_input)[0, 1]
decision = proba >= threshold
lpi      = loan_amnt / income if income > 0 else 0.0
dsr      = loan_int_rate * lpi

# ── AFFICHAGE ─────────────────────────────────────────────
st.title("Resultat du Scoring BankRisk")
st.caption(f"RF Baseline - AUC=0.929 - Seuil t={threshold:.2f}")

col1, col2, col3, col4 = st.columns(4)
with col1, st.container(key="kpi_card_0"):
    stat_tile("Probabilite de defaut", f"{proba:.1%}",
              f"{(proba - threshold) * 100:+.0f} pts vs seuil",
              status="bad" if decision else "good")
with col2, st.container(key="kpi_card_1"):
    stat_tile("Decision", "NO-GO" if decision else "GO",
              f"seuil t={threshold:.0%}",
              status="bad" if decision else "good")
with col3, st.container(key="kpi_card_2"):
    stat_tile("Charge / Revenu (LPI)", f"{lpi:.1%}",
              f"{(lpi - 0.35) * 100:+.0f} pts vs 35% max",
              status="bad" if lpi > 0.35 else "good")
with col4, st.container(key="kpi_card_3"):
    stat_tile("Risque combine (DSR)", f"{dsr:.3f}", status="neutral")

st.write("")

with st.container(key="card_gauge"):
    # Jauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": "Score de defaut (%)", "font": {"color": BICICI_TEXT, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": BICICI_GRAY},
            "bar": {"color": BICICI_RED if decision else BICICI_GREEN},
            "steps": [
                {"range": [0, threshold*100], "color": BICICI_GREEN_TINT},
                {"range": [threshold*100, 100], "color": "#FBE6E6"},
            ],
            "threshold": {"line": {"color": BICICI_RED, "width": 3}, "value": threshold*100},
        },
        number={"font": {"color": BICICI_TEXT}, "suffix": " %"},
    ))
    fig_gauge.update_layout(height=220, paper_bgcolor="#FFFFFF",
                            font=dict(color=BICICI_TEXT), margin=dict(t=30, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Grille de lecture du score")
    score_legend_table(threshold, proba)

st.write("")

# ── SHAP + LLM ──────────────────────────────────────────────
col_shap, col_llm = st.columns([3, 2])
shap_top3 = []
shap_contributions = []

with col_shap, st.container(key="card_shap"):
    st.subheader("Explication SHAP - Facteurs determinants")
    if explainer is not None and scaler_shap is not None:
        try:
            sv_arr = get_shap_values(explainer, scaler_shap, X_input)
            shap_sorted = sorted(zip(feat_for_shap, sv_arr),
                                 key=lambda x: abs(x[1]), reverse=True)
            shap_top3 = shap_sorted[:3]
            shap_contributions = [
                {"feature": f, "shap_value": float(v), "raw_value": float(X_input.iloc[0][f])}
                for f, v in shap_sorted
            ]
            st.plotly_chart(shap_waterfall(sv_arr, feat_for_shap),
                           use_container_width=True)
            st.caption("Top 3 : " + " | ".join([f"{f} ({v:+.3f})" for f, v in shap_top3]))
        except Exception as e:
            st.warning(f"SHAP non disponible : {e}")
    else:
        st.info("Parquet J1S4 requis pour SHAP (credit_risk_clean.parquet).")

with col_llm, st.container(key="card_llm"):
    st.subheader("Commentaire LLM (Mistral)")
    st.caption("Mistral-7B-Instruct via Hugging Face Inference Providers")
    if st.button("Generer commentaire", type="primary", use_container_width=True):
        decision_label = "REFUS" if decision else "APPROBATION"
        with st.spinner("Mistral analyse le dossier..."):
            comment, erreur = generate_shap_explanation(
                shap_contributions=shap_contributions,
                decision=decision_label,
                rf_proba=proba,
                seuil=threshold,
                hf_token=HF_TOKEN,
            )
        if erreur:
            st.warning(f"Mode degrade (SHAP seul) : {erreur}")
        elif decision:
            st.error(f"**Avis - NO-GO**\n\n{comment}")
        else:
            st.success(f"**Avis - GO**\n\n{comment}")

    st.markdown("---")
    st.subheader("Recapitulatif dossier")
    for k, v in [("Revenu", f"{income:,} $"), ("Pret", f"{loan_amnt:,} $"),
                 ("Taux", f"{loan_int_rate:.1f} %"), ("LPI", f"{lpi:.1%}"),
                 ("Logement", home), ("Objet", intent),
                 ("Defaut passe", "Oui" if default_hist == "Y" else "Non")]:
        st.write(f"**{k}** : {v}")

st.write("")
st.divider()
st.caption("BICICI · BankRisk Intelligence Platform · RF Baseline · AUC=0.929 · "
           "Pipeline J1S4->J3S1->J3S2 · Instruction BCEAO n 026-2016")
