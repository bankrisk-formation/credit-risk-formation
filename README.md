# 🏦 Credit Risk Analysis — BankRisk Intelligence Platform

![Python](https://img.shields.io/badge/python-3.10-blue)
![Pandas](https://img.shields.io/badge/pandas-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green)
![License](https://img.shields.io/badge/license-CC0-lightgrey)
![Status](https://img.shields.io/badge/status-en%20cours-yellow)

Analyse du risque de crédit sur le dataset Kaggle **Credit Risk Dataset**.  
Ce projet fait partie de la formation *BankRisk Intelligence Platform* et vise à construire un pipeline complet d’analyse, de scoring et d’explicabilité autour de la variable cible `loan_status`.

> **Taux de défaut de référence : 21,8 %**  
> 32 581 prêts · 12 variables · Cible binaire `loan_status` (0 = sain, 1 = défaut)

---

## 📖 Description

Ce projet a pour objectif d’explorer, nettoyer et modéliser un jeu de données de risque de crédit afin d’identifier les profils les plus risqués. Il couvre plusieurs étapes clés :

- exploration et analyse descriptive,
- ingénierie des features,
- modélisation supervisée,
- interprétabilité des prédictions,
- mise en place d’un workflow MLOps de base.

---

## 📁 Structure du repository

```text
bankrisk-credit/
├── data/
│   ├── raw/            ← données brutes, non versionnées
│   └── processed/      ← données transformées et features
├── notebooks/          ← notebooks de formation et analyses
├── src/                ← scripts Python réutilisables
├── tests/              ← tests unitaires
├── requirements.txt    ← dépendances Python
└── .gitignore          ← exclusions Git
```

---

## ⚙️ Installation

```bash
git clone https://github.com/FABIENNEY/bankrisk-credit.git
cd bankrisk-credit
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows Git Bash
pip install -r requirements.txt
```

---

## 📊 Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | Kaggle — Credit Risk Dataset |
| Licence | CC0 |
| Lignes | 32 581 |
| Colonnes | 12 |
| Cible | `loan_status` |
| Taux de défaut | **21,8 %** |

> Le fichier CSV doit être placé manuellement dans le dossier `data/raw/` avant l’exécution des notebooks.

---

## 🧰 Stack technique

- Python 3.10
- Pandas
- NumPy
- Plotly
- Scikit-learn
- MLflow
- DagsHub
- Hugging Face
- PySpark
- Delta Lake
- Streamlit
- GitHub Actions

---

## 🗓️ Roadmap sur 3 jours

| Jour | Session | Thème |
|------|---------|-------|
| J1 | S1 | Setup & GitHub Copilot |
| J1 | S2 | Pandas & NumPy |
| J1 | S3 | EDA & Plotly |
| J1 | S4 | Feature Engineering |
| J2 | S1 | K-Means Risk Profiling |
| J2 | S2 | Logistic Regression & Random Forest |
| J2 | S3 | MLflow & DagsHub |
| J2 | S4 | Explicabilité avec SHAP |
| J3 | S1 | Streamlit Dashboard |
| J3 | S2 | PySpark ETL |
| J3 | S3 | Streaming Structuré |
| J3 | S4 | Delta Lake & CI/CD |

---

## 📄 Licence

Le dataset est fourni sous licence **CC0**.  
Le code source de ce projet est proposé sous licence **MIT**.
