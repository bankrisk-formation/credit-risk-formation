"""
utils/llm_utils.py — BankRisk Intelligence Platform
Fonctions LLM pour l'explicabilite en langage naturel.
Utilise par : J3S2 (Streamlit), tests unitaires.

Dependances : openai>=1.0
Token        : HF_TOKEN dans .env (VS Code) ou st.secrets (Streamlit)
"""
import time
from openai import OpenAI

# Constantes API
HF_MODEL    = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
HF_BASE_URL = "https://router.huggingface.co/v1"


def get_hf_client(token: str) -> OpenAI:
    """Cree un client OpenAI pointant vers le router Hugging Face."""
    return OpenAI(base_url=HF_BASE_URL, api_key=token)


def build_shap_prompt(
    shap_contributions: list,
    decision: str,
    rf_proba: float,
    seuil: float,
) -> str:
    """Construit le prompt LLM a partir des contributions SHAP."""
    contrib_lines = []
    for c in shap_contributions[:6]:
        direction = "augmente" if c["shap_value"] > 0 else "reduit"
        impact    = "fortement" if abs(c["shap_value"]) > 0.05 else "moderement"
        contrib_lines.append(
            f"- {c['feature']} = {c['raw_value']:.3g} "
            f"({direction} {impact} le risque, contribution = {c['shap_value']:+.3f})"
        )

    contrib_text  = "\n".join(contrib_lines)
    decision_noun = "refus" if decision == "REFUS" else "approbation"

    return (
        "Tu es un analyste credit dans une banque ivoirienne (contexte UEMOA/BCEAO).\n"
        "Un modele de scoring a produit la decision suivante :\n\n"
        f"DECISION : {decision} "
        f"(probabilite de defaut = {rf_proba:.1%}, seuil = {seuil:.0%})\n\n"
        f"FACTEURS CLES (contributions SHAP) :\n{contrib_text}\n\n"
        f"Redige une explication de {decision_noun} en 3-4 phrases max. "
        "Mentionner les 2-3 facteurs cles. "
        "Pas de jargon technique. En francais, contexte bancaire ivoirien.\n\n"
        "Reponds uniquement avec l'explication."
    )


def generate_shap_explanation(
    shap_contributions: list,
    decision: str,
    rf_proba: float,
    seuil: float,
    hf_token: str,
    model: str = HF_MODEL,
    base_url: str = HF_BASE_URL,
    max_retries: int = 3,
    retry_delay: float = 8.0,
) -> tuple:
    """Appelle HF Mistral 7B — retourne (explication, None) ou (None, erreur).

    Le router HF Inference Providers (router.huggingface.co) ne supporte pas
    le parametre legacy `wait_for_model` de l'ancienne API d'inference : la
    seule parade cote client contre les erreurs "model is busy" (frequentes
    sur les providers gratuits type featherless-ai, souvent un cold-start
    transitoire) est de reessayer avec un delai.
    """
    if not hf_token:
        return None, "Token HF absent — mode degrade (SHAP seul)"

    client   = OpenAI(base_url=base_url, api_key=hf_token)
    prompt   = build_shap_prompt(shap_contributions, decision, rf_proba, seuil)
    messages = [
        {
            "role":    "system",
            "content": (
                "Tu es un analyste credit senior dans une banque ivoirienne. "
                "Tu rediges des explications de decisions de credit claires "
                "et professionnelles, en francais."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=300, temperature=0.3,
            )
            return response.choices[0].message.content.strip(), None
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return None, f"Erreur API Hugging Face : {str(last_error)}"
