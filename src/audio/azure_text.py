"""Análise do texto transcrito com Azure AI Language (Text Analytics).

- Análise de sentimento com mineração de opinião (pt-BR);
- Extração de frases-chave;
- Text Analytics for Health: entidades clínicas (sintomas, partes do corpo,
  tempo, gravidade) e suas relações/asserções (negação, certeza);
- Léxico clínico de termos críticos em português (sempre aplicado, permite
  mapear termos para síndromes de alerta: respiratória, cardíaca, neurológica/AVC,
  fadiga).
"""
from __future__ import annotations

import re
import time
import unicodedata

from src.config import AZURE

# termo (regex sem acento) -> (categoria, peso)
CRITICAL_TERMS = {
    r"falta de ar|nao consigo respirar|sem ar|sufoc|dispneia|respirar direito": ("respiratoria", 0.9),
    r"chiado|tosse|catarro": ("respiratoria", 0.5),
    r"dor no peito|aperto no peito|aperto e uma dor no peito|dor toracica": ("cardiaca", 0.9),
    r"palpitac|coracao acelerado|coracao disparado": ("cardiaca", 0.6),
    r"fala enrolada|fala arrastada|nao consigo falar|boca (meio )?torta|rosto torto": ("neurologica_avc", 1.0),
    r"braco [a-z\s]{0,30}?fraco|perna [a-z\s]{0,30}?fraca|formigando|formigamento|dormencia|dormente": ("neurologica_avc", 0.8),
    r"tontura|desmai|confus|visao turva|visao dupla": ("neurologica_avc", 0.5),
    r"cansad|cansaco|exaust|esgotament|fraqueza|sem forcas|sem energia|sonolen": ("fadiga", 0.6),
    r"sem vontade de comer|sem apetite|inapetencia": ("fadiga", 0.3),
    r"sangr|vomit|febre|convuls": ("outros_sinais_alarme", 0.7),
    r"medo|ansios|desesper|angusti": ("sofrimento_emocional", 0.4),
}
POSITIVE = ["melhor", "bem", "pronto", "diminuiu", "normalmente", "consegui", "otimo", "tranquilo"]
NEGATIVE = ["dor", "cansad", "fraco", "fraqueza", "medo", "piorou", "nao consigo", "exausto", "falta", "tontura", "torta"]


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def critical_terms(text: str) -> list[dict]:
    t = _norm(text)
    found = []
    for pat, (cat, w) in CRITICAL_TERMS.items():
        for m in re.finditer(pat, t):
            end = m.end()
            while end < len(t) and t[end].isalpha():  # estende até o fim da palavra ("cansad" -> "cansado")
                end += 1
            found.append({"termo": text[m.start():end], "categoria": cat, "peso": w})
    return found


def _client():
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    return TextAnalyticsClient(endpoint=AZURE.language_endpoint, credential=AzureKeyCredential(AZURE.language_key))


def analyze_azure(text: str) -> dict:
    client = _client()
    out: dict = {"provider": "Azure AI Language (Text Analytics)", "errors": []}
    t0 = time.time()
    docs = [{"id": "1", "language": "pt-BR", "text": text}]
    try:
        s = client.analyze_sentiment(docs, show_opinion_mining=True)[0]
        if not s.is_error:
            out["sentimento"] = {
                "geral": s.sentiment,
                "scores": {"positivo": s.confidence_scores.positive, "neutro": s.confidence_scores.neutral,
                           "negativo": s.confidence_scores.negative},
                "frases": [{"texto": x.text, "sentimento": x.sentiment,
                            "negativo": round(x.confidence_scores.negative, 3),
                            "opinioes": [{"alvo": o.target.text, "sentimento": o.target.sentiment,
                                          "avaliacoes": [a.text for a in o.assessments]} for o in x.mined_opinions]}
                           for x in s.sentences],
            }
        else:
            out["errors"].append(f"sentiment: {s.error}")
    except Exception as exc:
        out["errors"].append(f"sentiment: {exc}")
    try:
        k = client.extract_key_phrases(docs)[0]
        out["frases_chave"] = list(k.key_phrases) if not k.is_error else []
    except Exception as exc:
        out["errors"].append(f"key_phrases: {exc}")
    try:
        poller = client.begin_analyze_healthcare_entities([{"id": "1", "language": "pt", "text": text}])
        res = [d for d in poller.result() if not d.is_error]
        ents = []
        for d in res:
            for e in d.entities:
                ents.append({"texto": e.text, "categoria": e.category, "confianca": round(e.confidence_score, 3),
                             "negado": bool(e.assertion and e.assertion.certainty and "negative" in str(e.assertion.certainty)),
                             "links": [l.id for l in (e.data_sources or [])][:3]})
        out["entidades_saude"] = ents
    except Exception as exc:
        out["errors"].append(f"health: {exc}")
    out["latency_s"] = round(time.time() - t0, 2)
    return out


def analyze_local(text: str) -> dict:
    """Fallback léxico (sem Azure)."""
    t = _norm(text)
    pos = sum(t.count(w) for w in POSITIVE)
    neg = sum(t.count(w) for w in NEGATIVE)
    tot = max(pos + neg, 1)
    geral = "negative" if neg > pos else ("positive" if pos > neg else "neutral")
    return {"provider": "Léxico local (fallback)",
            "sentimento": {"geral": geral, "scores": {"positivo": pos / tot, "neutro": 0.0, "negativo": neg / tot},
                           "frases": []},
            "frases_chave": [], "entidades_saude": [], "errors": [], "latency_s": 0.0}


def analyze(text: str) -> dict:
    res = analyze_azure(text) if AZURE.language_enabled else analyze_local(text)
    res["termos_criticos"] = critical_terms(text)
    return res
