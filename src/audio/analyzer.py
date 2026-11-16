"""Análise de áudio de consultas: fusão de sinais acústicos e semânticos.

Fluxo:  WAV ──► Azure Speech to Text (transcrição + tempo de cada palavra)
            │        └─► Azure Text Analytics (sentimento, frases-chave, entidades de saúde)
            │        └─► léxico de termos críticos (PT-BR)
            ├─► biomarcadores acústicos (Praat/librosa) vs. linha de base
            └─► AST/AudioSet (sons respiratórios)
        ──► indicadores (fadiga, dificuldade respiratória, disartria) + síndromes de alerta
"""
from __future__ import annotations

from pathlib import Path

from src.audio import acoustic, azure_speech, azure_text
from src.audio.audioset import PT as AUDIOSET_PT

_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        from src.audio.audioset import AudioSetTagger
        _TAGGER = AudioSetTagger()
    return _TAGGER


def analyze_audio(path: str | Path, baseline: dict | None = None) -> dict:
    path = Path(path)
    y = acoustic.load(path)
    feats = acoustic.extract(y)
    tags = _tagger().tag(y)
    vocal = acoustic.vocal_indicators(feats, tags["max_prob"], baseline)
    stt = azure_speech.transcribe(path)
    stt["ritmo_transcricao"] = azure_speech.speech_rate_from_words(stt.get("words", []))
    nlp = azure_text.analyze(stt["text"]) if stt["text"] else {"termos_criticos": [], "errors": ["sem texto"]}

    text_cat: dict[str, float] = {}
    for t in nlp.get("termos_criticos", []):
        text_cat[t["categoria"]] = max(text_cat.get(t["categoria"], 0), t["peso"])
    # entidades de saúde do Azure reforçam a categoria (quando disponíveis)
    health = [e for e in nlp.get("entidades_saude", []) if not e.get("negado")]
    sintomas = [e["texto"] for e in health if e["categoria"] in ("SymptomOrSign", "Diagnosis")]

    neg = (nlp.get("sentimento") or {}).get("scores", {}).get("negativo", 0.0)
    findings = []

    def add(cond, sev, desc, ev):
        findings.append({"condicao": cond, "severidade": sev, "descricao": desc, "evidencias": ev})

    d_v, d_t = vocal["disartria"]["score"], text_cat.get("neurologica_avc", 0)
    if d_v >= 0.45 and d_t >= 0.8:
        add("suspeita_avc", "CRITICO", "Disartria na voz + relato de sinais neurológicos focais (protocolo AVC — SAMU/código AVC)",
            vocal["disartria"]["evidencias"] + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] == "neurologica_avc"])
    elif d_v >= 0.45 or d_t >= 0.8:
        add("disartria", "ALTO", "Alteração de fala compatível com disartria", vocal["disartria"]["evidencias"]
            + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] == "neurologica_avc"])

    r_v, r_t, c_t = vocal["dificuldade_respiratoria"]["score"], text_cat.get("respiratoria", 0), text_cat.get("cardiaca", 0)
    if (r_v >= 0.5 or r_t >= 0.9) and c_t >= 0.9:
        add("insuficiencia_respiratoria_dor_toracica", "CRITICO",
            "Desconforto respiratório com dor torácica — avaliar SCA/TEP imediatamente",
            vocal["dificuldade_respiratoria"]["evidencias"]
            + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] in ("respiratoria", "cardiaca")])
    elif r_v >= 0.5 or r_t >= 0.9:
        add("dificuldade_respiratoria", "ALTO", "Sinais de desconforto respiratório na consulta",
            vocal["dificuldade_respiratoria"]["evidencias"]
            + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] == "respiratoria"])

    f_v, f_t = vocal["fadiga"]["score"], text_cat.get("fadiga", 0)
    if f_v >= 0.5 and f_t >= 0.5:
        add("fadiga", "ALTO", "Fadiga marcada (voz + relato) — investigar anemia, infecção, distúrbio metabólico",
            vocal["fadiga"]["evidencias"] + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] == "fadiga"])
    elif f_v >= 0.5 or f_t >= 0.5:
        add("fadiga", "ATENCAO", "Indícios de fadiga", vocal["fadiga"]["evidencias"]
            + [f"relato: {t['termo']}" for t in nlp["termos_criticos"] if t["categoria"] == "fadiga"])

    if text_cat.get("sofrimento_emocional", 0) and neg > 0.6:
        add("sofrimento_emocional", "INFO", "Sentimento negativo intenso — considerar suporte psicológico",
            [f"sentimento negativo={neg:.2f}"])

    order = {"INFO": 0, "ATENCAO": 1, "ALTO": 2, "CRITICO": 3}
    status = max((f["severidade"] for f in findings), key=order.get, default="NORMAL")
    return {
        "arquivo": path.name,
        "status": status,
        "achados": findings,
        "indicadores_vocais": vocal,
        "biomarcadores": feats,
        "audioset": {"max_prob": tags["max_prob"],
                     "eventos": [{"t": r["t"], "top3": r["top3"]} for r in tags["timeline"]
                                 if max(r[k] for k in ("Breathing", "Wheeze", "Gasp", "Pant", "Cough", "Throat clearing")) > 0.15],
                     "rotulos_pt": AUDIOSET_PT},
        "transcricao": stt,
        "nlp": nlp,
        "sintomas_azure_health": sintomas,
    }
