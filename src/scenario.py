"""Monta o cenário multimodal de demonstração (config/scenario.json).

Para cada leito: janela de 24h de uma internação real do MIMIC-IV (sinais vitais,
prescrições, movimentação), deslocada para a data da demonstração, mais uma
consulta em áudio e/ou um vídeo do quarto em horários definidos.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.anomaly import movement as M
from src.anomaly import prescriptions as P
from src.anomaly import vitals as V
from src.config import ROOT

SCENARIO_FILE = ROOT / "config" / "scenario.json"


@dataclass
class PatientWindow:
    spec: dict
    shift: pd.Timedelta
    vitals: pd.DataFrame
    presc_events: pd.DataFrame
    move_events: pd.DataFrame
    prescriptions: pd.DataFrame
    audio_time: pd.Timestamp | None = None
    video_time: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)


def load_scenario() -> dict:
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


def _resolve_offset(off, vit: pd.DataFrame, start: pd.Timestamp) -> float:
    """Aceita horas (ex.: 9.0) ou âncora no sinal (ex.: "spo2_min-0.5" = 30 min antes da menor SpO2)."""
    if isinstance(off, (int, float)):
        return float(off)
    m = re.fullmatch(r"(\w+)_(min|max)([+-][\d.]+)?", off)
    if not m:
        raise ValueError(f"offset inválido: {off}")
    sig, how, delta = m.group(1), m.group(2), float(m.group(3) or 0)
    s = vit.set_index("charttime")[sig].dropna()
    t = s.idxmin() if how == "min" else s.idxmax()
    return (t - start).total_seconds() / 3600 + delta


def build(scn: dict | None = None) -> tuple[dict, list[PatientWindow], V.VitalsIForest]:
    scn = scn or load_scenario()
    demo_start = pd.Timestamp(scn["demo_start"])
    dur = pd.Timedelta(hours=scn["duration_h"])
    vit = V.load_vitals()
    clean, _ = V.remove_artifacts(vit)
    scen_stays = {p["stay_id"] for p in scn["patients"]}
    iforest = V.VitalsIForest().fit(clean[~clean["stay_id"].isin(scen_stays)])  # sem vazamento
    pr = P.load_prescriptions()
    pev = P.detect_rule_anomalies(pr)
    mv = M.load_movement()
    mev = M.detect_movement_anomalies(mv[mv["stay_id"].isin(scen_stays)])

    out = []
    for p in scn["patients"]:
        ws = pd.Timestamp(p["window_start"])
        shift = demo_start - ws
        v = vit[(vit["stay_id"] == p["stay_id"]) & (vit["charttime"] >= ws) & (vit["charttime"] < ws + dur)].copy()
        hadm = int(v["hadm_id"].iat[0])
        pe = pev[(pev["hadm_id"] == hadm) & (pev["time"] >= ws) & (pev["time"] < ws + dur)].copy()
        me = mev[(mev["stay_id"] == p["stay_id"]) & (mev["time"] >= ws) & (mev["time"] < ws + dur)].copy()
        prw = pr[(pr["hadm_id"] == hadm) & (pr["starttime"] >= ws - pd.Timedelta(hours=48)) & (pr["starttime"] < ws + dur)].copy()
        for df, col in ((v, "charttime"), (pe, "time"), (me, "time"), (prw, "starttime")):
            df[col] = df[col] + shift
        pw = PatientWindow(spec=p, shift=shift, vitals=v.reset_index(drop=True), presc_events=pe,
                           move_events=me, prescriptions=prw)
        if "audio" in p:
            pw.audio_time = demo_start + pd.Timedelta(hours=_resolve_offset(p["audio"]["offset_h"], pw.vitals, demo_start))
        if "video" in p:
            pw.video_time = demo_start + pd.Timedelta(hours=_resolve_offset(p["video"]["offset_h"], pw.vitals, demo_start))
        out.append(pw)
    return scn, out, iforest


# ------------------------------------------------------------- conversão p/ eventos
def vitals_events(pw: PatientWindow, iforest: V.VitalsIForest) -> tuple[list[dict], pd.DataFrame]:
    mon = V.VitalsStreamMonitor(iforest=iforest)
    events, trace = [], []
    for _, row in pw.vitals.iterrows():
        obs = row.to_dict()
        alerts = mon.update(obs)
        nw = V.news2(pd.DataFrame(list(mon.history))).iloc[-1]
        trace.append({"charttime": row["charttime"], "news2": int(nw["news2"]), **{s: row[s] for s in V.SIGNALS}})
        for a in alerts:
            events.append({"time": row["charttime"], "modality": "vitais", "severity": a["severity"],
                           "title": a["title"], "description": _vitals_desc(a), "source": _vitals_src(a["key"]),
                           "evidence": a["evidence"]})
    return events, pd.DataFrame(trace)


def _vitals_src(key: str) -> str:
    if key == "news2":
        return "Regras clínicas NEWS2"
    if key == "iforest":
        return "Isolation Forest multivariado"
    return "z-score robusto (linha de base do paciente)"


def _vitals_desc(a: dict) -> str:
    ev = a["evidence"]
    vit = ", ".join(f"{V.LABELS[s]} {ev[s]}{V.UNITS[s]}" for s in V.SIGNALS if ev.get(s) is not None)
    return f"{a['title']}. Últimos valores: {vit}."


def table_events(df: pd.DataFrame, modality: str, source: str) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append({"time": r["time"], "modality": modality, "severity": r["severity"],
                    "title": r["rule"].replace("_", " ").capitalize(), "description": r["detail"],
                    "source": source, "evidence": {"regra": r["rule"]}})
    return out
