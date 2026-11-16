"""Detecção de alterações inesperadas na evolução de prescrições (MIMIC-IV).

Regras de farmacovigilância + Isolation Forest sobre features de cada ordem:
  - escalonamento/redução abrupta de dose do mesmo fármaco (mesma via e unidade);
  - dose acima do percentil 99,5 da população para o fármaco (outlier populacional);
  - duplicidade terapêutica concomitante (ex.: dois anticoagulantes, dois opioides);
  - escalonamento de via (oral -> intravenosa) do mesmo fármaco;
  - "rajada" de novos medicamentos de alta vigilância (ISMP) em poucas horas,
    sinal indireto de deterioração clínica.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import DATA_RAW

# Medicamentos potencialmente perigosos (lista ISMP, adaptada)
HIGH_ALERT = {
    "insulin": "insulina", "heparin": "anticoagulante", "enoxaparin": "anticoagulante",
    "warfarin": "anticoagulante", "apixaban": "anticoagulante", "rivaroxaban": "anticoagulante",
    "argatroban": "anticoagulante", "bivalirudin": "anticoagulante",
    "morphine": "opioide", "hydromorphone": "opioide", "fentanyl": "opioide", "oxycodone": "opioide",
    "methadone": "opioide", "meperidine": "opioide",
    "potassium chloride": "eletrolito concentrado", "magnesium sulfate": "eletrolito concentrado",
    "norepinephrine": "vasopressor", "epinephrine": "vasopressor", "vasopressin": "vasopressor",
    "phenylephrine": "vasopressor", "dopamine": "vasopressor",
    "propofol": "sedativo", "midazolam": "sedativo", "lorazepam": "sedativo", "dexmedetomidine": "sedativo",
    "amiodarone": "antiarritmico", "digoxin": "antiarritmico",
}
DUPLICATION_CLASSES = {"anticoagulante", "opioide", "sedativo"}
IV_ROUTES = {"IV", "IV DRIP", "IV BOLUS", "IVPCA", "IVS"}
ORAL_ROUTES = {"PO", "PO/NG", "ORAL", "NG", "SL"}
# Formulações que não representam terapia (flush, priming, lock de cateter)
NON_THERAPEUTIC = re.compile(r"flush|dwell|priming|crrt|hemodialysis|iabp|lock", re.I)


def _parse_dose(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).replace(",", "").strip()
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return np.nan
    return max(float(n) for n in nums)  # faixa "1-2" -> limite superior


def drug_class(name: str) -> str | None:
    n = str(name).lower()
    for key, cls in HIGH_ALERT.items():
        if key in n:
            return cls
    return None


def load_prescriptions(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "mimic_iv_demo" / "prescriptions.csv.gz"
    pr = pd.read_csv(path, parse_dates=["starttime", "stoptime"])
    pr = pr[pr["drug_type"] == "MAIN"].copy()
    pr["drug_norm"] = pr["drug"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    pr["dose"] = pr["dose_val_rx"].map(_parse_dose)
    pr["freq"] = pd.to_numeric(pr["doses_per_24_hrs"], errors="coerce")
    pr["daily_dose"] = pr["dose"] * pr["freq"].where(pr["freq"] > 0, 1)
    pr["route"] = pr["route"].fillna("NA").str.upper()
    pr["unit"] = pr["dose_unit_rx"].fillna("NA")
    pr["hclass"] = pr["drug_norm"].map(drug_class)
    pr["therapeutic"] = ~pr["drug_norm"].str.contains(NON_THERAPEUTIC)
    pr = pr.dropna(subset=["starttime"]).sort_values(["hadm_id", "starttime"]).reset_index(drop=True)
    return pr


def detect_rule_anomalies(pr: pd.DataFrame, burst_hours: float = 6, burst_n: int = 4) -> pd.DataFrame:
    events = []
    ther = pr[pr["therapeutic"] & pr["dose"].notna() & (pr["dose"] > 0)].copy()

    # 1) Escalonamento / redução abrupta (mesmo fármaco, via e unidade)
    key = ["hadm_id", "drug_norm", "route", "unit"]
    ther = ther.sort_values(key + ["starttime"])
    ther["prev_daily"] = ther.groupby(key)["daily_dose"].shift()
    ther["prev_time"] = ther.groupby(key)["starttime"].shift()
    ther["ratio"] = ther["daily_dose"] / ther["prev_daily"]
    chg = ther[(ther["ratio"] >= 2.5) | (ther["ratio"] <= 0.3)]
    for _, r in chg.iterrows():
        up = r["ratio"] >= 2.5
        hv = pd.notna(r["hclass"])
        big = r["ratio"] >= 4 or r["ratio"] <= 0.2
        if hv and r["hclass"] == "insulina":  # insulina varia por protocolo de escala móvel
            sev = "ALTO" if r["ratio"] >= 5 else "INFO"
        elif hv:
            sev = "ALTO" if big else "ATENCAO"
        else:
            sev = "ATENCAO" if big else "INFO"
        events.append({"hadm_id": r["hadm_id"], "subject_id": r["subject_id"], "time": r["starttime"],
                       "rule": "escalonamento_dose" if up else "reducao_abrupta_dose", "severity": sev,
                       "drug": r["drug"],
                       "detail": (f"{r['drug']} ({r['route']}): dose diária {r['prev_daily']:g} → "
                                  f"{r['daily_dose']:g} {r['unit']} (x{r['ratio']:.2g})"
                                  + (f" — medicamento de alta vigilância ({r['hclass']})" if hv else ""))})

    # 2) Outlier populacional de dose (p99.5 por fármaco/unidade/via, n>=20)
    grp = ther.groupby(["drug_norm", "unit", "route"])["dose"]
    p995 = grp.transform(lambda s: s.quantile(0.995) if len(s) >= 20 else np.inf)
    med = grp.transform("median")
    out = ther[(ther["dose"] > p995) & (ther["dose"] > 2 * med)]
    for _, r in out.iterrows():
        events.append({"hadm_id": r["hadm_id"], "subject_id": r["subject_id"], "time": r["starttime"],
                       "rule": "dose_outlier_populacional", "severity": "ALTO" if pd.notna(r["hclass"]) else "ATENCAO",
                       "drug": r["drug"],
                       "detail": f"{r['drug']}: dose {r['dose']:g} {r['unit']} acima do p99,5 populacional (mediana {med[_]:g})"})

    # 3) Duplicidade terapêutica concomitante
    hc = pr[pr["therapeutic"] & pr["hclass"].isin(DUPLICATION_CLASSES)].copy()
    hc["stoptime"] = hc["stoptime"].fillna(hc["starttime"] + pd.Timedelta(hours=24))
    hc["base"] = hc["drug_norm"].str.extract(r"^([a-z]+)", expand=False)
    for (hadm, cls), g in hc.groupby(["hadm_id", "hclass"]):
        g = g.sort_values("starttime")
        rows = g.to_dict("records")
        seen = set()
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if b["starttime"] >= a["stoptime"]:
                    continue
                if a["base"] == b["base"]:
                    continue
                pair = tuple(sorted((a["base"], b["base"])))
                if pair in seen:
                    continue
                seen.add(pair)
                if cls == "opioide" and ("PCA" in a["route"] + b["route"]):
                    continue
                both_iv = a["route"] in IV_ROUTES and b["route"] in IV_ROUTES
                bridge = cls == "anticoagulante" and "warfarin" in pair  # ponte heparina/enoxaparina -> varfarina
                if cls == "anticoagulante":
                    sev = "ATENCAO" if bridge else "ALTO"
                elif cls == "opioide":
                    sev = "ATENCAO" if both_iv else "INFO"
                else:
                    sev = "INFO"
                events.append({"hadm_id": hadm, "subject_id": a["subject_id"], "time": b["starttime"],
                               "rule": "duplicidade_terapeutica", "severity": sev,
                               "drug": f"{a['drug']} + {b['drug']}",
                               "detail": (f"Dois {cls}s concomitantes: {a['drug']} ({a['route']}) e {b['drug']} ({b['route']})"
                                          + (" — compatível com ponte terapêutica, confirmar INR" if bridge else ""))})

    # 4) Escalonamento de via oral -> IV
    t2 = pr[pr["therapeutic"]].copy()
    t2["base"] = t2["drug_norm"].str.extract(r"^([a-z]+)", expand=False)
    t2 = t2.sort_values(["hadm_id", "base", "starttime"])
    t2["prev_route"] = t2.groupby(["hadm_id", "base"])["route"].shift()
    rt = t2[t2["route"].isin(IV_ROUTES) & t2["prev_route"].isin(ORAL_ROUTES)]
    for _, r in rt.iterrows():
        events.append({"hadm_id": r["hadm_id"], "subject_id": r["subject_id"], "time": r["starttime"],
                       "rule": "escalonamento_via", "severity": "INFO", "drug": r["drug"],
                       "detail": f"{r['drug']}: via {r['prev_route']} → {r['route']} (possível piora/intolerância oral)"})

    # 5) Rajada de novos medicamentos de alta vigilância
    hv = pr[pr["therapeutic"] & pr["hclass"].notna()].sort_values(["hadm_id", "starttime"])
    hv = hv.drop_duplicates(["hadm_id", "drug_norm"])  # apenas inícios de novos fármacos
    for hadm, g in hv.groupby("hadm_id"):
        times = g["starttime"].to_numpy()
        last_flag = None
        for i in range(len(g)):
            w = g[(g["starttime"] >= g["starttime"].iat[i]) &
                  (g["starttime"] < g["starttime"].iat[i] + pd.Timedelta(hours=burst_hours))]
            if w["drug_norm"].nunique() >= burst_n and (last_flag is None or w["starttime"].iat[0] > last_flag):
                last_flag = w["starttime"].iat[-1]
                events.append({"hadm_id": hadm, "subject_id": g["subject_id"].iat[0], "time": w["starttime"].iat[-1],
                               "rule": "rajada_alta_vigilancia", "severity": "ATENCAO",
                               "drug": ", ".join(sorted(w["drug"].unique())[:6]),
                               "detail": (f"{w['drug_norm'].nunique()} novos medicamentos de alta vigilância em "
                                          f"{burst_hours:.0f}h ({', '.join(sorted(w['hclass'].unique()))})")})
        del times
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev = ev.sort_values(["hadm_id", "time"]).reset_index(drop=True)
        # no máximo 1 alerta por internação/regra/fármaco a cada 24h (evita fadiga de alarmes)
        ev["_day"] = ev.groupby(["hadm_id", "rule", "drug"])["time"].transform(
            lambda s: ((s - s.iloc[0]) // pd.Timedelta(hours=24)))
        ev = ev.drop_duplicates(["hadm_id", "rule", "drug", "_day"]).drop(columns="_day").reset_index(drop=True)
    return ev


def iforest_orders(pr: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Isolation Forest no nível da ordem de prescrição."""
    t = pr[pr["therapeutic"] & pr["dose"].notna() & (pr["dose"] > 0)].copy()
    key = ["drug_norm", "unit", "route"]
    t["log_dose"] = np.log1p(t["dose"])
    g = t.groupby(key)["log_dose"]
    t["dose_z_pop"] = (t["log_dose"] - g.transform("median")) / g.transform("std").replace(0, np.nan).fillna(1)
    k2 = ["hadm_id"] + key
    t = t.sort_values(k2 + ["starttime"])
    t["log_ratio_prev"] = np.log(t["daily_dose"] / t.groupby(k2)["daily_dose"].shift()).fillna(0)
    t["n_changes_24h"] = t.groupby("hadm_id")["starttime"].transform(
        lambda s: s.map(lambda x: ((s > x - pd.Timedelta(hours=24)) & (s <= x)).sum()))
    t["high_alert"] = t["hclass"].notna().astype(int)
    t["iv"] = t["route"].isin(IV_ROUTES).astype(int)
    feats = ["dose_z_pop", "log_ratio_prev", "n_changes_24h", "high_alert", "iv"]
    X = t[feats].replace([np.inf, -np.inf], 0).fillna(0)
    m = IsolationForest(n_estimators=300, contamination=contamination, random_state=42).fit(X)
    t["if_score"] = -m.score_samples(X)
    t["if_anomaly"] = m.predict(X) == -1
    return t
