"""Padrões de movimentação do paciente durante a internação.

Duas fontes complementares:
  * Registros de enfermagem do MIMIC-IV (PhysioNet): mudança de decúbito
    ("Turn", itemid 224082), posição no leito (224093) e nível de atividade
    (224084: repouso no leito, cadeira, deambulação...);
  * Eventos derivados do vídeo (queda, imobilidade) — ver src/video.

Anomalias detectadas:
  - intervalo sem reposicionamento > 3h (protocolo de prevenção de lesão por
    pressão prevê 2h; tolerância de 1h para registro; > 4h = ALTO) em paciente restrito ao leito;
  - intervalo atípico para o próprio paciente (z-score robusto dos intervalos);
  - declínio funcional: melhor nível de atividade do dia >= 2 níveis abaixo
    do melhor nível do dia anterior;
  - perda de autonomia: paciente que "virava sozinho" passa a depender da equipe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW

ITEM_TURN, ITEM_POSITION, ITEM_ACTIVITY = 224082, 224093, 224084
ACTIVITY_LEVEL = {"Bedrest": 1, "Bed as Chair": 2, "Dangle": 2, "Commode": 3, "Chair": 3, "Ambulate": 4}
ACTIVITY_PT = {"Bedrest": "repouso no leito", "Bed as Chair": "leito em posição de cadeira",
               "Dangle": "sentado à beira do leito", "Commode": "cadeira higiênica",
               "Chair": "cadeira", "Ambulate": "deambulando"}


def load_movement(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW / "mimic_iv_demo" / "chartevents.csv.gz"
    ce = pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"])
    ce = ce[ce["itemid"].isin([ITEM_TURN, ITEM_POSITION, ITEM_ACTIVITY])].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["value"] = ce["value"].astype(str).str.strip()
    ce["kind"] = ce["itemid"].map({ITEM_TURN: "turn", ITEM_POSITION: "position", ITEM_ACTIVITY: "activity"})
    return ce.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def detect_movement_anomalies(mv: pd.DataFrame, gap_warn_h: float = 3.0, gap_high_h: float = 4.0,
                              max_gap_h: float = 12.0) -> pd.DataFrame:
    events = []
    for stay, g in mv.groupby("stay_id"):
        subj, hadm = g["subject_id"].iat[0], g["hadm_id"].iat[0]
        act = g[g["kind"] == "activity"].copy()
        act["level"] = act["value"].map(ACTIVITY_LEVEL)
        # --- intervalos sem reposicionamento (apenas enquanto restrito ao leito)
        repos = g[g["kind"].isin(["turn", "position"])].drop_duplicates("charttime")["charttime"]
        if len(repos) >= 4:
            gaps = repos.diff().dt.total_seconds() / 3600
            med = gaps.median()
            mad = max((gaps - med).abs().median() * 1.4826, 0.25)
            for t_prev, t, gap in zip(repos.shift(), repos, gaps):
                if pd.isna(gap) or gap > max_gap_h:  # lacuna de registro, não de cuidado
                    continue
                last_act = act[act["charttime"] <= t]["level"].dropna()
                bedbound = last_act.empty or last_act.iat[-1] <= 2
                if gap > gap_warn_h and bedbound:
                    z = (gap - med) / mad
                    sev = "ALTO" if gap > gap_high_h else "ATENCAO"
                    events.append({"stay_id": stay, "subject_id": subj, "hadm_id": hadm, "time": t,
                                   "rule": "sem_reposicionamento", "severity": sev,
                                   "detail": (f"Paciente restrito ao leito sem mudança de decúbito por {gap:.1f}h "
                                              f"(protocolo: a cada 2h; padrão do paciente {med:.1f}h, z={z:+.1f}) "
                                              f"— risco de lesão por pressão"),
                                   "value": round(float(gap), 2)})
        # --- declínio funcional (melhor nível diário vs. dia anterior)
        act = act.dropna(subset=["level"])
        if not act.empty:
            act["day"] = act["charttime"].dt.floor("D")
            daily = act.groupby("day").agg(level=("level", "max"), t=("charttime", "max"),
                                           value=("value", lambda v: v.iloc[0]))
            best_val = act.loc[act.groupby("day")["level"].idxmax()].set_index("day")["value"]
            for i in range(1, len(daily)):
                d0, d1 = daily.index[i - 1], daily.index[i]
                if (d1 - d0).days == 1 and daily["level"].iat[i - 1] - daily["level"].iat[i] >= 2:
                    events.append({"stay_id": stay, "subject_id": subj, "hadm_id": hadm, "time": daily["t"].iat[i],
                                   "rule": "declinio_funcional", "severity": "ATENCAO",
                                   "detail": (f"Declínio funcional: melhor nível de atividade caiu de "
                                              f"'{ACTIVITY_PT.get(best_val[d0], best_val[d0])}' para "
                                              f"'{ACTIVITY_PT.get(best_val[d1], best_val[d1])}' de um dia para o outro"),
                                   "value": float(daily["level"].iat[i - 1] - daily["level"].iat[i])})
        # --- perda de autonomia na mudança de decúbito (sustentada)
        turns = g[g["kind"] == "turn"].reset_index(drop=True)
        selfturn = turns["value"].eq("Turns by Self").to_numpy()
        last_flag = None
        for i in range(3, len(turns) - 2):
            if selfturn[i - 3:i].all() and not selfturn[i:i + 3].any():
                t = turns["charttime"].iat[i]
                if last_flag is None or t - last_flag > pd.Timedelta(hours=24):
                    last_flag = t
                    events.append({"stay_id": stay, "subject_id": subj, "hadm_id": hadm, "time": t,
                                   "rule": "perda_autonomia", "severity": "ATENCAO",
                                   "detail": ("Paciente que se reposicionava sozinho passou a depender da equipe "
                                              f"('{turns['value'].iat[i]}') de forma sustentada"),
                                   "value": np.nan})
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev = ev.sort_values(["stay_id", "time"]).reset_index(drop=True)
        # agrupa alertas de reposicionamento consecutivos (1 por 8h)
        mask = ev["rule"] == "sem_reposicionamento"
        r = ev[mask].copy()
        r["_b"] = r.groupby("stay_id")["time"].transform(lambda s: (s - s.iloc[0]) // pd.Timedelta(hours=8))
        r = r.sort_values("value", ascending=False).drop_duplicates(["stay_id", "_b"]).drop(columns="_b")
        ev = pd.concat([ev[~mask], r]).sort_values(["stay_id", "time"]).reset_index(drop=True)
    return ev


def hourly_profile(mv: pd.DataFrame, stay_id: int) -> pd.DataFrame:
    """Série horária de eventos de reposicionamento/atividade (para gráficos)."""
    g = mv[mv["stay_id"] == stay_id].copy()
    g["hour"] = g["charttime"].dt.floor("h")
    return g.pivot_table(index="hour", columns="kind", values="value", aggfunc="count").fillna(0)
