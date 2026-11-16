"""Detecção de anomalias em séries temporais de sinais vitais (MIMIC-IV / PhysioNet).

Sinais: frequência cardíaca (FC), pressão arterial sistólica/diastólica/média,
SpO2 (oxigenação), frequência respiratória (FR) e temperatura.

Três detectores complementares:
  1. Regras clínicas — escore NEWS2 (Royal College of Physicians, 2017), parcial
     (sem nível de consciência e O2 suplementar, ausentes no extrato);
  2. Estatístico por paciente — z-score robusto (mediana/MAD) contra a linha de
     base móvel do próprio paciente: detecta mudanças abruptas mesmo dentro da
     faixa "normal" populacional;
  3. Aprendizado de máquina multivariado — Isolation Forest treinado em dados
     populacionais (valores + variações), detecta combinações atípicas.

`VitalsStreamMonitor` aplica os três em modo streaming (uma observação por vez),
com supressão de alertas repetidos (cooldown), para monitoramento em tempo real.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import DATA_RAW

ITEMS = {220045: "hr", 220179: "sbp_ni", 220180: "dbp_ni", 220181: "map_ni", 220050: "sbp_a",
         220051: "dbp_a", 220052: "map_a", 220277: "spo2", 220210: "rr", 223761: "temp_f", 223762: "temp_c"}
SIGNALS = ["hr", "sbp", "dbp", "map", "spo2", "rr", "temp"]
LABELS = {"hr": "Frequência cardíaca", "sbp": "PA sistólica", "dbp": "PA diastólica", "map": "PA média",
          "spo2": "SpO2", "rr": "Frequência respiratória", "temp": "Temperatura"}
UNITS = {"hr": "bpm", "sbp": "mmHg", "dbp": "mmHg", "map": "mmHg", "spo2": "%", "rr": "irpm", "temp": "°C"}
# Faixas fisiologicamente plausíveis — fora delas é artefato de sensor/registro
PLAUSIBLE = {"hr": (20, 250), "sbp": (40, 260), "dbp": (15, 180), "map": (25, 200),
             "spo2": (50, 100), "rr": (3, 60), "temp": (30, 43)}


# ---------------------------------------------------------------- carga
def load_vitals(path: Path | None = None) -> pd.DataFrame:
    """Lê chartevents e devolve tabela larga (stay_id, charttime, hr, sbp, ..., temp)."""
    path = path or DATA_RAW / "mimic_iv_demo" / "chartevents.csv.gz"
    ce = pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"])
    ce = ce[ce["itemid"].isin(ITEMS)].dropna(subset=["valuenum"])
    ce["var"] = ce["itemid"].map(ITEMS)
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    wide = ce.pivot_table(index=["subject_id", "hadm_id", "stay_id", "charttime"], columns="var",
                          values="valuenum", aggfunc="mean").reset_index()
    for c in ITEMS.values():
        if c not in wide:
            wide[c] = np.nan
    # Pressão: prioriza a invasiva (arterial), senão a não invasiva
    wide["sbp"] = wide["sbp_a"].fillna(wide["sbp_ni"])
    wide["dbp"] = wide["dbp_a"].fillna(wide["dbp_ni"])
    wide["map"] = wide["map_a"].fillna(wide["map_ni"])
    wide["temp"] = wide["temp_c"].where(wide["temp_c"].between(25, 45),
                                        (wide["temp_f"] - 32) * 5 / 9)
    out = wide[["subject_id", "hadm_id", "stay_id", "charttime"] + SIGNALS].sort_values(["stay_id", "charttime"])
    return out.reset_index(drop=True)


def remove_artifacts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Substitui valores implausíveis por NaN e devolve (limpo, log de artefatos)."""
    df = df.copy()
    logs = []
    for s, (lo, hi) in PLAUSIBLE.items():
        bad = df[s].notna() & ~df[s].between(lo, hi)
        if bad.any():
            logs.append(df.loc[bad, ["stay_id", "charttime"]].assign(signal=s, value=df.loc[bad, s]))
            df.loc[bad, s] = np.nan
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["stay_id", "charttime", "signal", "value"])
    return df, log


# ---------------------------------------------------------------- NEWS2
def _news_rr(v):
    return np.select([v <= 8, v <= 11, v <= 20, v <= 24], [3, 1, 0, 2], 3)


def _news_spo2(v):
    return np.select([v <= 91, v <= 93, v <= 95], [3, 2, 1], 0)


def _news_sbp(v):
    return np.select([v <= 90, v <= 100, v <= 110, v <= 219], [3, 2, 1, 0], 3)


def _news_hr(v):
    return np.select([v <= 40, v <= 50, v <= 90, v <= 110, v <= 130], [3, 1, 0, 1, 2], 3)


def _news_temp(v):
    return np.select([v <= 35.0, v <= 36.0, v <= 38.0, v <= 39.0], [3, 1, 0, 1], 2)


NEWS_FUNCS = {"rr": _news_rr, "spo2": _news_spo2, "sbp": _news_sbp, "hr": _news_hr, "temp": _news_temp}


def news2(df: pd.DataFrame) -> pd.DataFrame:
    """Escore NEWS2 parcial. Usa o último valor conhecido (até 4h) de cada parâmetro."""
    out = pd.DataFrame(index=df.index)
    filled = df.groupby("stay_id")[list(NEWS_FUNCS)].transform(lambda s: s.ffill(limit=4))
    for s, f in NEWS_FUNCS.items():
        v = filled[s].to_numpy()
        out[f"news_{s}"] = np.where(np.isnan(v), 0, f(np.nan_to_num(v, nan=0)))
    out["news2"] = out[[c for c in out if c.startswith("news_")]].sum(axis=1)
    out["news2_red"] = (out[[c for c in out if c.startswith("news_") and c != "news2"]] == 3).any(axis=1)
    return out


def news2_severity(score: int, red: bool) -> str | None:
    if score >= 7:
        return "CRITICO"
    if score >= 5 or red:
        return "ALTO"
    if score >= 3:
        return "ATENCAO"
    return None


# ------------------------------------------------------- z-score robusto
def robust_z(df: pd.DataFrame, window: int = 12, min_periods: int = 6) -> pd.DataFrame:
    """z = (x - mediana_movel) / (1.4826 * MAD_movel), usando apenas observações anteriores."""
    out = pd.DataFrame(index=df.index)
    floor = {"hr": 3, "sbp": 5, "dbp": 4, "map": 4, "spo2": 1, "rr": 2, "temp": 0.2}  # MAD mínimo
    for s in SIGNALS:
        g = df.groupby("stay_id")[s]
        med = g.transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).median())
        mad = g.transform(lambda x: (x.shift(1) - x.shift(1).rolling(window, min_periods=min_periods).median())
                          .abs().rolling(window, min_periods=min_periods).median())
        scale = np.maximum(1.4826 * mad, floor[s])
        out[f"z_{s}"] = (df[s] - med) / scale
    return out


# ------------------------------------------------------- Isolation Forest
def _iforest_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.groupby("stay_id")[SIGNALS].transform(lambda s: s.ffill(limit=3))
    feats = f.copy()
    g = f.groupby(df["stay_id"])
    for s in SIGNALS:
        # desvio em relação à linha de base recente do próprio paciente
        base = g[s].transform(lambda x: x.shift(1).rolling(12, min_periods=4).median())
        feats[f"dev_{s}"] = f[s] - base
    for s in ("hr", "sbp", "spo2", "rr"):
        feats[f"d_{s}"] = g[s].diff()
    feats["shock_index"] = f["hr"] / f["sbp"]
    return feats


@dataclass
class VitalsIForest:
    contamination: float = 0.02
    model: IsolationForest | None = None
    medians: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "VitalsIForest":
        X = _iforest_features(df)
        self.medians = X.median()
        self.model = IsolationForest(n_estimators=300, contamination=self.contamination, random_state=42)
        self.model.fit(X.fillna(self.medians))
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _iforest_features(df).fillna(self.medians)
        s = -self.model.score_samples(X)  # maior = mais anômalo
        thr = -self.model.offset_
        return pd.DataFrame({"if_score": s, "if_anomaly": s > thr}, index=df.index)


# ------------------------------------------------------- pipeline em lote
def analyze(df: pd.DataFrame, iforest: VitalsIForest, z_thr: float = 4.0) -> pd.DataFrame:
    res = pd.concat([df, news2(df), robust_z(df), iforest.score(df)], axis=1)
    zcols = [f"z_{s}" for s in SIGNALS]
    res["z_max"] = res[zcols].abs().max(axis=1)
    res["z_signal"] = res[zcols].abs().fillna(-1).idxmax(axis=1).str[2:].where(res["z_max"].notna())
    res["z_anomaly"] = res["z_max"] > z_thr
    res["news_anomaly"] = res["news2"] >= 5
    res["ensemble_votes"] = res[["z_anomaly", "news_anomaly", "if_anomaly"]].sum(axis=1)
    return res


# ------------------------------------------------------- injeção para avaliação
def inject_anomalies(df: pd.DataFrame, n_per_stay: int = 3, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Injeta eventos clínicos sintéticos (rotulados) para medir precisão/recall.

    Tipos: taquicardia súbita, dessaturação, hipotensão, taquipneia, febre.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    label = pd.Series(False, index=df.index)
    kinds = {
        "taquicardia": [("hr", +45)],
        "dessaturacao": [("spo2", -10), ("rr", +8)],
        "hipotensao": [("sbp", -40), ("dbp", -20), ("map", -25), ("hr", +20)],
        "taquipneia": [("rr", +14)],
        "febre": [("temp", +2.3), ("hr", +15)],
    }
    df["injected"] = ""
    for stay, idx in df.groupby("stay_id").groups.items():
        idx = list(idx)
        if len(idx) < 30:
            continue
        for _ in range(n_per_stay):
            pos = int(rng.integers(15, len(idx) - 3))
            kind = rng.choice(list(kinds))
            dur = int(rng.integers(1, 4))
            for p in idx[pos:pos + dur]:
                for s, delta in kinds[kind]:
                    if pd.notna(df.at[p, s]):
                        df.at[p, s] = float(np.clip(df.at[p, s] + delta, *PLAUSIBLE[s]))
                label[p] = True
                df.at[p, "injected"] = kind
    return df, label


def evaluate(res: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Métricas em dados com anomalias injetadas.

    - recall_evento: fração dos eventos injetados detectados (ao menos 1 ponto sinalizado);
    - taxa_alarme_base: fração das medições SEM injeção que são sinalizadas (carga de alarmes;
      inclui anomalias reais do paciente de UTI, por isso não é "falso positivo" puro);
    - precisao_pontual: tp / (tp + fp) considerando apenas os rótulos injetados.
    """
    ev_id = (label != label.shift()).cumsum().where(label)
    rows = []
    for name, pred in {"NEWS2 (>=5)": res["news_anomaly"], "z-score robusto (|z|>4)": res["z_anomaly"],
                       "Isolation Forest": res["if_anomaly"],
                       "Ensemble (>=2 votos)": res["ensemble_votes"] >= 2}.items():
        tp = int((pred & label).sum())
        fp = int((pred & ~label).sum())
        ev = pred[label].groupby(ev_id[label]).any()
        rows.append({"detector": name,
                     "recall_evento": round(float(ev.mean()), 3),
                     "taxa_alarme_base": round(float(pred[~label].mean()), 3),
                     "precisao_pontual": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                     "eventos_detectados": f"{int(ev.sum())}/{len(ev)}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------- streaming
@dataclass
class VitalsStreamMonitor:
    """Monitor em tempo real: recebe uma observação por vez e devolve alertas."""
    iforest: VitalsIForest
    window: int = 12
    z_thr: float = 4.0
    cooldown: int = 3  # nº de observações sem repetir alerta do mesmo tipo
    news_reminder: int = 6  # lembrete do NEWS2 persistente (nº de observações)
    history: deque = field(default_factory=lambda: deque(maxlen=48))
    _last: dict = field(default_factory=dict)
    _step: int = 0
    _news_level: int = 0

    def update(self, obs: dict) -> list[dict]:
        self._step += 1
        obs = {k: (np.nan if obs.get(k) is None else obs.get(k)) for k in ["stay_id", "charttime"] + SIGNALS}
        for s, (lo, hi) in PLAUSIBLE.items():
            if pd.notna(obs[s]) and not lo <= obs[s] <= hi:
                obs[s] = np.nan
        self.history.append(obs)
        hist = pd.DataFrame(list(self.history))
        last = hist.iloc[[-1]].copy()
        alerts = []
        # NEWS2
        nw = news2(hist).iloc[-1]
        sev = news2_severity(int(nw["news2"]), bool(nw["news2_red"]))
        level = {"ATENCAO": 1, "ALTO": 2, "CRITICO": 3}.get(sev, 0)
        # NEWS2 alerta quando o nível de risco PIORA, ou como lembrete se persistir
        if level and (level > self._news_level or self._step - self._last.get("news2", -10 ** 9) > self.news_reminder):
            comp = {s: int(nw[f"news_{s}"]) for s in NEWS_FUNCS if nw[f"news_{s}"] > 0}
            piora = " (piora)" if level > self._news_level and self._news_level > 0 else ""
            alerts.append(("news2", sev, f"NEWS2 = {int(nw['news2'])}{piora}", {"componentes": comp}))
            self._last["news2"] = -10 ** 9  # não aplica o cooldown genérico ao NEWS2
        self._news_level = level
        # z-score robusto
        zz = robust_z(hist, self.window).iloc[-1]
        for s in SIGNALS:
            z = zz[f"z_{s}"]
            if pd.notna(z) and abs(z) > self.z_thr:
                sev = "ALTO" if abs(z) > 2 * self.z_thr else "ATENCAO"
                alerts.append((f"z_{s}", sev, f"{LABELS[s]} fora do padrão do paciente (z={z:+.1f})",
                               {"valor": round(float(last[s].iat[0]), 1), "z": round(float(z), 2)}))
        # Isolation Forest
        sc = self.iforest.score(hist).iloc[-1]
        if bool(sc["if_anomaly"]):
            alerts.append(("iforest", "ATENCAO", "Combinação multivariada atípica (Isolation Forest)",
                           {"score": round(float(sc["if_score"]), 3)}))
        out = []
        for key, sev, title, ev in alerts:
            if self._step - self._last.get(key, -10 ** 9) <= self.cooldown:
                continue
            self._last[key] = self._step
            ev.update({s: (None if pd.isna(last[s].iat[0]) else round(float(last[s].iat[0]), 1)) for s in SIGNALS})
            out.append({"key": key, "severity": sev, "title": title, "evidence": ev})
        return out
