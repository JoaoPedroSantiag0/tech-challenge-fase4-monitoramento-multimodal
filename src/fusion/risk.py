"""Fusão multimodal tardia (late fusion) e escalonamento de alertas.

Cada módulo produz eventos (modalidade, severidade, horário). A fusão:
  1. converte severidade em probabilidade de risco
     (INFO 0,05 · ATENÇÃO 0,25 · ALTO 0,55 · CRÍTICO 0,85);
  2. aplica decaimento exponencial com meia-vida específica da modalidade
     (um evento de vídeo/vitais "envelhece" mais rápido que uma alteração de prescrição);
  3. dentro de cada modalidade usa o MAIOR risco vigente (alertas repetidos da
     mesma fonte não são evidências independentes); entre modalidades combina
     por noisy-OR:  R(t) = 1 − Π_m (1 − max_i wᵢ·e^{−Δtᵢ·ln2/h_m});
  4. regra de corroboração em janela de 6h: >= 2 modalidades com ALTO+ → alerta
     de fusão CRÍTICO; 1 modalidade ALTO+ corroborada por outra ATENÇÃO+ →
     alerta de fusão ALTO ("deterioração multimodal").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEV_W = {"INFO": 0.05, "ATENCAO": 0.25, "ALTO": 0.55, "CRITICO": 0.85}
SEV_ORDER = {"INFO": 0, "ATENCAO": 1, "ALTO": 2, "CRITICO": 3}
HALF_LIFE_H = {"vitais": 2.0, "video": 3.0, "audio": 8.0, "prescricao": 12.0, "movimentacao": 6.0, "fusao": 2.0}


@dataclass
class FusionEngine:
    window_h: float = 6.0

    def risk_at(self, events: pd.DataFrame, t: pd.Timestamp) -> dict:
        past = events[events["time"] <= t]
        if past.empty:
            return {"risk": 0.0, "by_modality": {}}
        dt_h = (t - past["time"]).dt.total_seconds() / 3600
        hl = past["modality"].map(HALF_LIFE_H).fillna(4.0)
        w = past["severity"].map(SEV_W) * np.exp(-dt_h * np.log(2) / hl)
        by_mod = {}
        for mod, ww in w.groupby(past["modality"]):
            by_mod[mod] = float(ww.max())
        risk = float(1 - np.prod([1 - v for v in by_mod.values()]))
        return {"risk": round(risk, 4), "by_modality": {k: round(v, 4) for k, v in by_mod.items()}}

    def timeline(self, events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, freq: str = "15min") -> pd.DataFrame:
        rows = []
        for t in pd.date_range(start, end, freq=freq):
            r = self.risk_at(events, t)
            rows.append({"time": t, "risk": r["risk"], **{f"risk_{k}": v for k, v in r["by_modality"].items()}})
        return pd.DataFrame(rows).fillna(0)

    def corroboration_alerts(self, events: pd.DataFrame) -> list[dict]:
        """Avalia, a cada novo evento, se há corroboração multimodal na janela."""
        out, last_level = [], -1
        ev = events[events["modality"] != "fusao"].sort_values("time")
        last_time = None
        for _, e in ev.iterrows():
            win = ev[(ev["time"] <= e["time"]) & (ev["time"] > e["time"] - pd.Timedelta(hours=self.window_h))]
            high = win[win["severity"].map(SEV_ORDER) >= 2]["modality"].unique()
            att = win[win["severity"].map(SEV_ORDER) >= 1]["modality"].unique()
            if len(high) >= 2:
                level, mods, sev = 3, high, "CRITICO"
            elif len(high) >= 1 and len(att) >= 2:
                level, mods, sev = 2, att, "ALTO"
            else:
                continue
            # evita repetir o mesmo nível de fusão dentro da janela
            if last_time is not None and e["time"] - last_time < pd.Timedelta(hours=self.window_h) and level <= last_level:
                continue
            last_level, last_time = level, e["time"]
            contrib = win[win["modality"].isin(mods)].sort_values("severity", key=lambda s: s.map(SEV_ORDER), ascending=False)
            out.append({
                "time": e["time"], "severity": sev, "modality": "fusao",
                "title": f"Deterioração multimodal ({' + '.join(sorted(mods))})",
                "description": "Corroboração entre modalidades nas últimas "
                               f"{self.window_h:.0f}h: " + "; ".join(contrib["title"].head(4)),
                "source": "Fusão tardia (noisy-OR + regra de corroboração)",
                "evidence": {"modalidades": sorted(mods.tolist()), "n_eventos": int(len(win))},
            })
        return out
