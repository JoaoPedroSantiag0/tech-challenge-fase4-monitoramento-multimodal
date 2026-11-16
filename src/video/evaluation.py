"""Calibração e avaliação do detector de quedas no UR Fall Detection Dataset.

Divisão fixa: treino = quedas 01–15 + atividades diárias (ADL) 01–20;
               teste  = quedas 16–30 + ADL 21–40.
Os limiares são escolhidos por busca em grade APENAS no treino (maximizando a
acurácia balanceada) e depois aplicados uma única vez no teste.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.video.analyzer import VideoAnalyzer, VideoConfig


class _RulesOnly(VideoAnalyzer):
    """Usa só as regras (sem carregar modelos) sobre séries já extraídas."""

    def __init__(self, cfg: VideoConfig) -> None:  # noqa: D401
        self.cfg = cfg


def load_series(folder: Path) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}


def split_of(name: str) -> str:
    kind, num = name.split("-")[0], int(name.split("-")[1])
    return "train" if (num <= 15 if kind == "fall" else num <= 20) else "test"


def run(series: dict[str, pd.DataFrame], cfg: VideoConfig) -> pd.DataFrame:
    va = _RulesOnly(cfg)
    rows = []
    for name, df in series.items():
        events, d = va.detect_events(df)
        falls = [e for e in events if e.kind == "queda"]
        rows.append({"video": name, "queda_real": name.startswith("fall"), "split": split_of(name),
                     "queda_detectada": bool(falls),
                     "t_deteccao_s": round(falls[0].t_start, 2) if falls else None,
                     "eventos": ",".join(e.kind for e in events),
                     "max_drop_rate": round(float(d["drop_rate"].max()), 3),
                     "frac_no_chao": round(float(d["on_ground"].mean()), 3)})
    return pd.DataFrame(rows)


def metrics(r: pd.DataFrame) -> dict:
    tp = int((r["queda_detectada"] & r["queda_real"]).sum())
    fp = int((r["queda_detectada"] & ~r["queda_real"]).sum())
    fn = int((~r["queda_detectada"] & r["queda_real"]).sum())
    tn = int((~r["queda_detectada"] & ~r["queda_real"]).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "sensibilidade": round(sens, 3),
            "especificidade": round(spec, 3), "precisao": round(prec, 3),
            "f1": round(2 * prec * sens / (prec + sens), 3) if prec + sens else 0.0,
            "acuracia_balanceada": round((sens + spec) / 2, 3)}


GRID = {
    "fall_drop_rate": [0.35, 0.4, 0.45, 0.5, 0.55],
    "fall_window_s": [0.5, 0.8, 1.0],
    "min_ground_s": [0.5, 0.8, 1.2],
    "lying_torso_deg": [50.0, 60.0],
    "floor_zone_y": [0.58, 0.62, 0.66],
}


def calibrate(series: dict[str, pd.DataFrame], base: VideoConfig) -> tuple[VideoConfig, pd.DataFrame]:
    train = {k: v for k, v in series.items() if split_of(k) == "train"}
    rows = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        cfg = replace(base, **dict(zip(keys, combo)))
        m = metrics(run(train, cfg))
        rows.append({**dict(zip(keys, combo)), **m})
    g = pd.DataFrame(rows)
    # desempate: maior acurácia balanceada, depois maior especificidade, depois limiar mais conservador
    g = g.sort_values(["acuracia_balanceada", "especificidade", "fall_drop_rate", "min_ground_s"],
                      ascending=[False, False, False, False])
    best = g.iloc[0]
    return replace(base, **{k: (float(best[k])) for k in keys}), g
