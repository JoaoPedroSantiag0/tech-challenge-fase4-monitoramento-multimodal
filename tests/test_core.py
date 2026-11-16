import json

import numpy as np
import pandas as pd
import pytest

from src.anomaly import prescriptions as P
from src.anomaly import vitals as V
from src.audio.azure_text import critical_terms
from src.common.alerts import Alert, AlertBus, load_alerts
from src.fusion.risk import FusionEngine
from src.video.pose_features import joint_angle, torso_angle


def test_news2_reference_values():
    df = pd.DataFrame({"stay_id": [1, 1], "rr": [18, 26], "spo2": [97, 90], "sbp": [125, 88],
                       "hr": [75, 135], "temp": [37.0, 39.5]})
    n = V.news2(df)
    assert n["news2"].tolist() == [0, 3 + 3 + 3 + 3 + 2]
    assert V.news2_severity(14, True) == "CRITICO"
    assert V.news2_severity(3, True) == "ALTO"  # escore vermelho em um parâmetro
    assert V.news2_severity(2, False) is None


def test_artifacts_removed():
    df = pd.DataFrame({"stay_id": [1, 1], "charttime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                       "hr": [0, 80], "sbp": [120, 120], "dbp": [70, 70], "map": [801, 85],
                       "spo2": [98, 98], "rr": [16, 16], "temp": [37, 37]})
    clean, log = V.remove_artifacts(df)
    assert np.isnan(clean.loc[0, "hr"]) and np.isnan(clean.loc[0, "map"])
    assert set(log["signal"]) == {"hr", "map"}


def test_stream_monitor_flags_desaturation():
    rng = np.random.default_rng(0)
    rows = [{"stay_id": 1, "charttime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i),
             "hr": 80 + rng.normal(0, 2), "sbp": 120 + rng.normal(0, 3), "dbp": 70, "map": 85,
             "spo2": 97 + rng.normal(0, 0.5), "rr": 16, "temp": 37.0} for i in range(30)]
    iff = V.VitalsIForest().fit(pd.DataFrame(rows))
    mon = V.VitalsStreamMonitor(iforest=iff)
    for r in rows:
        mon.update(r)
    bad = dict(rows[-1], charttime=rows[-1]["charttime"] + pd.Timedelta(hours=1), spo2=85, rr=28, hr=120)
    keys = {a["key"] for a in mon.update(bad)}
    assert "news2" in keys and "z_spo2" in keys


def test_parse_dose_and_class():
    assert P._parse_dose("25,000") == 25000
    assert P._parse_dose("1-2") == 2
    assert np.isnan(P._parse_dose(None))
    assert P.drug_class("HYDROmorphone (Dilaudid)") == "opioide"
    assert P.drug_class("Acetaminophen") is None


def test_critical_terms_portuguese():
    found = critical_terms("Estou com falta de ar e dor no peito, a fala enrolada")
    cats = {f["categoria"] for f in found}
    assert {"respiratoria", "cardiaca", "neurologica_avc"} <= cats


def test_fusion_noisy_or_and_corroboration():
    t0 = pd.Timestamp("2026-01-01 10:00")
    ev = pd.DataFrame([
        {"time": t0, "modality": "vitais", "severity": "ALTO", "title": "NEWS2 = 6"},
        {"time": t0 + pd.Timedelta(hours=1), "modality": "audio", "severity": "ALTO", "title": "Dispneia"},
    ])
    fe = FusionEngine()
    r = fe.risk_at(ev, t0 + pd.Timedelta(hours=1))
    assert r["risk"] > max(r["by_modality"].values())  # combinação supera cada modalidade isolada
    fused = fe.corroboration_alerts(ev)
    assert fused and fused[-1]["severity"] == "CRITICO"
    later = fe.risk_at(ev, t0 + pd.Timedelta(hours=24))["risk"]
    assert later < r["risk"]  # decaimento temporal


def test_pose_angles():
    kp = np.full((18, 3), np.nan)
    kp[1] = [100, 50, 1]   # pescoço
    kp[8] = [95, 150, 1]   # quadril D
    kp[11] = [105, 150, 1]  # quadril E
    assert torso_angle(kp) == pytest.approx(0, abs=1)
    kp[1] = [200, 150, 1]  # tronco horizontal
    assert torso_angle(kp) == pytest.approx(90, abs=1)
    kp[9] = [100, 200, 1]
    kp[10] = [100, 250, 1]
    kp[8] = [100, 150, 1]
    assert joint_angle(kp, 8, 9, 10) == pytest.approx(180, abs=1)


def test_alert_bus_roundtrip(tmp_path):
    bus = AlertBus(path=tmp_path / "a.jsonl", webhook_url=None, echo=False)
    bus.publish(Alert(patient_id="Leito X", modality="video", severity="CRITICO", title="Queda",
                      description="teste", source="unit"))
    out = load_alerts(tmp_path / "a.jsonl")
    assert out[0]["notify"][-1] == "Time de resposta rápida"
    with pytest.raises(ValueError):
        Alert(patient_id="x", modality="x", severity="URGENTE", title="", description="", source="")
