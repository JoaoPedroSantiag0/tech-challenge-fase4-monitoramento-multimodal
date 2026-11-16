"""Executa o pipeline multimodal completo e gera todos os artefatos em outputs/.

  1. Vídeo  — calibração (treino) e avaliação (teste) do detector de quedas no
              UR Fall; vídeos anotados + relatórios automáticos dos vídeos do cenário;
  2. Áudio  — análise das 4 consultas (Azure Speech + Text Analytics, AST/AudioSet,
              biomarcadores vocais) com relatórios;
  3. Vitais — avaliação dos detectores com anomalias injetadas;
  4. Prescrições e movimentação — detecção em toda a base MIMIC-IV Demo;
  5. Cenário — linha do tempo multimodal de 4 leitos, fusão, alertas à equipe.

Uso:  python -m scripts.run_pipeline [--skip-video-render] [--skip-audio]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.alerts import Alert, AlertBus
from src.config import AZURE, DATA_PROCESSED, DATA_RAW, OUTPUTS

OUT_V, OUT_A = OUTPUTS / "video", OUTPUTS / "audio"
OUT_VI, OUT_P, OUT_M, OUT_S = OUTPUTS / "vitais", OUTPUTS / "prescricoes", OUTPUTS / "movimentacao", OUTPUTS / "cenario"
for d in (OUT_V, OUT_A, OUT_VI, OUT_P, OUT_M, OUT_S):
    d.mkdir(parents=True, exist_ok=True)


def _json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# ------------------------------------------------------------------ 1. vídeo
def step_video(scn_videos: list[str], render: bool) -> dict:
    from src.video import evaluation as E
    from src.video.analyzer import VideoConfig, report_markdown

    t0 = time.time()
    series = E.load_series(DATA_PROCESSED / "video")
    base = VideoConfig(crop="right_half")
    cfg, grid = E.calibrate(series, base)
    grid.head(20).to_csv(OUT_V / "calibracao_grid_top20.csv", index=False)
    res = E.run(series, cfg)
    res.to_csv(OUT_V / "avaliacao_por_video.csv", index=False)
    m = {"treino": E.metrics(res[res["split"] == "train"]), "teste": E.metrics(res[res["split"] == "test"]),
         "limiares_calibrados": {k: getattr(cfg, k) for k in E.GRID},
         "n_videos": {"treino": int((res["split"] == "train").sum()), "teste": int((res["split"] == "test").sum())}}
    _json(m, OUT_V / "metricas_quedas.json")
    print(f"[vídeo] treino {m['treino']}\n[vídeo] teste  {m['teste']}  ({time.time() - t0:.0f}s)")

    reports = {}
    if render:
        from src.video.analyzer import VideoAnalyzer

        cfg_r = VideoConfig(**{**asdict(cfg), "crop": "right_half", "frame_step": 1})
        va = VideoAnalyzer(cfg_r)
        for v in scn_videos:
            path = Path(v)
            df, kps, fps = va.extract(path)
            events, d = va.detect_events(df)
            rep = va.build_report(path.stem, d, events)
            va.render(path, d, kps, events, OUT_V / f"{path.stem}_anotado.mp4", title=path.stem)
            key_t = events[0].t_start + 1.0 if events else float(d["t"].iloc[len(d) // 2])
            va.keyframe(path, d, kps, key_t, OUT_V / f"{path.stem}_keyframe.png")
            d.drop(columns=[c for c in ("objects", "object_boxes", "pbox") if c in d]).to_parquet(OUT_V / f"{path.stem}_serie.parquet")
            _json(rep, OUT_V / f"{path.stem}_relatorio.json")
            (OUT_V / f"{path.stem}_relatorio.md").write_text(report_markdown(rep), encoding="utf-8")
            reports[path.stem] = rep
            print(f"[vídeo] {path.stem}: {rep['status']} {[e['kind'] for e in rep['eventos']]}")
    else:
        for v in scn_videos:
            p = OUT_V / f"{Path(v).stem}_relatorio.json"
            if p.exists():
                reports[Path(v).stem] = json.loads(p.read_text(encoding="utf-8"))
    return {"metricas": m, "relatorios": reports}


# ------------------------------------------------------------------ 2. áudio
def step_audio(run: bool) -> dict:
    out = {}
    files = sorted((DATA_RAW / "audio" / "consultas").glob("*.wav"))
    if run:
        from src.audio.analyzer import analyze_audio

        for f in files:
            t0 = time.time()
            r = analyze_audio(f)
            r["tempo_processamento_s"] = round(time.time() - t0, 1)
            _json(r, OUT_A / f"{f.stem}.json")
            out[f.stem] = r
            print(f"[áudio] {f.stem}: {r['status']} via {r['transcricao']['provider']} / "
                  f"{r['nlp'].get('provider')} — \"{r['transcricao']['text'][:70]}...\"")
    else:
        for f in files:
            p = OUT_A / f"{f.stem}.json"
            if p.exists():
                out[f.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


# ------------------------------------------------------------------ 3-4. tabulares
def step_tabular() -> dict:
    from src.anomaly import movement as M
    from src.anomaly import prescriptions as P
    from src.anomaly import vitals as V

    vit = V.load_vitals()
    clean, art = V.remove_artifacts(vit)
    art.to_csv(OUT_VI / "artefatos_removidos.csv", index=False)
    stays = clean["stay_id"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(stays)
    tr = clean[clean["stay_id"].isin(stays[: int(0.7 * len(stays))])]
    te = clean[clean["stay_id"].isin(stays[int(0.7 * len(stays)):])].reset_index(drop=True)
    iff = V.VitalsIForest().fit(tr)
    inj, lab = V.inject_anomalies(te)
    res = V.analyze(inj, iff)
    ev = V.evaluate(res, lab)
    ev.to_csv(OUT_VI / "avaliacao_detectores.csv", index=False)
    real = V.analyze(clean.reset_index(drop=True), iff)
    vit_summary = {
        "n_internacoes": int(clean["stay_id"].nunique()), "n_medicoes": int(len(clean)),
        "artefatos_removidos": int(len(art)), "artefatos_por_sinal": art["signal"].value_counts().to_dict(),
        "eventos_injetados": int((lab != lab.shift()).cumsum().where(lab).nunique()),
        "avaliacao": ev.to_dict("records"),
        "taxa_sinalizacao_dados_reais": {"NEWS2>=5": round(float(real["news_anomaly"].mean()), 3),
                                          "z-score": round(float(real["z_anomaly"].mean()), 3),
                                          "IsolationForest": round(float(real["if_anomaly"].mean()), 3),
                                          "ensemble>=2": round(float((real["ensemble_votes"] >= 2).mean()), 3)},
    }
    print("[vitais]\n" + ev.to_string(index=False))

    pr = P.load_prescriptions()
    pev = P.detect_rule_anomalies(pr)
    pev.to_csv(OUT_P / "eventos_prescricao.csv", index=False)
    io = P.iforest_orders(pr)
    top = io[io["if_anomaly"]].sort_values("if_score", ascending=False)
    top[["hadm_id", "starttime", "drug", "dose", "unit", "route", "dose_z_pop", "log_ratio_prev", "if_score"]].head(50).to_csv(
        OUT_P / "isolation_forest_top50.csv", index=False)
    presc_summary = {"n_ordens": int(len(pr)), "n_internacoes": int(pr["hadm_id"].nunique()),
                     "por_regra_severidade": pd.crosstab(pev["rule"], pev["severity"]).to_dict("index"),
                     "iforest_anomalias": int(io["if_anomaly"].sum()),
                     "concordancia_iforest_regras": _overlap(io, pev),
                     "exemplos": pev[pev["severity"] == "ALTO"].groupby("rule").head(2)[["rule", "detail"]].to_dict("records")}
    print(f"[prescrições] {len(pev)} eventos, IF {int(io['if_anomaly'].sum())} ordens atípicas")

    mv = M.load_movement()
    mev = M.detect_movement_anomalies(mv)
    mev.to_csv(OUT_M / "eventos_movimentacao.csv", index=False)
    mov_summary = {"n_registros": int(len(mv)), "n_internacoes": int(mv["stay_id"].nunique()),
                   "por_regra_severidade": pd.crosstab(mev["rule"], mev["severity"]).to_dict("index"),
                   "exemplos": mev.groupby("rule").head(2)[["rule", "detail"]].to_dict("records")}
    print(f"[movimentação] {len(mev)} eventos")
    return {"vitais": vit_summary, "prescricoes": presc_summary, "movimentacao": mov_summary}


def _overlap(io: pd.DataFrame, pev: pd.DataFrame) -> dict:
    a = io[io["if_anomaly"]][["hadm_id", "starttime"]].rename(columns={"starttime": "time"})
    m = a.merge(pev[["hadm_id", "time"]], on=["hadm_id", "time"], how="left", indicator=True)
    both = int((m.drop_duplicates(["hadm_id", "time"])["_merge"] == "both").sum())
    return {"ordens_if": int(len(a)), "tambem_sinalizadas_por_regra": both}


# ------------------------------------------------------------------ 5. cenário
def step_scenario(video_reports: dict, audio_reports: dict) -> dict:
    from src import scenario as S
    from src.fusion.risk import FusionEngine

    scn, pws, iff = S.build()
    bus = AlertBus(echo=False)
    bus.reset()
    all_events, summary = [], {}
    for pw in pws:
        pid = pw.spec["patient_id"]
        ev_v, trace = S.vitals_events(pw, iff)
        trace.to_parquet(OUT_S / f"{pid.replace(' ', '_')}_vitais.parquet")
        events = ev_v + S.table_events(pw.presc_events, "prescricao", "Regras de farmacovigilância (ISMP)") \
            + S.table_events(pw.move_events, "movimentacao", "Registros de enfermagem (MIMIC-IV)")
        if pw.audio_time is not None:
            a = audio_reports.get(Path(pw.spec["audio"]["file"]).stem)
            if a:
                events += _audio_events(a, pw.audio_time, pw.spec["audio"]["contexto"])
        if pw.video_time is not None:
            v = video_reports.get(Path(pw.spec["video"]["file"]).stem)
            if v:
                events += _video_events(v, pw.video_time, pw.spec["video"]["contexto"])
        edf = pd.DataFrame(events).sort_values("time").reset_index(drop=True)
        fe = FusionEngine()
        fused = fe.corroboration_alerts(edf)
        edf = pd.concat([edf, pd.DataFrame(fused)], ignore_index=True).sort_values("time").reset_index(drop=True)
        edf["patient_id"] = pid
        start = pd.Timestamp(scn["demo_start"])
        tl = fe.timeline(edf, start, start + pd.Timedelta(hours=scn["duration_h"]))
        tl.to_parquet(OUT_S / f"{pid.replace(' ', '_')}_risco.parquet")
        all_events.append(edf)
        summary[pid] = {"nome": pw.spec["nome"], "stay_id": pw.spec["stay_id"],
                        "eventos_por_modalidade": edf.groupby("modality")["severity"].value_counts().unstack(fill_value=0).to_dict("index"),
                        "risco_max": round(float(tl["risk"].max()), 3),
                        "hora_risco_max": tl.loc[tl["risk"].idxmax(), "time"].strftime("%H:%M"),
                        "alertas_fusao": [{"hora": f["time"].strftime("%H:%M"), "severidade": f["severity"], "titulo": f["title"]} for f in fused],
                        "audio": pw.spec.get("audio", {}).get("file"), "video": pw.spec.get("video", {}).get("file"),
                        "hora_audio": pw.audio_time.strftime("%H:%M") if pw.audio_time is not None else None,
                        "hora_video": pw.video_time.strftime("%H:%M") if pw.video_time is not None else None}
    allv = pd.concat(all_events, ignore_index=True).sort_values("time").reset_index(drop=True)
    allv["evidence"] = allv["evidence"].map(lambda e: json.dumps(e, ensure_ascii=False, default=_default))
    allv.to_parquet(OUT_S / "eventos.parquet")
    for _, e in allv.iterrows():
        bus.publish(Alert(patient_id=e["patient_id"], modality=e["modality"], severity=e["severity"], title=e["title"],
                          description=e["description"], source=e["source"], event_time=e["time"].isoformat(),
                          evidence=json.loads(e["evidence"])))
    cnt = allv.groupby(["patient_id", "severity"]).size().unstack(fill_value=0)
    print("[cenário] alertas por leito:\n" + cnt.to_string())
    return summary


def _audio_events(a: dict, t: pd.Timestamp, ctx: str) -> list[dict]:
    src = f"{a['transcricao']['provider']} + {a['nlp'].get('provider', '')} + AST/AudioSet + biomarcadores vocais"
    if not a["achados"]:
        return [{"time": t, "modality": "audio", "severity": "INFO", "title": "Consulta analisada — sem alterações vocais",
                 "description": f"{ctx}: \"{a['transcricao']['text'][:160]}\"", "source": src,
                 "evidence": {"indicadores": {k: v["score"] for k, v in a["indicadores_vocais"].items()}}}]
    return [{"time": t, "modality": "audio", "severity": f["severidade"],
             "title": f"{f['condicao'].replace('_', ' ').capitalize()} (consulta)",
             "description": f"{ctx}: {f['descricao']}. Evidências: " + "; ".join(f["evidencias"][:5]),
             "source": src, "evidence": {"transcricao": a["transcricao"]["text"][:300],
                                         "indicadores": {k: v["score"] for k, v in a["indicadores_vocais"].items()}}}
            for f in a["achados"]]


def _video_events(v: dict, t: pd.Timestamp, ctx: str) -> list[dict]:
    src = "OpenPose (pose) + YOLOv8 (objetos/zona crítica)"
    if not v["eventos"]:
        return [{"time": t, "modality": "video", "severity": "INFO", "title": "Vídeo analisado — movimento dentro do padrão",
                 "description": f"{ctx}: posturas {v['tempo_por_postura_s']}", "source": src,
                 "evidence": {"video": v["video"]}}]
    return [{"time": t + pd.Timedelta(seconds=e["t_start"]), "modality": "video", "severity": e["severity"],
             "title": e["kind"].replace("_", " ").capitalize() + " (vídeo)", "description": f"{ctx}: {e['description']}",
             "source": src, "evidence": {"video": v["video"], **e["evidence"]}} for e in v["eventos"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video-render", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    from src.scenario import load_scenario

    scn = load_scenario()
    scn_videos = [p["video"]["file"] for p in scn["patients"] if "video" in p]
    extra = ["data/raw/urfd/fall-01-cam0.mp4"]  # exemplo adicional para o relatório
    video = step_video(scn_videos + [v for v in extra if v not in scn_videos], render=not args.skip_video_render)
    audio = step_audio(run=not args.skip_audio)
    tab = step_tabular()
    scen = step_scenario(video["relatorios"], audio)
    summary = {"gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
               "azure": {"speech": AZURE.speech_enabled, "language": AZURE.language_enabled,
                         "regiao_speech": AZURE.speech_region},
               "video": video["metricas"], "video_relatorios": video["relatorios"],
               "audio": {k: {kk: vv for kk, vv in v.items() if kk != "audioset"} | {"audioset_max": v["audioset"]["max_prob"]}
                         for k, v in audio.items()},
               **tab, "cenario": scen, "tempo_total_s": round(time.time() - t0, 1)}
    _json(summary, OUTPUTS / "summary.json")
    print(f"\nOK — outputs/summary.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
