"""Simulação de monitoramento em TEMPO REAL dos 4 leitos.

Reproduz as 24h do cenário em velocidade acelerada. A cada observação:
  * sinais vitais passam pelo VitalsStreamMonitor (NEWS2 + z-score + Isolation Forest),
    calculado ao vivo, observação a observação;
  * prescrições/movimentação/áudio/vídeo entram no horário em que ocorreram
    (com --live-media o vídeo e o áudio são processados pelos modelos na hora);
  * a fusão multimodal reavalia o risco do leito e emite alertas de corroboração;
  * cada alerta é publicado no AlertBus -> Central de Alertas (dashboard) e webhook.

Uso:  python -m scripts.stream_demo [--speed 1800] [--live-media] [--start-h 0]
      (speed = segundos simulados por segundo real; 600 => 1h a cada 6s, 24h em ~2,5 min)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src import scenario as S
from src.anomaly import vitals as V
from src.common.alerts import Alert, AlertBus
from src.config import OUTPUTS
from src.fusion.risk import FusionEngine

STATE = OUTPUTS / "stream_state.json"


def _write_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(STATE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=600.0)
    ap.add_argument("--live-media", action="store_true", help="processa vídeo/áudio com os modelos no momento do evento")
    ap.add_argument("--start-h", type=float, default=0.0)
    args = ap.parse_args()

    print("Carregando cenário (MIMIC-IV, modelos)...")
    scn, pws, iff = S.build()
    start = pd.Timestamp(scn["demo_start"])
    end = start + pd.Timedelta(hours=scn["duration_h"])
    bus = AlertBus(echo=True)
    bus.reset()

    # fila de eventos externos (não-vitais)
    queue = []
    from scripts.run_pipeline import _audio_events, _video_events

    for pw in pws:
        pid = pw.spec["patient_id"]
        for e in S.table_events(pw.presc_events, "prescricao", "Regras de farmacovigilância (ISMP)") + \
                S.table_events(pw.move_events, "movimentacao", "Registros de enfermagem (MIMIC-IV)"):
            queue.append((e["time"], pid, "evento", e))
        if pw.audio_time is not None:
            queue.append((pw.audio_time, pid, "audio", pw.spec["audio"]))
        if pw.video_time is not None:
            queue.append((pw.video_time, pid, "video", pw.spec["video"]))
        for _, row in pw.vitals.iterrows():
            queue.append((row["charttime"], pid, "vitais", row.to_dict()))
    queue.sort(key=lambda x: x[0])

    monitors = {pw.spec["patient_id"]: V.VitalsStreamMonitor(iforest=iff) for pw in pws}
    events = {pw.spec["patient_id"]: [] for pw in pws}
    fusion = FusionEngine()
    fused_seen = {pid: set() for pid in events}
    state = {"sim_time": None, "speed": args.speed, "running": True,
             "patients": {pw.spec["patient_id"]: {"nome": pw.spec["nome"], "vitais": {}, "news2": 0, "risk": 0.0,
                                                  "by_modality": {}} for pw in pws}}
    va = None

    def emit(pid: str, e: dict) -> None:
        events[pid].append(e)
        bus.publish(Alert(patient_id=pid, modality=e["modality"], severity=e["severity"], title=e["title"],
                          description=e["description"], source=e["source"], event_time=pd.Timestamp(e["time"]).isoformat(),
                          evidence=json.loads(json.dumps(e.get("evidence", {}), default=str))))
        edf = pd.DataFrame(events[pid])
        for f in fusion.corroboration_alerts(edf):
            k = (f["time"], f["severity"])
            if k not in fused_seen[pid] and f["time"] == pd.Timestamp(e["time"]):
                fused_seen[pid].add(k)
                events[pid].append(f)
                bus.publish(Alert(patient_id=pid, modality="fusao", severity=f["severity"], title=f["title"],
                                  description=f["description"], source=f["source"], event_time=f["time"].isoformat(),
                                  evidence=f["evidence"]))

    def refresh_risk(now: pd.Timestamp) -> None:
        """Recalcula o risco fundido de todos os leitos no instante simulado `now`."""
        for p_id, evs in events.items():
            if evs:
                edf = pd.DataFrame(evs)
                # eventos de vídeo podem estar alguns segundos à frente do relógio (t_start dentro do clipe)
                r = fusion.risk_at(edf, max(now, edf["time"].max()))
                state["patients"][p_id]["risk"] = r["risk"]
                state["patients"][p_id]["by_modality"] = r["by_modality"]

    sim_t0 = start + pd.Timedelta(hours=args.start_h)
    wall_t0 = time.time()
    print(f"Iniciando simulação {sim_t0:%d/%m %H:%M} -> {end:%H:%M} (x{args.speed:.0f})")
    for t, pid, kind, payload in queue:
        if t < sim_t0:
            continue
        target_wall = wall_t0 + (t - sim_t0).total_seconds() / args.speed
        while time.time() < target_wall:
            now = sim_t0 + pd.Timedelta(seconds=(time.time() - wall_t0) * args.speed)
            state["sim_time"] = now.isoformat()
            refresh_risk(now)
            _write_state(state)
            time.sleep(min(0.25, max(0, target_wall - time.time())))
        state["sim_time"] = t.isoformat()
        if kind == "vitais":
            for a in monitors[pid].update(payload):
                emit(pid, {"time": t, "modality": "vitais", "severity": a["severity"], "title": a["title"],
                           "description": S._vitals_desc(a), "source": S._vitals_src(a["key"]), "evidence": a["evidence"]})
            hist = pd.DataFrame(list(monitors[pid].history))
            st = state["patients"][pid]
            st["vitais"] = {s: (None if pd.isna(payload[s]) else round(float(payload[s]), 1)) for s in V.SIGNALS}
            st["news2"] = int(V.news2(hist).iloc[-1]["news2"])
        elif kind == "evento":
            emit(pid, payload)
        elif kind == "audio":
            print(f"  >> {pid}: analisando áudio da consulta ({Path(payload['file']).name})...")
            if args.live_media:
                from src.audio.analyzer import analyze_audio
                rep = analyze_audio(payload["file"])
            else:
                rep = json.loads((OUTPUTS / "audio" / f"{Path(payload['file']).stem}.json").read_text(encoding="utf-8"))
            for e in _audio_events(rep, t, payload["contexto"]):
                emit(pid, e)
        elif kind == "video":
            print(f"  >> {pid}: analisando vídeo ({Path(payload['file']).name}) com OpenPose + YOLOv8...")
            if args.live_media:
                from src.video.analyzer import VideoAnalyzer, VideoConfig
                m = json.loads((OUTPUTS / "video" / "metricas_quedas.json").read_text(encoding="utf-8"))
                if va is None:
                    va = VideoAnalyzer(VideoConfig(crop="right_half", **m["limiares_calibrados"]))
                df, kps, _ = va.extract(Path(payload["file"]))
                evs, d = va.detect_events(df)
                rep = va.build_report(Path(payload["file"]).stem, d, evs)
            else:
                rep = json.loads((OUTPUTS / "video" / f"{Path(payload['file']).stem}_relatorio.json").read_text(encoding="utf-8"))
            for e in _video_events(rep, t, payload["contexto"]):
                emit(pid, e)
        refresh_risk(t)
        _write_state(state)
    state["running"] = False
    _write_state(state)
    print("Simulação concluída.")


if __name__ == "__main__":
    main()
