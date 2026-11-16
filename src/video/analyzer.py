"""Análise de vídeo clínico: OpenPose (postura) + YOLOv8 (objetos e áreas críticas).

Pipeline por vídeo:
  1. `extract`  -> para cada frame: keypoints OpenPose do paciente, features
                   biomecânicas, detecções YOLOv8 (pessoa, cama, sofá, cadeira...)
                   e ocupação das zonas críticas;
  2. `detect_events` -> regras clínicas sobre as séries temporais
                   (queda, permanência no chão, imobilidade pós-queda,
                    desvios posturais/assimetria em exercício);
  3. `render`   -> vídeo anotado (esqueleto, caixas, zonas, status, alertas);
  4. `build_report` -> relatório automático (JSON + Markdown) com os desvios.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.video.openpose import OpenPose, draw_pose
from src.video.pose_features import frame_features, motion_energy

# Classes COCO relevantes para o ambiente hospitalar
YOLO_CLASSES = {0: "pessoa", 56: "cadeira", 57: "sofa", 59: "cama", 60: "mesa",
                61: "vaso sanitario", 43: "faca", 76: "tesoura", 39: "frasco", 67: "celular"}
SUPPORT_CLASSES = {"cama", "sofa"}  # superfícies onde DEITAR é esperado (não se deita em cadeira)


@dataclass
class VideoConfig:
    crop: str = "none"  # none | right_half (UR Fall: profundidade | RGB)
    target_h: int = 360
    frame_step: int = 1
    # zona crítica "piso": faixa inferior da imagem (coordenadas normalizadas)
    floor_zone_y: float = 0.62
    # limiares das regras (calibrados no conjunto de treino do UR Fall — ver relatório)
    fall_drop_rate: float = 0.55  # queda da cabeça em alturas-de-frame por segundo
    fall_window_s: float = 0.5
    lying_torso_deg: float = 55.0
    lying_aspect: float = 1.15
    min_ground_s: float = 0.8
    immobility_motion: float = 0.004
    immobility_s: float = 2.0
    asym_deg: float = 25.0


@dataclass
class VideoEvent:
    kind: str
    severity: str
    t_start: float
    t_end: float
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


class VideoAnalyzer:
    def __init__(self, cfg: VideoConfig | None = None, yolo_weights: str = "yolov8s.pt") -> None:
        from ultralytics import YOLO

        self.cfg = cfg or VideoConfig()
        self.pose = OpenPose(MODELS_DIR / "body_pose_model.pth")
        w = MODELS_DIR / yolo_weights
        self.yolo = YOLO(str(w) if w.exists() else yolo_weights)
        if not w.exists() and Path(yolo_weights).exists():
            Path(yolo_weights).replace(w)

    # ------------------------------------------------------------------ leitura
    def _frames(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % self.cfg.frame_step == 0:
                if self.cfg.crop == "right_half":
                    frame = frame[:, frame.shape[1] // 2:]
                h, w = frame.shape[:2]
                s = self.cfg.target_h / h
                frame = cv2.resize(frame, (int(round(w * s)), self.cfg.target_h))
                yield i, i / fps, fps, frame
            i += 1
        cap.release()

    # --------------------------------------------------------------- extração
    def extract(self, path: str | Path) -> tuple[pd.DataFrame, list[np.ndarray], float]:
        path = Path(path)
        rows, kps = [], []
        prev_kp, fps = None, 30.0
        for idx, t, fps, frame in self._frames(path):
            H, W = frame.shape[:2]
            det = self.yolo.predict(frame, verbose=False, conf=0.35, classes=list(YOLO_CLASSES))[0]
            objs = []
            for b, c, s in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy().astype(int),
                               det.boxes.conf.cpu().numpy()):
                objs.append({"cls": YOLO_CLASSES.get(c, str(c)), "box": b.tolist(), "conf": float(s)})
            persons = [o for o in objs if o["cls"] == "pessoa"]
            pbox = max(persons, key=lambda o: o["conf"])["box"] if persons else None

            people = self.pose(frame)
            kp = self._select_patient(people, pbox)
            feats = frame_features(kp, H) if kp is not None else {}
            me = motion_energy(prev_kp, kp, H) if kp is not None else np.nan
            prev_kp = kp if kp is not None else prev_kp

            head_y = np.nan
            if kp is not None:
                for j in (1, 0, 14, 15, 16, 17, 2, 5):  # pescoço, nariz, olhos, orelhas, ombros
                    if not np.isnan(kp[j, 1]):
                        head_y = kp[j, 1] / H
                        break
            if np.isnan(head_y) and pbox is not None:
                head_y = pbox[1] / H

            bbox_aspect = np.nan
            on_support = False
            if pbox is not None:
                bw, bh = pbox[2] - pbox[0], pbox[3] - pbox[1]
                bbox_aspect = bw / max(bh, 1)
                for o in objs:
                    if o["cls"] in SUPPORT_CLASSES and _iou_over(pbox, o["box"]) > 0.35:
                        on_support = True
            in_floor = bool(pbox is not None and (pbox[1] / H) > self.cfg.floor_zone_y - 0.25
                            and (pbox[3] / H) > self.cfg.floor_zone_y)
            rows.append({"frame": idx, "t": t, "has_pose": kp is not None, "has_person": pbox is not None,
                         "head_y": head_y, "bbox_aspect": bbox_aspect, "on_support": on_support,
                         "in_floor_zone": in_floor, "motion": me,
                         "objects": json.dumps([o["cls"] for o in objs], ensure_ascii=False),
                         "object_boxes": json.dumps([[o["cls"], [round(v, 1) for v in o["box"]]] for o in objs],
                                                    ensure_ascii=False),
                         "pbox": json.dumps(pbox), **feats})
            kps.append(kp if kp is not None else np.full((18, 3), np.nan))
        df = pd.DataFrame(rows)
        df.attrs["fps"] = fps / self.cfg.frame_step
        return df, kps, fps

    @staticmethod
    def _select_patient(people: list[np.ndarray], pbox) -> np.ndarray | None:
        if not people:
            return None
        def score(kp):
            n = int((~np.isnan(kp[:, 0])).sum())
            if pbox is None:
                return n
            x1, y1, x2, y2 = pbox
            inside = np.nansum((kp[:, 0] >= x1) & (kp[:, 0] <= x2) & (kp[:, 1] >= y1) & (kp[:, 1] <= y2))
            return inside * 2 + n
        return max(people, key=score)

    # ----------------------------------------------------------- eventos
    def detect_events(self, df: pd.DataFrame) -> tuple[list[VideoEvent], pd.DataFrame]:
        cfg = self.cfg
        if df.empty:
            return [], df
        df = df.copy()
        dt = float(np.median(np.diff(df["t"]))) if len(df) > 1 else 1 / 30
        k = max(3, int(round(0.2 / dt)) | 1)
        for col in ("head_y", "torso_angle", "bbox_aspect", "motion"):
            if col in df:
                df[col + "_s"] = df[col].interpolate(limit_direction="both").rolling(k, center=True, min_periods=1).median()
            else:
                df[col + "_s"] = np.nan
        win = max(1, int(round(cfg.fall_window_s / dt)))
        df["drop_rate"] = (df["head_y_s"] - df["head_y_s"].shift(win)) / (win * dt)
        df["lying"] = (df["torso_angle_s"] > cfg.lying_torso_deg) | (df["bbox_aspect_s"] > cfg.lying_aspect)
        df["low"] = df["head_y_s"] > cfg.floor_zone_y
        df["on_ground"] = (df["lying"] | df["low"]) & ~df["on_support"]
        df["posture"] = np.select(
            [df["on_ground"], df["lying"], df["torso_angle_s"] > 30, df["head_y_s"].notna()],
            ["no chao", "deitado", "inclinado/sentado", "em pe"], default="indeterminado")

        events: list[VideoEvent] = []
        ground_frames = int(round(cfg.min_ground_s / dt))
        i, n = 0, len(df)
        while i < n:
            if df["drop_rate"].iat[i] >= cfg.fall_drop_rate:
                look = df.iloc[i: min(n, i + int(round(2.0 / dt)))]
                g = look["on_ground"].to_numpy()
                run, best, start = 0, 0, None
                for j, v in enumerate(g):
                    run = run + 1 if v else 0
                    if run > best:
                        best = run
                        if run == 1:
                            start = j
                if best >= ground_frames:
                    t0 = float(df["t"].iat[i])
                    # quanto tempo permanece no chão após a queda
                    after = df.iloc[i:]
                    stay = after["on_ground"].to_numpy()
                    j = int(np.argmax(stay)) if stay.any() else 0
                    end_idx = j
                    while end_idx < len(stay) and stay[end_idx]:
                        end_idx += 1
                    t_ground_end = float(after["t"].iat[min(end_idx, len(after) - 1)])
                    events.append(VideoEvent(
                        kind="queda", severity="CRITICO", t_start=t0 - cfg.fall_window_s, t_end=t_ground_end,
                        description=(f"Queda detectada: descida rápida da cabeça "
                                     f"({df['drop_rate'].iat[i]:.2f} alturas/s) seguida de postura "
                                     f"horizontal/no piso por {t_ground_end - t0:.1f}s fora de cama/sofá."),
                        evidence={"drop_rate": round(float(df["drop_rate"].iat[i]), 3),
                                  "torso_angle_max": round(float(after["torso_angle_s"].max()), 1),
                                  "bbox_aspect_max": round(float(after["bbox_aspect_s"].max()), 2),
                                  "objetos_em_cena": sorted(set(sum((json.loads(o) for o in df["objects"]), [])))}))
                    # imobilidade após a queda
                    ground = after.iloc[j:end_idx]
                    if len(ground) * dt >= cfg.immobility_s:
                        still = ground["motion_s"] < cfg.immobility_motion
                        if still.mean() > 0.7:
                            events.append(VideoEvent(
                                kind="imobilidade_pos_queda", severity="CRITICO",
                                t_start=float(ground["t"].iat[0]), t_end=float(ground["t"].iat[-1]),
                                description="Paciente permanece imóvel no piso após a queda — possível perda de consciência.",
                                evidence={"duracao_s": round(len(ground) * dt, 1),
                                          "motion_mediana": round(float(ground["motion_s"].median()), 5)}))
                    i = i + max(end_idx, 1) + int(round(1 / dt))
                    continue
            i += 1

        # Desvios posturais durante exercício/deambulação (fisioterapia)
        if not any(e.kind == "queda" for e in events):
            for side_a, side_b, label in (("knee_r", "knee_l", "joelhos"), ("elbow_r", "elbow_l", "cotovelos")):
                if side_a in df and side_b in df:
                    diff = (df[side_a] - df[side_b]).abs().rolling(k, center=True, min_periods=1).median()
                    upright = df["posture"].isin(["em pe", "inclinado/sentado"])
                    mask = (diff > cfg.asym_deg) & upright
                    if mask.sum() * dt > 1.0:
                        events.append(VideoEvent(
                            kind=f"assimetria_{label}", severity="ATENCAO",
                            t_start=float(df.loc[mask, "t"].min()), t_end=float(df.loc[mask, "t"].max()),
                            description=(f"Assimetria entre {label} D/E acima de {cfg.asym_deg:.0f}° por "
                                         f"{mask.sum() * dt:.1f}s — possível compensação/padrão anômalo de movimento."),
                            evidence={"diferenca_mediana_graus": round(float(diff[mask].median()), 1)}))
        return events, df

    # ------------------------------------------------------------ render
    def render(self, path: str | Path, df: pd.DataFrame, kps: list[np.ndarray], events: list[VideoEvent],
               out_path: str | Path, title: str = "", playback: float = 0.5) -> Path:
        """Gera o vídeo anotado. `playback` < 1 grava em câmera lenta (melhor para apresentação)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.mp4")
        writer = None
        fps = df.attrs.get("fps", 30.0)
        for (idx, t, _, frame), (_, row), kp in zip(self._frames(Path(path)), df.iterrows(), kps):
            H, W = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps * playback, (W, H + 70))
            overlay = frame.copy()
            fy = int(self.cfg.floor_zone_y * H)
            cv2.rectangle(overlay, (0, fy), (W, H), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.putText(frame, "ZONA CRITICA: PISO", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            pbox = json.loads(row["pbox"]) if isinstance(row.get("pbox"), str) else None
            active = [e for e in events if e.t_start <= t <= e.t_end + 1.5]
            color = (0, 0, 255) if active else (0, 200, 0)
            if pbox:
                x1, y1, x2, y2 = map(int, pbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"YOLOv8 pessoa | {row.get('posture', '')}", (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            if not np.isnan(kp[:, 0]).all():
                draw_pose(frame, kp)
            panel = np.full((70, W, 3), 25, np.uint8)
            ang, dr = row.get("torso_angle_s", np.nan), row.get("drop_rate", np.nan)
            ang_s = f"{ang:5.1f}deg" if np.isfinite(ang) else "--"
            dr_s = f"{dr:+.2f}/s" if np.isfinite(dr) else "--"
            speed = f"  [{playback:g}x]" if playback != 1 else ""
            cv2.putText(panel, f"{title}  t={t:5.2f}s  tronco={ang_s}  queda_cabeca={dr_s}{speed}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
            if active:
                msg = " | ".join(e.kind.upper() for e in active)
                cv2.putText(panel, f"ALERTA: {msg}", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(panel, "Monitorando: OpenPose + YOLOv8", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            writer.write(np.vstack([frame, panel]))
        if writer:
            writer.release()
        # H.264 para tocar no navegador / dashboard
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out_path)], check=True)
        tmp.unlink(missing_ok=True)
        return out_path

    def keyframe(self, path: str | Path, df: pd.DataFrame, kps: list[np.ndarray], t: float, out_png: str | Path) -> Path:
        i = int((df["t"] - t).abs().idxmin())
        for (idx, tt, _, frame) in self._frames(Path(path)):
            if idx == df["frame"].iat[i]:
                draw_pose(frame, kps[i], color=(0, 0, 255))
                pbox = json.loads(df["pbox"].iat[i])
                if pbox:
                    x1, y1, x2, y2 = map(int, pbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                ok, buf = cv2.imencode(".png", frame)
                Path(out_png).write_bytes(buf.tobytes())
                break
        return Path(out_png)

    # ------------------------------------------------------------ relatório
    def build_report(self, video_name: str, df: pd.DataFrame, events: list[VideoEvent]) -> dict[str, Any]:
        dt = float(np.median(np.diff(df["t"]))) if len(df) > 1 else 1 / 30
        posture_time = (df["posture"].value_counts() * dt).round(1).to_dict() if "posture" in df else {}
        rom = {}
        for col in ("knee_r", "knee_l", "hip_r", "hip_l", "elbow_r", "elbow_l", "shoulder_r", "shoulder_l"):
            if col in df and df[col].notna().sum() > 5:
                rom[col] = {"min": round(float(df[col].quantile(0.05)), 1),
                            "max": round(float(df[col].quantile(0.95)), 1)}
        return {
            "video": video_name,
            "duracao_s": round(float(df["t"].iloc[-1]) if len(df) else 0, 2),
            "frames_analisados": int(len(df)),
            "cobertura_pose_pct": round(100 * float(df["has_pose"].mean()), 1) if len(df) else 0,
            "tempo_por_postura_s": posture_time,
            "amplitude_articular_graus": rom,
            "eventos": [asdict(e) for e in events],
            "status": "ANOMALIA" if any(e.severity in ("ALTO", "CRITICO") for e in events)
                      else ("ATENCAO" if events else "NORMAL"),
        }


def report_markdown(rep: dict[str, Any]) -> str:
    lines = [f"# Relatório automático de vídeo — {rep['video']}", "",
             f"- **Status:** {rep['status']}", f"- **Duração:** {rep['duracao_s']} s",
             f"- **Frames analisados:** {rep['frames_analisados']} (pose detectada em {rep['cobertura_pose_pct']}%)",
             "", "## Tempo por postura (s)", ""]
    lines += [f"- {k}: {v}" for k, v in rep["tempo_por_postura_s"].items()]
    lines += ["", "## Eventos / desvios detectados", ""]
    if not rep["eventos"]:
        lines.append("Nenhum desvio detectado.")
    for e in rep["eventos"]:
        lines.append(f"- **[{e['severity']}] {e['kind']}** ({e['t_start']:.2f}s – {e['t_end']:.2f}s): {e['description']}")
    lines += ["", "## Amplitude articular (p5–p95, graus)", "", "| articulação | mín | máx |", "|---|---|---|"]
    lines += [f"| {k} | {v['min']} | {v['max']} |" for k, v in rep["amplitude_articular_graus"].items()]
    return "\n".join(lines) + "\n"


def _iou_over(a, b) -> float:
    """Fração da caixa `a` sobreposta por `b`."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area
