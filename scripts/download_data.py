"""Baixa os dados públicos e os pesos dos modelos usados no projeto.

  * MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, ODbL) — sinais vitais,
    prescrições e registros de movimentação de 100 pacientes;
  * UR Fall Detection Dataset (Kwolek & Kepski, 2014) — 30 quedas + 40 ADL, câmera 0;
  * ESC-50 (metadados; os clipes de tosse/respiração são baixados sob demanda);
  * OpenPose body_pose_model.pth (pesos CMU convertidos para PyTorch);
  * YOLOv8s (Ultralytics) é baixado automaticamente no primeiro uso.

Uso: python -m scripts.download_data
"""
from __future__ import annotations

from pathlib import Path

import requests

from src.config import DATA_RAW, MODELS_DIR

MIMIC = "https://physionet.org/files/mimic-iv-demo/2.2/"
MIMIC_FILES = ["hosp/prescriptions.csv.gz", "hosp/patients.csv.gz", "hosp/admissions.csv.gz",
               "icu/chartevents.csv.gz", "icu/d_items.csv.gz", "icu/icustays.csv.gz", "LICENSE.txt"]
URFD = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
OPENPOSE = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
ESC50_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"ok  {dest.relative_to(DATA_RAW.parent.parent)}")


def main() -> None:
    for f in MIMIC_FILES:
        fetch(MIMIC + f, DATA_RAW / "mimic_iv_demo" / Path(f).name)
    for i in range(1, 31):
        fetch(f"{URFD}fall-{i:02d}-cam0.mp4", DATA_RAW / "urfd" / f"fall-{i:02d}-cam0.mp4")
    for i in range(1, 41):
        fetch(f"{URFD}adl-{i:02d}-cam0.mp4", DATA_RAW / "urfd" / f"adl-{i:02d}-cam0.mp4")
    fetch(ESC50_META, DATA_RAW / "audio" / "esc50.csv")
    fetch(OPENPOSE, MODELS_DIR / "body_pose_model.pth")


if __name__ == "__main__":
    main()
