"""Extrai séries temporais (OpenPose + YOLOv8) de todos os vídeos do UR Fall Dataset.

Uso: python -m scripts.extract_video_features [--step 2]
Saída: data/processed/video/<nome>.parquet e <nome>_kps.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED, DATA_RAW
from src.video.analyzer import VideoAnalyzer, VideoConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--pattern", default="*-cam0.mp4")
    args = ap.parse_args()
    out = DATA_PROCESSED / "video"
    out.mkdir(parents=True, exist_ok=True)
    va = VideoAnalyzer(VideoConfig(crop="right_half", frame_step=args.step))
    videos = sorted((DATA_RAW / "urfd").glob(args.pattern))
    t0 = time.time()
    for k, v in enumerate(videos, 1):
        name = v.stem
        if (out / f"{name}.parquet").exists():
            continue
        df, kps, fps = va.extract(v.relative_to(Path.cwd()) if v.is_relative_to(Path.cwd()) else v)
        df.to_parquet(out / f"{name}.parquet")
        np.save(out / f"{name}_kps.npy", np.stack(kps))
        print(f"[{k}/{len(videos)}] {name}: {len(df)} frames, pose {df['has_pose'].mean():.0%} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
