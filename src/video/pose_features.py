"""Features biomecânicas extraídas dos keypoints do OpenPose."""
from __future__ import annotations

import numpy as np

NECK, R_SH, R_EL, R_WR, L_SH, L_EL, L_WR = 1, 2, 3, 4, 5, 6, 7
R_HIP, R_KNEE, R_ANK, L_HIP, L_KNEE, L_ANK = 8, 9, 10, 11, 12, 13


def _pt(kp: np.ndarray, i: int) -> np.ndarray | None:
    p = kp[i, :2]
    return None if np.isnan(p).any() else p


def _mid(kp: np.ndarray, a: int, b: int) -> np.ndarray | None:
    pa, pb = _pt(kp, a), _pt(kp, b)
    if pa is None and pb is None:
        return None
    if pa is None:
        return pb
    if pb is None:
        return pa
    return (pa + pb) / 2


def joint_angle(kp: np.ndarray, a: int, b: int, c: int) -> float:
    """Ângulo (graus) no vértice b formado por a-b-c."""
    pa, pb, pc = _pt(kp, a), _pt(kp, b), _pt(kp, c)
    if pa is None or pb is None or pc is None:
        return np.nan
    v1, v2 = pa - pb, pc - pb
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / den, -1, 1))))


def torso_angle(kp: np.ndarray) -> float:
    """Inclinação do tronco em relação à vertical (0 = em pé, 90 = deitado, >90 = invertido)."""
    neck = _pt(kp, NECK)
    if neck is None:
        neck = _mid(kp, R_SH, L_SH)
    hip = _mid(kp, R_HIP, L_HIP)
    if neck is None or hip is None:
        return np.nan
    dx, dy = neck - hip
    # dy negativo = pescoço acima do quadril (imagem cresce para baixo)
    return float(np.degrees(np.arctan2(abs(dx), -dy)))


def frame_features(kp: np.ndarray, frame_h: int) -> dict[str, float]:
    valid = ~np.isnan(kp[:, 0])
    ys = kp[valid, 1]
    xs = kp[valid, 0]
    centroid_y = float(np.mean(ys) / frame_h) if valid.any() else np.nan
    extent_h = float((ys.max() - ys.min()) / frame_h) if valid.sum() >= 2 else np.nan
    extent_w = float((xs.max() - xs.min()) / frame_h) if valid.sum() >= 2 else np.nan
    return {
        "n_keypoints": int(valid.sum()),
        "torso_angle": torso_angle(kp),
        "centroid_y": centroid_y,
        "pose_aspect": extent_w / extent_h if extent_h and extent_h > 1e-3 else np.nan,
        "knee_r": joint_angle(kp, R_HIP, R_KNEE, R_ANK),
        "knee_l": joint_angle(kp, L_HIP, L_KNEE, L_ANK),
        "hip_r": joint_angle(kp, NECK, R_HIP, R_KNEE),
        "hip_l": joint_angle(kp, NECK, L_HIP, L_KNEE),
        "elbow_r": joint_angle(kp, R_SH, R_EL, R_WR),
        "elbow_l": joint_angle(kp, L_SH, L_EL, L_WR),
        "shoulder_r": joint_angle(kp, R_HIP, R_SH, R_EL),
        "shoulder_l": joint_angle(kp, L_HIP, L_SH, L_EL),
    }


def motion_energy(prev: np.ndarray | None, cur: np.ndarray, frame_h: int) -> float:
    if prev is None:
        return np.nan
    both = ~np.isnan(prev[:, 0]) & ~np.isnan(cur[:, 0])
    if both.sum() < 3:
        return np.nan
    return float(np.mean(np.linalg.norm(cur[both, :2] - prev[both, :2], axis=1)) / frame_h)
