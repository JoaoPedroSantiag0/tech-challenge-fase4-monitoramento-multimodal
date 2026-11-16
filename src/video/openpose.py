"""OpenPose (CMU) — estimação de pose 2D multi-pessoa.

Implementação em PyTorch da rede original do OpenPose (Cao et al., CVPR 2017,
"Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"),
modelo COCO de 18 keypoints, com os pesos oficiais convertidos do Caffe
(`body_pose_model.pth`, mesmo arquivo usado pelo ControlNet/OpenPose).

Baseado na conversão aberta pytorch-openpose (Hzzone, Apache-2.0).

Saída por pessoa: array (18, 3) -> (x, y, confiança), NaN quando ausente.
Ordem dos keypoints (COCO-18 do OpenPose):
 0 nariz, 1 pescoço, 2 ombro D, 3 cotovelo D, 4 punho D, 5 ombro E,
 6 cotovelo E, 7 punho E, 8 quadril D, 9 joelho D, 10 tornozelo D,
 11 quadril E, 12 joelho E, 13 tornozelo E, 14 olho D, 15 olho E,
 16 orelha D, 17 orelha E
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KEYPOINT_NAMES = [
    "nariz", "pescoco", "ombro_d", "cotovelo_d", "punho_d", "ombro_e", "cotovelo_e",
    "punho_e", "quadril_d", "joelho_d", "tornozelo_d", "quadril_e", "joelho_e",
    "tornozelo_e", "olho_d", "olho_e", "orelha_d", "orelha_e",
]

# Pares de membros (1-indexado, como no OpenPose) e índices dos canais de PAF
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18], [3, 17], [6, 18]]
MAP_IDX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
           [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
           [55, 56], [37, 38], [45, 46]]
# Esqueleto para desenho (0-indexado)
SKELETON = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


def _make_layers(block: OrderedDict, no_relu: list[str]) -> nn.Sequential:
    layers = []
    for name, v in block.items():
        if "pool" in name:
            layers.append((name, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])))
        else:
            layers.append((name, nn.Conv2d(v[0], v[1], kernel_size=v[2], stride=v[3], padding=v[4])))
            if name not in no_relu:
                layers.append(("relu_" + name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class BodyPoseModel(nn.Module):
    """VGG-19 (10 primeiras camadas) + 6 estágios de dois ramos (PAF L1 e heatmap L2)."""

    def __init__(self) -> None:
        super().__init__()
        no_relu = ["conv5_5_CPM_L1", "conv5_5_CPM_L2"] + [
            f"Mconv7_stage{i}_L{b}" for i in range(2, 7) for b in (1, 2)]
        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]), ("conv1_2", [64, 64, 3, 1, 1]), ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]), ("conv2_2", [128, 128, 3, 1, 1]), ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]), ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]), ("conv3_4", [256, 256, 3, 1, 1]), ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]), ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3_CPM", [512, 256, 3, 1, 1]), ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])
        self.model0 = _make_layers(block0, no_relu)

        def stage1(b: int, out_c: int) -> OrderedDict:
            return OrderedDict([
                (f"conv5_1_CPM_L{b}", [128, 128, 3, 1, 1]), (f"conv5_2_CPM_L{b}", [128, 128, 3, 1, 1]),
                (f"conv5_3_CPM_L{b}", [128, 128, 3, 1, 1]), (f"conv5_4_CPM_L{b}", [128, 512, 1, 1, 0]),
                (f"conv5_5_CPM_L{b}", [512, out_c, 1, 1, 0]),
            ])

        def stage_t(i: int, b: int, out_c: int) -> OrderedDict:
            return OrderedDict(
                [(f"Mconv1_stage{i}_L{b}", [185, 128, 7, 1, 3])]
                + [(f"Mconv{k}_stage{i}_L{b}", [128, 128, 7, 1, 3]) for k in range(2, 6)]
                + [(f"Mconv6_stage{i}_L{b}", [128, 128, 1, 1, 0]),
                   (f"Mconv7_stage{i}_L{b}", [128, out_c, 1, 1, 0])])

        self.model1_1 = _make_layers(stage1(1, 38), no_relu)
        self.model1_2 = _make_layers(stage1(2, 19), no_relu)
        for i in range(2, 7):
            setattr(self, f"model{i}_1", _make_layers(stage_t(i, 1, 38), no_relu))
            setattr(self, f"model{i}_2", _make_layers(stage_t(i, 2, 19), no_relu))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.model0(x)
        l1, l2 = self.model1_1(feat), self.model1_2(feat)
        for i in range(2, 7):
            inp = torch.cat([l1, l2, feat], 1)
            l1 = getattr(self, f"model{i}_1")(inp)
            l2 = getattr(self, f"model{i}_2")(inp)
        return l1, l2  # PAFs (38 canais), heatmaps (19 canais)


class OpenPose:
    def __init__(self, weights: str | Path, device: str | None = None, boxsize: int = 368) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BodyPoseModel()
        raw = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = {}
        for k in self.model.state_dict().keys():  # "model0.conv1_1.weight" -> "conv1_1.weight"
            state[k] = raw[".".join(k.split(".")[1:])]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.boxsize, self.stride, self.pad_value = boxsize, 8, 128
        self.thre1, self.thre2 = 0.1, 0.05

    @torch.no_grad()
    def _maps(self, img: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Executa a rede e devolve heatmaps (GPU, 18xHxW) e PAFs (CPU, HxWx38) na resolução original."""
        h, w = img.shape[:2]
        scale = self.boxsize / h
        im = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ph = (self.stride - im.shape[0] % self.stride) % self.stride
        pw = (self.stride - im.shape[1] % self.stride) % self.stride
        im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT,
                                value=(self.pad_value,) * 3)
        t = torch.from_numpy(np.ascontiguousarray(
            np.transpose(im.astype(np.float32), (2, 0, 1))[None] / 256 - 0.5)).to(self.device)
        paf, heat = self.model(t)
        out = []
        for m in (heat, paf):
            m = F.interpolate(m, scale_factor=self.stride, mode="bicubic", align_corners=False)
            m = m[:, :, : im.shape[0] - ph, : im.shape[1] - pw]
            out.append(F.interpolate(m, size=(h, w), mode="bicubic", align_corners=False))
        heat_t = out[0][0, :18]
        paf_np = out[1][0].permute(1, 2, 0).float().cpu().numpy()
        return heat_t, paf_np

    @torch.no_grad()
    def _peaks(self, heat: torch.Tensor) -> list[list[tuple]]:
        """Suavização gaussiana (sigma=3) + supressão de não-máximos, na GPU."""
        sigma, radius = 3.0, 12
        x = torch.arange(-radius, radius + 1, device=heat.device, dtype=heat.dtype)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = (g / g.sum())
        hm = heat[:, None]
        hm = F.conv2d(F.pad(hm, (radius, radius, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
        hm = F.conv2d(F.pad(hm, (0, 0, radius, radius), mode="reflect"), g.view(1, 1, -1, 1))
        hm = hm[:, 0]
        p = F.pad(hm, (1, 1, 1, 1), value=-1.0)
        c = hm
        binary = ((c >= p[:, :-2, 1:-1]) & (c >= p[:, 2:, 1:-1]) & (c >= p[:, 1:-1, :-2])
                  & (c >= p[:, 1:-1, 2:]) & (c > self.thre1))
        idx = torch.nonzero(binary).cpu().numpy()  # (part, y, x)
        scores = heat[binary].float().cpu().numpy()
        all_peaks: list[list[tuple]] = [[] for _ in range(18)]
        counter = 0
        for (part, y, x), s in zip(idx, scores):  # nonzero já vem ordenado por parte
            all_peaks[part].append((int(x), int(y), float(s), counter))
            counter += 1
        return all_peaks

    def __call__(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        """Retorna lista de pessoas, cada uma um array (18, 3) [x, y, conf]."""
        heat, paf = self._maps(img_bgr)
        all_peaks = self._peaks(heat)
        counter = sum(len(p) for p in all_peaks)

        connection_all, special_k, mid_num = [], [], 10
        h = img_bgr.shape[0]
        for k, (a, b) in enumerate(LIMB_SEQ):
            score_mid = paf[:, :, [x - 19 for x in MAP_IDX[k]]]
            cand_a, cand_b = all_peaks[a - 1], all_peaks[b - 1]
            if not cand_a or not cand_b:
                special_k.append(k)
                connection_all.append([])
                continue
            cands = []
            for i, pa in enumerate(cand_a):
                for j, pb in enumerate(cand_b):
                    vec = np.subtract(pb[:2], pa[:2]).astype(float)
                    norm = max(0.001, math.hypot(vec[0], vec[1]))
                    vec /= norm
                    xs = np.linspace(pa[0], pb[0], num=mid_num).round().astype(int)
                    ys = np.linspace(pa[1], pb[1], num=mid_num).round().astype(int)
                    vx, vy = score_mid[ys, xs, 0], score_mid[ys, xs, 1]
                    s = vx * vec[0] + vy * vec[1]
                    prior = s.mean() + min(0.5 * h / norm - 1, 0)
                    if (s > self.thre2).sum() > 0.8 * len(s) and prior > 0:
                        cands.append([i, j, prior, prior + pa[2] + pb[2]])
            cands.sort(key=lambda c: c[2], reverse=True)
            conn = np.zeros((0, 5))
            for i, j, s, _ in cands:
                if i not in conn[:, 3] and j not in conn[:, 4]:
                    conn = np.vstack([conn, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(conn) >= min(len(cand_a), len(cand_b)):
                        break
            connection_all.append(conn)

        candidate = np.array([p for peaks in all_peaks for p in peaks]) if counter else np.zeros((0, 4))
        subset = -1 * np.ones((0, 20))
        for k in range(len(MAP_IDX)):
            if k in special_k:
                continue
            part_as, part_bs = connection_all[k][:, 0], connection_all[k][:, 1]
            ia, ib = np.array(LIMB_SEQ[k]) - 1
            for i in range(len(connection_all[k])):
                found, idx = 0, [-1, -1]
                for j in range(len(subset)):
                    if subset[j][ia] == part_as[i] or subset[j][ib] == part_bs[i]:
                        if found < 2:
                            idx[found] = j
                        found += 1
                if found == 1:
                    j = idx[0]
                    if subset[j][ib] != part_bs[i]:
                        subset[j][ib] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[int(part_bs[i]), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][ib] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[int(part_bs[i]), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[ia], row[ib] = part_as[i], part_bs[i]
                    row[-1] = 2
                    row[-2] = candidate[connection_all[k][i, :2].astype(int), 2].sum() + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
        keep = [i for i in range(len(subset)) if subset[i][-1] >= 4 and subset[i][-2] / subset[i][-1] >= 0.4]
        subset = subset[keep]

        people = []
        for row in subset:
            kp = np.full((18, 3), np.nan)
            for part in range(18):
                ci = int(row[part])
                if ci >= 0:
                    kp[part] = candidate[ci, :3]
            people.append(kp)
        return people


def draw_pose(img: np.ndarray, kp: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    for a, b in SKELETON:
        if not (np.isnan(kp[a, 0]) or np.isnan(kp[b, 0])):
            cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), color, 2)
    for x, y, _ in kp:
        if not np.isnan(x):
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    return img
