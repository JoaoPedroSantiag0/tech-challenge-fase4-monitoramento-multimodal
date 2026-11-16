"""Classificação de eventos sonoros com o Audio Spectrogram Transformer (AST)
treinado no Google AudioSet (527 classes) — MIT/ast-finetuned-audioset-10-10-0.4593.

Usado para localizar sons respiratórios (respiração ofegante, sibilo, tosse,
arquejo) que acompanham a fala do paciente durante a consulta.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
RESPIRATORY = ["Breathing", "Wheeze", "Gasp", "Pant", "Cough", "Snoring", "Throat clearing", "Sneeze"]
PT = {"Breathing": "respiração", "Wheeze": "sibilo", "Gasp": "arquejo", "Pant": "ofegar",
      "Cough": "tosse", "Snoring": "ronco", "Throat clearing": "pigarro", "Sneeze": "espirro", "Speech": "fala"}


class AudioSetTagger:
    def __init__(self, device: str | None = None) -> None:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        self.labels = self.model.config.id2label
        self.idx = {v: k for k, v in self.labels.items()}

    @torch.no_grad()
    def tag(self, y: np.ndarray, sr: int = 16000, win_s: float = 3.0, hop_s: float = 1.0) -> dict:
        win, hop = int(win_s * sr), int(hop_s * sr)
        starts = list(range(0, max(1, len(y) - win + 1), hop)) or [0]
        segs = [y[s:s + win] for s in starts]
        inputs = self.fe(segs, sampling_rate=sr, return_tensors="pt").to(self.device)
        probs = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()
        timeline = []
        for s, p in zip(starts, probs):
            row = {"t": round(s / sr, 2)}
            for lab in RESPIRATORY + ["Speech"]:
                row[lab] = round(float(p[self.idx[lab]]), 3)
            top = np.argsort(p)[::-1][:3]
            row["top3"] = [(self.labels[int(i)], round(float(p[i]), 3)) for i in top]
            timeline.append(row)
        summary = {lab: round(float(probs[:, self.idx[lab]].max()), 3) for lab in RESPIRATORY + ["Speech"]}
        return {"max_prob": summary, "timeline": timeline}
