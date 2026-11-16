"""Gera as gravações de consulta usadas na demonstração da análise de áudio.

Não há dataset público de consultas médicas em PT-BR com fala patológica rotulada,
então os casos são SIMULADOS de forma controlada e documentada:
  * a fala do paciente é sintetizada com vozes neurais pt-BR (Microsoft Edge TTS);
  * marcadores vocais são aplicados por processamento de sinal (velocidade,
    pausas, energia, monotonia, irregularidade de ritmo, perda de nitidez);
  * sons respiratórios REAIS (respiração ofegante, tosse) vêm do ESC-50
    (Piczak, 2015 — CC BY-NC 3.0), cujas classes seguem a ontologia AudioSet.

Casos:
  caso_01_controle     — paciente estável (linha de base)
  caso_02_fadiga       — fala lenta, baixa energia, pausas longas, monótona
  caso_03_dispneia     — frases curtas entrecortadas por respiração e tosse
  caso_04_disartria    — fala arrastada/irregular e pouco nítida (sinal de AVC)

Uso: python -m scripts.generate_audio_cases
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from src.config import DATA_RAW

SR = 16000
OUT = DATA_RAW / "audio" / "consultas"
ESC_DIR = DATA_RAW / "audio" / "esc50"
ESC_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"

CASES = {
    "caso_01_controle": {
        "voice": "pt-BR-AntonioNeural", "rate": "+0%", "pitch": "+0Hz", "volume": "+0%",
        "sentences": [
            "Bom dia, doutor.",
            "Hoje eu estou me sentindo bem melhor.",
            "Dormi a noite toda e consegui caminhar no corredor com a fisioterapeuta.",
            "Estou me alimentando normalmente e a dor no joelho diminuiu bastante.",
            "Acho que já estou pronto para ir para casa.",
        ],
    },
    "caso_02_fadiga": {
        "voice": "pt-BR-AntonioNeural", "rate": "-35%", "pitch": "-12Hz", "volume": "-40%",
        "pause_s": 1.6, "monotone": True,
        "sentences": [
            "Doutor, eu estou muito cansado.",
            "Desde ontem não tenho forças nem para levantar da cama.",
            "Sinto uma fraqueza no corpo todo, um esgotamento.",
            "Durmo o dia inteiro, mas acordo exausto.",
            "Estou sem vontade de comer.",
        ],
    },
    "caso_03_dispneia": {
        "voice": "pt-BR-FranciscaNeural", "rate": "+8%", "pitch": "+0Hz", "volume": "+0%",
        "insert": ["breathing", "breathing", "coughing", "breathing", "coughing"],
        "sentences": [
            "Doutora, eu estou com muita falta de ar.",
            "Não consigo terminar as frases.",
            "Sinto um aperto e uma dor no peito quando respiro.",
            "Piorou muito desde a madrugada.",
            "Estou com medo, não consigo respirar direito.",
        ],
    },
    "caso_04_disartria": {
        "voice": "pt-BR-FranciscaNeural", "rate": "-20%", "pitch": "-5Hz", "volume": "-10%",
        "slur": True, "pause_s": 0.9,
        "sentences": [
            "Eu acordei com a fala enrolada.",
            "Meu braço direito está fraco e formigando.",
            "A minha boca está meio torta e eu estou com tontura.",
            "Começou faz umas duas horas.",
        ],
    },
}


async def _tts(text: str, voice: str, rate: str, pitch: str, volume: str) -> np.ndarray:
    import edge_tts

    com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    buf = bytearray()
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    y, sr = librosa.load(io.BytesIO(bytes(buf)), sr=SR, mono=True)
    return y


def _esc50_clips(category: str, n: int = 3) -> list[np.ndarray]:
    ESC_DIR.mkdir(parents=True, exist_ok=True)
    meta = (DATA_RAW / "audio" / "esc50.csv").read_text(encoding="utf-8").splitlines()[1:]
    files = [l.split(",")[0] for l in meta if l.split(",")[3] == category][:n]
    clips = []
    for f in files:
        p = ESC_DIR / f
        if not p.exists():
            r = requests.get(ESC_BASE + f, timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        y, _ = librosa.load(str(p), sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=25)
        clips.append(yt[: int(2.2 * SR)] if len(yt) > 0 else y[: int(2 * SR)])
    return clips


def _monotone(y: np.ndarray) -> np.ndarray:
    """Reduz a prosódia: achata o contorno de F0 em direção à mediana (PSOLA via Praat)
    e aplica compressão dinâmica leve (voz "sem vida")."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 60, 300)
    tier = call(manip, "Extract pitch tier")
    med = call(snd.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
    call(tier, "Multiply frequencies", snd.xmin, snd.xmax, 1.0)
    n = call(tier, "Get number of points")
    for i in range(1, n + 1):
        t = call(tier, "Get time from index", i)
        f = call(tier, "Get value at index", i)
        call(tier, "Remove point", i)
        call(tier, "Add point", t, med + 0.3 * (f - med))
    call([tier, manip], "Replace pitch tier")
    y = call(manip, "Get resynthesis (overlap-add)").values[0]
    return np.tanh(2.0 * y) / 2.0


def _slur(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Disartria simulada: ritmo irregular (time-stretch variável por segmento),
    perda de nitidez articulatória (passa-baixas) e leve tremor de amplitude."""
    seg = int(0.18 * SR)
    parts = []
    for i in range(0, len(y), seg):
        s = y[i:i + seg]
        if len(s) < 512:
            parts.append(s)
            continue
        parts.append(librosa.effects.time_stretch(s, rate=float(rng.choice([0.55, 0.7, 1.0, 1.3]))))
        if rng.random() < 0.25:  # hesitações / micro-pausas irregulares
            parts.append(np.zeros(int(rng.uniform(0.12, 0.3) * SR)))
    y = np.concatenate(parts)
    sos = butter(4, 1800, "lowpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, y)
    t = np.arange(len(y)) / SR
    return y * (1 + 0.25 * np.sin(2 * np.pi * 5.5 * t))


def build_case(name: str, spec: dict, rng: np.random.Generator) -> Path:
    pieces = []
    inserts = {c: _esc50_clips(c) for c in set(spec.get("insert", []))}
    for i, sent in enumerate(spec["sentences"]):
        y = asyncio.run(_tts(sent, spec["voice"], spec["rate"], spec["pitch"], spec["volume"]))
        y, _ = librosa.effects.trim(y, top_db=35)
        if spec.get("monotone"):
            y = _monotone(y)
        if spec.get("slur"):
            y = _slur(y, rng)
        pieces.append(y)
        if "insert" in spec and i < len(spec["insert"]):
            clip = inserts[spec["insert"][i]][i % len(inserts[spec["insert"][i]])]
            clip = clip / (np.abs(clip).max() + 1e-9) * 0.5 * np.abs(y).max()
            pieces += [np.zeros(int(0.15 * SR)), clip, np.zeros(int(0.15 * SR))]
        else:
            pieces.append(np.zeros(int(spec.get("pause_s", 0.5) * SR)))
    y = np.concatenate(pieces)
    y = y + rng.normal(0, 0.002, len(y))  # ruído de sala
    y = y / (np.abs(y).max() + 1e-9) * (0.45 if "fadiga" in name else 0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.wav"
    sf.write(str(path), y.astype(np.float32), SR, subtype="PCM_16")
    return path


def main() -> None:
    rng = np.random.default_rng(42)
    manifest = {}
    for name, spec in CASES.items():
        p = build_case(name, spec, rng)
        dur = sf.info(str(p)).duration
        manifest[name] = {"arquivo": p.name, "duracao_s": round(dur, 1), "texto_roteiro": " ".join(spec["sentences"]),
                          "simulacao": {k: v for k, v in spec.items() if k != "sentences"}}
        print(f"{name}: {dur:.1f}s")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
