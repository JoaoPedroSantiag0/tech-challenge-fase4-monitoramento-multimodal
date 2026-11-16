"""Transcrição com Azure AI Speech (Speech to Text), pt-BR.

Usa reconhecimento contínuo com timestamps por palavra, o que permite medir
a velocidade de fala (palavras/min) e as pausas entre palavras a partir da
própria transcrição — complementando os biomarcadores acústicos.

O áudio é enviado por PushAudioInputStream (evita problemas de caminhos com
acentos no Windows).
"""
from __future__ import annotations

import json
import threading
import time
import wave
from pathlib import Path

from src.config import AZURE


def transcribe_azure(path: str | Path, language: str | None = None) -> dict:
    import azure.cognitiveservices.speech as speechsdk

    if not AZURE.speech_enabled:
        raise RuntimeError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION não configurados")
    cfg = speechsdk.SpeechConfig(subscription=AZURE.speech_key, region=AZURE.speech_region)
    cfg.speech_recognition_language = language or AZURE.speech_language
    cfg.request_word_level_timestamps()
    cfg.output_format = speechsdk.OutputFormat.Detailed

    with wave.open(str(path), "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=sr, bits_per_sample=8 * sw, channels=ch)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    audio_cfg = speechsdk.audio.AudioConfig(stream=stream)
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

    segments, words, errors = [], [], []
    done = threading.Event()

    def on_recognized(evt):
        r = evt.result
        if r.reason == speechsdk.ResultReason.RecognizedSpeech and r.text:
            detail = json.loads(r.json)
            best = (detail.get("NBest") or [{}])[0]
            segments.append({"text": r.text, "offset_s": r.offset / 1e7, "duration_s": r.duration / 1e7,
                             "confidence": best.get("Confidence")})
            for w in best.get("Words", []):
                words.append({"word": w["Word"], "start_s": w["Offset"] / 1e7,
                              "end_s": (w["Offset"] + w["Duration"]) / 1e7})

    def on_canceled(evt):
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            errors.append(evt.cancellation_details.error_details)
        done.set()

    rec.recognized.connect(on_recognized)
    rec.session_stopped.connect(lambda _: done.set())
    rec.canceled.connect(on_canceled)
    t0 = time.time()
    rec.start_continuous_recognition()
    chunk = sr * sw * ch // 10
    for i in range(0, len(pcm), chunk):
        stream.write(pcm[i:i + chunk])
    stream.close()
    done.wait(timeout=120)
    rec.stop_continuous_recognition()
    if errors and not segments:
        raise RuntimeError(f"Azure Speech: {errors[0]}")
    return {"provider": "Azure Speech to Text", "language": cfg.speech_recognition_language,
            "text": " ".join(s["text"] for s in segments), "segments": segments, "words": words,
            "latency_s": round(time.time() - t0, 2), "errors": errors}


def transcribe_local(path: str | Path) -> dict:
    """Fallback offline (Whisper small) — usado apenas sem credenciais Azure."""
    import librosa
    import torch
    from transformers import pipeline

    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                   device=0 if torch.cuda.is_available() else -1)
    y, _ = librosa.load(str(path), sr=16000)
    out = asr(y, return_timestamps="word", generate_kwargs={"language": "portuguese", "task": "transcribe"})
    words = [{"word": c["text"].strip(), "start_s": c["timestamp"][0], "end_s": c["timestamp"][1]}
             for c in out.get("chunks", []) if c["timestamp"][0] is not None]
    return {"provider": "Whisper local (fallback)", "language": "pt", "text": out["text"].strip(),
            "segments": [{"text": out["text"].strip(), "offset_s": 0.0}], "words": words,
            "latency_s": round(time.time() - t0, 2), "errors": []}


def transcribe(path: str | Path) -> dict:
    if AZURE.speech_enabled:
        return transcribe_azure(path)
    return transcribe_local(path)


def speech_rate_from_words(words: list[dict]) -> dict:
    if len(words) < 3:
        return {}
    dur = words[-1]["end_s"] - words[0]["start_s"]
    gaps = [b["start_s"] - a["end_s"] for a, b in zip(words, words[1:]) if b["start_s"] is not None and a["end_s"] is not None]
    long_gaps = [g for g in gaps if g >= 0.5]
    return {"palavras_por_min": round(60 * len(words) / dur, 1) if dur > 0 else None,
            "pausas_entre_palavras_>=0.5s": len(long_gaps),
            "pausa_max_s": round(max(gaps), 2) if gaps else 0.0}
