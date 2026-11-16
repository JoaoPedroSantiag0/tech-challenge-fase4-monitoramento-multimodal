"""Biomarcadores vocais extraídos do sinal de áudio.

- Praat (parselmouth): F0 média e variabilidade (semitons), jitter, shimmer, HNR;
- librosa: energia, pausas, segmentos de fonação, núcleos silábicos
  (taxa de fala / articulação, adaptação de De Jong & Wempe, 2009),
  irregularidade rítmica e nitidez espectral.

Valores de referência (fala adulta conversacional / limiares do Praat):
  taxa de articulação 4–6 sílabas/s; proporção de pausas 0,15–0,30;
  variabilidade de F0 > 2 st; jitter < 1,04 %; shimmer < 3,81 %; HNR > 15 dB.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

SR = 16000


def load(path: str | Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def praat_features(y: np.ndarray) -> dict[str, float]:
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    out = {"f0_mean_hz": float(np.mean(f0)) if len(f0) else np.nan,
           "f0_std_st": float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else np.nan}
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 400)
        out["jitter_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer_pct"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
        out["hnr_db"] = call(harm, "Get mean", 0, 0)
    except Exception:  # sinal sem vozeamento suficiente
        out.update({"jitter_pct": np.nan, "shimmer_pct": np.nan, "hnr_db": np.nan})
    return out


def temporal_features(y: np.ndarray, silence_db: float = -35.0) -> dict[str, float]:
    hop = int(0.01 * SR)
    rms = librosa.feature.rms(y=y, frame_length=int(0.03 * SR), hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    db_rel = db - db.max()
    speech = db_rel > silence_db
    total = len(speech) * 0.01

    # segmentos de fala e pausas (pausa = silêncio >= 250 ms)
    runs, cur, val = [], 0, speech[0]
    for s in speech:
        if s == val:
            cur += 1
        else:
            runs.append((val, cur))
            val, cur = s, 1
    runs.append((val, cur))
    pauses = [n * 0.01 for v, n in runs if not v and n * 0.01 >= 0.25]
    phon = [n * 0.01 for v, n in runs if v and n * 0.01 >= 0.1]
    # enunciados: trechos de fala delimitados por pausas >= 250 ms
    utt, cur_len = [], 0.0
    for v, n in runs:
        d = n * 0.01
        if v or d < 0.25:
            cur_len += d
        else:
            if cur_len >= 0.3:
                utt.append(cur_len)
            cur_len = 0.0
    if cur_len >= 0.3:
        utt.append(cur_len)
    speaking_time = speech.sum() * 0.01

    # núcleos silábicos: picos de intensidade proeminentes dentro da fala
    env = np.convolve(db_rel, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(env, prominence=2.0, distance=10)
    peaks = peaks[speech[peaks]]
    n_syl = len(peaks)
    ipi = np.diff(peaks) * 0.01
    ipi = ipi[ipi < 0.6]  # intervalos dentro de um mesmo trecho de fala

    # nitidez articulatória: energia acima de 2 kHz nos quadros de fala
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    m = min(S.shape[1], len(speech))
    Ssp = S[:, :m][:, speech[:m]]
    hf_ratio = float(Ssp[freqs >= 2000].sum() / (Ssp.sum() + 1e-12)) if Ssp.size else np.nan
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0][:m][speech[:m]]

    return {
        "duracao_s": round(total, 2),
        "energia_dbfs": float(np.median(20 * np.log10(rms[speech] + 1e-9))) if speech.any() else np.nan,
        "proporcao_pausas": float(1 - speaking_time / total) if total else np.nan,
        "n_pausas": len(pauses),
        "pausa_media_s": float(np.mean(pauses)) if pauses else 0.0,
        "fonacao_media_s": float(np.mean(phon)) if phon else np.nan,
        "enunciado_medio_s": float(np.mean(utt)) if utt else np.nan,
        "taxa_fala_silabas_s": n_syl / total if total else np.nan,
        "taxa_articulacao_silabas_s": n_syl / speaking_time if speaking_time else np.nan,
        "irregularidade_ritmo_cv": float(np.std(ipi) / np.mean(ipi)) if len(ipi) > 5 else np.nan,
        "energia_alta_freq_ratio": hf_ratio,
        "centroide_espectral_hz": float(np.mean(centroid)) if len(centroid) else np.nan,
    }


def extract(path_or_y) -> dict[str, float]:
    y = load(path_or_y) if isinstance(path_or_y, (str, Path)) else path_or_y
    feats = temporal_features(y)
    feats.update(praat_features(y))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v)
            for k, v in feats.items()}


# Perfil de referência (fala adulta conversacional típica) — usado quando o
# paciente ainda não tem gravação de linha de base própria.
POPULATION_BASELINE = {
    "taxa_fala_silabas_s": 4.0, "taxa_articulacao_silabas_s": 5.2, "proporcao_pausas": 0.22,
    "energia_dbfs": -18.0, "f0_std_st": 3.5, "energia_alta_freq_ratio": 0.025,
    "irregularidade_ritmo_cv": 0.37, "enunciado_medio_s": 2.5, "shimmer_pct": 9.0,
}


def vocal_indicators(f: dict[str, float], audioset: dict[str, float] | None = None,
                     baseline: dict[str, float] | None = None) -> dict[str, dict]:
    """Converte biomarcadores em indicadores clínicos (0–1) com justificativa.

    As alterações são medidas em relação à linha de base (`baseline`): gravação
    prévia do próprio paciente em estado estável ou, na falta dela, o perfil
    populacional de referência. Isso neutraliza diferenças individuais de voz.
    """
    audioset = audioset or {}
    b = {**POPULATION_BASELINE, **(baseline or {})}

    def clip(x):
        return float(np.clip(x, 0, 1)) if not np.isnan(x) else 0.0

    def g(k):
        v = f.get(k)
        return np.nan if v is None else float(v)

    # ---------------- fadiga: fala mais lenta, mais pausas, menos energia, monotonia
    ev_fad = []
    rate_drop = 1 - g("taxa_fala_silabas_s") / b["taxa_fala_silabas_s"]
    pause_inc = g("proporcao_pausas") - b["proporcao_pausas"]
    energy_drop = b["energia_dbfs"] - g("energia_dbfs")
    mono = 1 - g("f0_std_st") / b["f0_std_st"]
    s_rate, s_pause = clip((rate_drop - 0.12) / 0.3), clip((pause_inc - 0.05) / 0.15)
    s_energy, s_mono = clip((energy_drop - 2) / 6), clip((mono - 0.15) / 0.4)
    if s_rate > 0.2: ev_fad.append(f"fala {100 * rate_drop:.0f}% mais lenta que a linha de base ({g('taxa_fala_silabas_s'):.1f} síl/s)")
    if s_pause > 0.2: ev_fad.append(f"pausas ocupam {100 * g('proporcao_pausas'):.0f}% do tempo (base {100 * b['proporcao_pausas']:.0f}%)")
    if s_energy > 0.2: ev_fad.append(f"energia vocal {energy_drop:.1f} dB abaixo da base")
    if s_mono > 0.2: ev_fad.append(f"prosódia monótona (variação de F0 {100 * mono:.0f}% menor)")
    fadiga = 0.3 * s_rate + 0.3 * s_pause + 0.2 * s_energy + 0.2 * s_mono

    # ---------------- dificuldade respiratória: sons respiratórios + enunciados curtos
    ev_resp = []
    resp_keys = ("Breathing", "Wheeze", "Gasp", "Pant", "Cough", "Throat clearing")
    top = max(resp_keys, key=lambda k: audioset.get(k, 0))
    s_sound = clip(audioset.get(top, 0) / 0.4)
    utt_drop = 1 - g("enunciado_medio_s") / b["enunciado_medio_s"]
    s_utt = clip((utt_drop - 0.2) / 0.4)
    if s_sound > 0.2: ev_resp.append(f"som respiratório (AudioSet/AST): {top} p={audioset.get(top, 0):.2f}")
    if audioset.get("Cough", 0) > 0.1 and top != "Cough": ev_resp.append(f"tosse p={audioset['Cough']:.2f}")
    if s_utt > 0.2: ev_resp.append(f"frases entrecortadas (enunciado médio {g('enunciado_medio_s'):.1f}s)")
    respiratoria = 0.7 * s_sound + 0.3 * s_utt

    # ---------------- disartria: perda de nitidez, ritmo irregular, articulação lenta
    ev_dis = []
    hf_loss = 1 - g("energia_alta_freq_ratio") / b["energia_alta_freq_ratio"]
    irreg = g("irregularidade_ritmo_cv") - b["irregularidade_ritmo_cv"]
    art_drop = 1 - g("taxa_articulacao_silabas_s") / b["taxa_articulacao_silabas_s"]
    s_blur, s_irreg, s_art = clip((hf_loss - 0.5) / 0.4), clip((irreg - 0.05) / 0.2), clip((art_drop - 0.08) / 0.25)
    if s_blur > 0.2: ev_dis.append(f"articulação pouco nítida: energia de consoantes (>2 kHz) {100 * hf_loss:.0f}% menor")
    if s_irreg > 0.2: ev_dis.append(f"ritmo silábico irregular (CV {g('irregularidade_ritmo_cv'):.2f} vs {b['irregularidade_ritmo_cv']:.2f})")
    if s_art > 0.2: ev_dis.append(f"articulação {100 * art_drop:.0f}% mais lenta")
    disartria = 0.5 * s_blur + 0.25 * s_irreg + 0.25 * s_art

    return {
        "fadiga": {"score": round(fadiga, 3), "evidencias": ev_fad},
        "dificuldade_respiratoria": {"score": round(respiratoria, 3), "evidencias": ev_resp},
        "disartria": {"score": round(disartria, 3), "evidencias": ev_dis},
    }
