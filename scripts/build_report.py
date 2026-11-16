"""Gera outputs/relatorio_apresentacao.html a partir dos resultados do pipeline.

Uso: python -m scripts.build_report
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import OUTPUTS, ROOT  # noqa: E402

S = json.loads((OUTPUTS / "summary.json").read_text(encoding="utf-8"))
SCN = json.loads((ROOT / "config" / "scenario.json").read_text(encoding="utf-8"))
SEV = {"CRITICO": "#d62828", "ALTO": "#f77f00", "ATENCAO": "#e0b000", "INFO": "#8d99ae"}
SEV_PT = {"CRITICO": "Crítico", "ALTO": "Alto", "ATENCAO": "Atenção", "INFO": "Info", "NORMAL": "Normal", "ANOMALIA": "Anomalia"}
MOD_PT = {"vitais": "Sinais vitais", "video": "Vídeo", "audio": "Áudio", "prescricao": "Prescrição",
          "movimentacao": "Movimentação", "fusao": "Fusão"}
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c7ced6",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6})


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(s) -> str:
    return html.escape(str(s))


def rel(p: Path | str) -> str:
    """Caminho relativo a outputs/ (onde o HTML fica)."""
    p = (ROOT / p) if not Path(p).is_absolute() else Path(p)
    return Path("..", p.relative_to(ROOT)).as_posix() if not p.is_relative_to(OUTPUTS) else p.relative_to(OUTPUTS).as_posix()


# ------------------------------------------------------------------ gráficos
def chart_confusion() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, title in zip(axes, ("treino", "teste"), ("Treino (calibração)", "Teste (dados não vistos)")):
        m = S["video"][key]
        mat = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
        ax.imshow([[0, 1], [1, 0]], cmap="Greys", alpha=0.0)
        for i in range(2):
            for j in range(2):
                good = i == j
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#2a9d8f" if good else "#e76f51", alpha=0.18 if good else 0.14))
                ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
        ax.set_xticks([0, 1], ["previsto\nnormal", "previsto\nqueda"])
        ax.set_yticks([0, 1], ["real\nnormal", "real\nqueda"])
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(1.5, -.5)
        ax.grid(False)
        ax.set_title(f"{title}\nsens. {m['sensibilidade']:.0%} · espec. {m['especificidade']:.0%}", fontsize=10, color=INK)
    fig.tight_layout()
    return fig_b64(fig)


def chart_video_series(stem: str, title: str) -> str | None:
    f = OUTPUTS / "video" / f"{stem}_serie.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    rep = S["video_relatorios"].get(stem, {})
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(d["t"], d["head_y_s"], color="#264653", lw=2, label="altura da cabeça (↓ = mais perto do chão)")
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["torso_angle_s"], color="#e76f51", lw=1.6, label="inclinação do tronco (°)")
    ax2.set_ylim(0, 180)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.invert_yaxis()
    ax.axhline(S["video"]["limiares_calibrados"]["floor_zone_y"], color="#d62828", ls=":", lw=1)
    for e in rep.get("eventos", []):
        ax.axvspan(e["t_start"], e["t_end"], color=SEV.get(e["severity"], "#999"), alpha=0.15)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("cabeça (fração da altura)")
    ax2.set_ylabel("tronco (°)")
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig_b64(fig)


def chart_audio() -> str:
    cases = list(S["audio"])
    inds = ["fadiga", "dificuldade_respiratoria", "disartria"]
    cols = {"fadiga": "#457b9d", "dificuldade_respiratoria": "#2a9d8f", "disartria": "#e76f51"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.26
    for k, ind in enumerate(inds):
        vals = [S["audio"][c]["indicadores_vocais"][ind]["score"] for c in cases]
        xs = [i + (k - 1) * w for i in range(len(cases))]
        ax.bar(xs, vals, width=w, color=cols[ind], label=ind.replace("_", " "))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(cases)), [c.split("_", 2)[-1] for c in cases])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("escore (0–1)")
    ax.axhline(0.45, color="#999", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    return fig_b64(fig)


def chart_vitals_eval() -> str:
    ev = pd.DataFrame(S["vitais"]["avaliacao"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = range(len(ev))
    ax.barh([i + 0.2 for i in y], ev["recall_evento"], height=0.38, color="#2a9d8f", label="eventos detectados (recall)")
    ax.barh([i - 0.2 for i in y], ev["taxa_alarme_base"], height=0.38, color="#e76f51", label="carga de alarmes (medições sem evento)")
    for i, (r, a) in enumerate(zip(ev["recall_evento"], ev["taxa_alarme_base"])):
        ax.text(r + 0.01, i + 0.2, f"{r:.0%}", va="center", fontsize=8)
        ax.text(a + 0.01, i - 0.2, f"{a:.0%}", va="center", fontsize=8)
    ax.set_yticks(list(y), ev["detector"])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return fig_b64(fig)


def chart_bed(pid: str) -> str | None:
    key = pid.replace(" ", "_")
    fv, fr = OUTPUTS / "cenario" / f"{key}_vitais.parquet", OUTPUTS / "cenario" / f"{key}_risco.parquet"
    if not (fv.exists() and fr.exists()):
        return None
    vt, rk = pd.read_parquet(fv), pd.read_parquet(fr)
    ev = pd.read_parquet(OUTPUTS / "cenario" / "eventos.parquet")
    ev = ev[ev["patient_id"] == pid]
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})
    for ax, (s, lab, c) in zip(axes, (("hr", "FC (bpm)", "#e76f51"), ("spo2", "SpO₂ (%)", "#457b9d"), ("sbp", "PAS (mmHg)", "#6a4c93"))):
        ax.plot(vt["charttime"], vt[s], color=c, lw=1.6, marker="o", ms=2.5)
        ax.set_ylabel(lab, fontsize=9)
        m = ev[(ev["modality"] == "vitais") & ev["severity"].isin(["CRITICO", "ALTO"])]
        yy = vt.set_index("charttime")[s].reindex(m["time"])
        ax.scatter(m["time"], yy, color=[SEV[x] for x in m["severity"]], marker="x", s=36, zorder=3)
    ax = axes[-1]
    ax.fill_between(rk["time"], rk["risk"], color="#d62828", alpha=0.15)
    ax.plot(rk["time"], rk["risk"], color="#d62828", lw=2.2, label="risco fundido")
    for c, col in (("risk_vitais", "#e76f51"), ("risk_audio", "#2a9d8f"), ("risk_video", "#264653"),
                   ("risk_prescricao", "#6a4c93"), ("risk_movimentacao", "#a8a29e")):
        if c in rk:
            ax.plot(rk["time"], rk[c], lw=1, color=col, label=c[5:])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("risco", fontsize=9)
    for _, e in ev[ev["modality"].isin(["audio", "video", "fusao"])].iterrows():
        for a in axes:
            a.axvline(e["time"], color=SEV.get(e["severity"], "#999"), lw=1, ls="--", alpha=0.7)
    ax.legend(frameon=False, fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return fig_b64(fig)


# ------------------------------------------------------------------ helpers HTML
def sev_badge(s: str) -> str:
    return f"<span class='badge' style='--c:{SEV.get(s, '#2a9d8f')}'>{SEV_PT.get(s, s)}</span>"


def table(rows: list[list], head: list[str], cls: str = "") -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def highlight(text: str, terms: list[str]) -> str:
    out = esc(text)
    for t in sorted(set(terms), key=len, reverse=True):
        out = out.replace(esc(t), f"<mark>{esc(t)}</mark>")
    return out


# ------------------------------------------------------------------ seções
def section_flow() -> str:
    return """
<svg viewBox="0 0 980 360" class="diagram" role="img" aria-label="Fluxo multimodal">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
  <g class="lane">
    <rect x="10" y="20" width="170" height="70" rx="10"/><text x="95" y="50">🎥 Vídeo do quarto</text><text x="95" y="70" class="s">UR Fall Detection</text>
    <rect x="10" y="110" width="170" height="70" rx="10"/><text x="95" y="140">🎙️ Áudio da consulta</text><text x="95" y="160" class="s">pt-BR · ESC-50/AudioSet</text>
    <rect x="10" y="200" width="170" height="70" rx="10"/><text x="95" y="230">💓 Sinais vitais</text><text x="95" y="250" class="s">MIMIC-IV (PhysioNet)</text>
    <rect x="10" y="290" width="170" height="60" rx="10"/><text x="95" y="316">💊 Prescrições · 🛏️ Mobilidade</text><text x="95" y="334" class="s">MIMIC-IV (PhysioNet)</text>
  </g>
  <g class="proc">
    <rect x="240" y="20" width="230" height="70" rx="10"/><text x="355" y="48">OpenPose (18 keypoints)</text><text x="355" y="68" class="s">+ YOLOv8 objetos · zona crítica</text>
    <rect x="240" y="110" width="230" height="70" rx="10" class="az"/><text x="355" y="138">Azure Speech to Text</text><text x="355" y="158" class="s">→ Azure Text Analytics · AST · Praat</text>
    <rect x="240" y="200" width="230" height="70" rx="10"/><text x="355" y="228">NEWS2 · z-score robusto</text><text x="355" y="248" class="s">Isolation Forest (streaming)</text>
    <rect x="240" y="290" width="230" height="60" rx="10"/><text x="355" y="316">Regras ISMP · protocolo 2h</text><text x="355" y="334" class="s">Isolation Forest de ordens</text>
  </g>
  <g class="fuse"><rect x="540" y="110" width="190" height="160" rx="14"/><text x="635" y="160">Fusão tardia</text><text x="635" y="184" class="s">noisy-OR + decaimento</text><text x="635" y="204" class="s">corroboração ≥2 modalidades</text><text x="635" y="236" class="s">risco do leito 0–100%</text></g>
  <g class="out"><rect x="790" y="60" width="180" height="80" rx="10"/><text x="880" y="94">Alert Bus</text><text x="880" y="114" class="s">severidade → escalonamento</text>
  <rect x="790" y="170" width="180" height="70" rx="10"/><text x="880" y="200">Central de Alertas</text><text x="880" y="220" class="s">dashboard da equipe</text>
  <rect x="790" y="265" width="180" height="70" rx="10"/><text x="880" y="295">Webhook Teams</text><text x="880" y="315" class="s">plantonista · resposta rápida</text></g>
  <g class="links" marker-end="url(#ar)">
    <path d="M180 55 H240"/><path d="M180 145 H240"/><path d="M180 235 H240"/><path d="M180 320 H240"/>
    <path d="M470 55 C510 55 505 150 540 160"/><path d="M470 145 H540"/><path d="M470 235 H540"/><path d="M470 320 C510 320 505 250 540 245"/>
    <path d="M730 170 C760 170 760 100 790 100"/><path d="M730 200 H790"/><path d="M730 230 C760 230 760 300 790 300"/>
  </g>
</svg>"""


def section_video() -> str:
    m = S["video"]
    th = m["limiares_calibrados"]
    parts = []
    for p in SCN["patients"]:
        if "video" not in p:
            continue
        stem = Path(p["video"]["file"]).stem
        rep = S["video_relatorios"].get(stem)
        if not rep:
            continue
        ev = "".join(f"<li>{sev_badge(e['severity'])} <b>{esc(e['kind'].replace('_', ' '))}</b> "
                     f"({e['t_start']:.1f}–{e['t_end']:.1f}s) — {esc(e['description'])}</li>" for e in rep["eventos"]) \
            or "<li>Nenhum desvio — movimento dentro do padrão esperado.</li>"
        series = chart_video_series(stem, f"{stem}: séries extraídas pelo OpenPose")
        parts.append(f"""
<div class="card media">
  <div class="mh"><h4>{esc(p['patient_id'])} · {esc(p['video']['contexto'])}</h4>{sev_badge(rep['status'] if rep['status'] != 'ANOMALIA' else 'CRITICO')}</div>
  <video controls muted preload="metadata" src="{rel(OUTPUTS / 'video' / (stem + '_anotado.mp4'))}"></video>
  <ul class="ev">{ev}</ul>
  {f'<img src="{series}" alt="séries temporais do vídeo {esc(stem)}">' if series else ''}
  <p class="foot">Pose detectada em {rep['cobertura_pose_pct']}% dos {rep['frames_analisados']} frames · relatório automático: <code>outputs/video/{stem}_relatorio.md</code></p>
</div>""")
    return f"""
<div class="grid2">
  <div class="card">
    <h4>Como funciona</h4>
    <ol class="steps">
      <li><b>OpenPose</b> (rede original CMU, 18 keypoints, PAFs) estima o esqueleto do paciente a cada frame, ~25 fps na GPU.</li>
      <li><b>YOLOv8s</b> detecta pessoa, cama, sofá, cadeira e objetos; define a <b>zona crítica "piso"</b> e distingue deitar na cama (esperado) de deitar no chão.</li>
      <li>Features biomecânicas: inclinação do tronco, altura e velocidade de descida da cabeça, ângulos de joelho, quadril e cotovelo, energia de movimento.</li>
      <li>Regras temporais: <b>queda</b> (descida rápida + postura horizontal no piso), <b>imobilidade pós-queda</b> e <b>assimetria</b> D/E em exercício.</li>
      <li>Saídas: vídeo anotado, relatório automático (JSON/Markdown) com desvios e amplitude articular, e alerta.</li>
    </ol>
  </div>
  <div class="card">
    <h4>Avaliação no UR Fall Detection Dataset (70 vídeos reais)</h4>
    <img src="{chart_confusion()}" alt="matrizes de confusão">
    <p class="foot">Limiares escolhidos por busca em grade só no treino (35 vídeos) e aplicados uma única vez no teste (35 vídeos).
    Calibrados: descida ≥ {th['fall_drop_rate']} alt./s em {th['fall_window_s']}s · ≥ {th['min_ground_s']}s no piso ·
    tronco &gt; {th['lying_torso_deg']:.0f}° · zona piso y &gt; {th['floor_zone_y']}.</p>
  </div>
</div>
<div class="grid2">{''.join(parts)}</div>"""


def section_audio() -> str:
    rows, cards = [], []
    for stem, a in S["audio"].items():
        terms = [t["termo"] for t in a["nlp"].get("termos_criticos", [])]
        ind = a["indicadores_vocais"]
        rows.append([f"<b>{esc(stem.split('_', 2)[-1])}</b>", sev_badge(a["status"]),
                     f"{ind['fadiga']['score']:.2f}", f"{ind['dificuldade_respiratoria']['score']:.2f}",
                     f"{ind['disartria']['score']:.2f}", esc(a["nlp"].get("sentimento", {}).get("geral", "–")),
                     esc(", ".join(f"{k} {v:.2f}" for k, v in a["audioset_max"].items() if v > 0.1 and k != "Speech") or "–")])
        achados = "".join(f"<li>{sev_badge(f['severidade'])} <b>{esc(f['condicao'].replace('_', ' '))}</b>: {esc(f['descricao'])}"
                          f"<br><span class='muted'>{esc('; '.join(f['evidencias']))}</span></li>" for f in a["achados"]) \
            or "<li>Sem alterações — linha de base.</li>"
        ent = a.get("sintomas_azure_health") or []
        kp = a["nlp"].get("frases_chave") or []
        bio = a["biomarcadores"]
        cards.append(f"""
<div class="card media">
  <div class="mh"><h4>{esc(stem.replace('_', ' '))}</h4>{sev_badge(a['status'])}</div>
  <audio controls preload="none" src="{rel(ROOT / 'data/raw/audio/consultas' / (stem + '.wav'))}"></audio>
  <p class="quote">“{highlight(a['transcricao']['text'], terms)}”</p>
  <p class="foot">STT: {esc(a['transcricao']['provider'])} ({a['transcricao'].get('latency_s', '–')}s) · NLP: {esc(a['nlp'].get('provider', '–'))}
  {('· entidades de saúde: ' + esc(', '.join(ent))) if ent else ''}{(' · frases-chave: ' + esc(', '.join(kp[:6]))) if kp else ''}</p>
  <ul class="ev">{achados}</ul>
  <p class="foot">Biomarcadores: {bio['taxa_fala_silabas_s']:.1f} síl/s · pausas {100 * bio['proporcao_pausas']:.0f}% · F0 dp {bio['f0_std_st']:.1f} st ·
  energia {bio['energia_dbfs']:.0f} dBFS · energia &gt;2 kHz {100 * bio['energia_alta_freq_ratio']:.1f}% · jitter {bio['jitter_pct']:.1f}% · HNR {bio['hnr_db']:.1f} dB</p>
</div>""")
    return f"""
<div class="grid2">
  <div class="card">
    <h4>Pipeline de áudio</h4>
    <ol class="steps">
      <li><b>Azure Speech to Text</b> (pt-BR, reconhecimento contínuo com tempo por palavra): transcrição e ritmo de fala.</li>
      <li><b>Azure Text Analytics</b>: sentimento com mineração de opinião, frases-chave e <b>Text Analytics for Health</b> (sintomas, negação). Um léxico clínico PT-BR mapeia os termos críticos em síndromes: respiratória, cardíaca, AVC e fadiga.</li>
      <li><b>Biomarcadores vocais</b> (Praat/librosa): taxa de fala e articulação, pausas, F0, jitter, shimmer, HNR, ritmo silábico e nitidez de consoantes, comparados à <b>linha de base</b>.</li>
      <li><b>AST treinado no AudioSet</b> (527 classes): detecta respiração ofegante, arquejo, tosse e sibilo durante a fala.</li>
      <li>Regras de fusão: disartria + sinais focais → <b>suspeita de AVC (crítico)</b>; dispneia + dor torácica → crítico.</li>
    </ol>
  </div>
  <div class="card">
    <h4>Indicadores vocais por consulta</h4>
    <img src="{chart_audio()}" alt="indicadores vocais">
    {table(rows, ['caso', 'status', 'fadiga', 'resp.', 'disartria', 'sentimento', 'AudioSet'], 'compact')}
  </div>
</div>
<div class="grid2">{''.join(cards)}</div>"""


def section_anomaly() -> str:
    v, p, mv = S["vitais"], S["prescricoes"], S["movimentacao"]
    pr_rows = [[esc(k.replace('_', ' ')), *(str(sv.get(s, 0)) for s in ("CRITICO", "ALTO", "ATENCAO", "INFO"))]
               for k, sv in p["por_regra_severidade"].items()]
    mv_rows = [[esc(k.replace('_', ' ')), *(str(sv.get(s, 0)) for s in ("CRITICO", "ALTO", "ATENCAO", "INFO"))]
               for k, sv in mv["por_regra_severidade"].items()]
    ex_p = "".join(f"<li><b>{esc(e['rule'].replace('_', ' '))}</b>: {esc(e['detail'])}</li>" for e in p["exemplos"][:6])
    ex_m = "".join(f"<li><b>{esc(e['rule'].replace('_', ' '))}</b>: {esc(e['detail'])}</li>" for e in mv["exemplos"][:4])
    art = ", ".join(f"{k}: {n}" for k, n in v["artefatos_por_sinal"].items())
    ev_df = pd.DataFrame(v["avaliacao"]).set_index("detector")
    ratio = ev_df.loc["NEWS2 (>=5)", "taxa_alarme_base"] / max(ev_df.loc["Ensemble (>=2 votos)", "taxa_alarme_base"], 1e-6)
    return f"""
<div class="grid2">
  <div class="card">
    <h4>Sinais vitais: FC, pressão arterial, SpO₂, FR, temperatura</h4>
    <p>{v['n_medicoes']:,} medições de {v['n_internacoes']} internações de UTI reais. Foram removidos {v['artefatos_removidos']} artefatos de sensor ({esc(art)}), por exemplo FC = 0 e PAM = 801.
    Três detectores rodam em <b>streaming</b>, uma observação por vez:</p>
    <ul class="ev"><li><b>NEWS2</b>: escore clínico de alerta precoce (Royal College of Physicians).</li>
    <li><b>z-score robusto</b> contra a linha de base móvel do <i>próprio</i> paciente (mediana/MAD).</li>
    <li><b>Isolation Forest</b> multivariado: valores, desvios da linha de base, variações e índice de choque.</li></ul>
    <img src="{chart_vitals_eval()}" alt="avaliação dos detectores de sinais vitais">
    <p class="foot">Avaliação com {v['eventos_injetados']} eventos clínicos injetados (taquicardia, dessaturação, hipotensão, taquipneia, febre) em 30% das internações, não usadas no treino.
    Em UTI o NEWS2 sozinho já fica ≥5 em {100 * v['taxa_sinalizacao_dados_reais']['NEWS2>=5']:.0f}% do tempo, o que gera fadiga de alarmes. O ensemble mantém a detecção com carga {ratio:.0f}× menor.</p>
  </div>
  <div class="card">
    <h4>Evolução de prescrições: alterações inesperadas no tratamento</h4>
    <p>{p['n_ordens']:,} ordens de {p['n_internacoes']} internações. Regras de farmacovigilância (lista ISMP de alta vigilância) + Isolation Forest no nível da ordem
    ({p['iforest_anomalias']} ordens atípicas).</p>
    {table(pr_rows, ['regra', 'crít.', 'alto', 'atenção', 'info'], 'compact')}
    <ul class="ev small">{ex_p}</ul>
    <h4 style="margin-top:18px">Movimentação durante a internação</h4>
    <p>{mv['n_registros']:,} registros de enfermagem (mudança de decúbito, posição, nível de atividade) + eventos de vídeo.</p>
    {table(mv_rows, ['regra', 'crít.', 'alto', 'atenção', 'info'], 'compact')}
    <ul class="ev small">{ex_m}</ul>
  </div>
</div>"""


def section_azure() -> str:
    az = S["azure"]
    a = next(iter(S["audio"].values()))
    lat_stt = [x["transcricao"].get("latency_s") for x in S["audio"].values()]
    lat_nlp = [x["nlp"].get("latency_s") for x in S["audio"].values()]
    ok = az["speech"] and az["language"]
    status = ("<p class='ok'>✅ Serviços Azure ativos nesta execução"
              + (f" (região <b>{esc(az['regiao_speech'])}</b>)" if az.get("regiao_speech") else "") + ".</p>") if ok else \
        "<p class='warn'>⚠️ Esta execução usou o fallback local (Whisper + léxico) porque as chaves Azure não estavam configuradas. Preencha o <code>.env</code> e rode o pipeline novamente.</p>"
    sample = json.dumps({k: a["nlp"].get(k) for k in ("sentimento", "frases_chave") if a["nlp"].get(k)}, ensure_ascii=False, indent=1)[:1100]
    return f"""
<div class="grid2">
  <div class="card">
    <h4>Serviços gerenciados utilizados</h4>
    {status}
    {table([
        ['Azure AI Speech: Speech to Text', 'Transcrição pt-BR contínua com timestamps por palavra (SDK, PushAudioInputStream)', f"{sum(x or 0 for x in lat_stt) / max(len(lat_stt), 1):.1f}s / consulta"],
        ['Azure AI Language: Sentimento + Opinion Mining', 'Polaridade por frase e alvo das opiniões', '—'],
        ['Azure AI Language: Key Phrases', 'Frases-chave da queixa', '—'],
        ['Azure AI Language: Text Analytics for Health', 'Sintomas, diagnósticos, negação/certeza', f"{sum(x or 0 for x in lat_nlp) / max(len(lat_nlp), 1):.1f}s / consulta"],
    ], ['serviço', 'uso no projeto', 'latência média'], 'compact')}
    <p class="foot">Credenciais via <code>.env</code> (nunca versionado). Arquivos: <code>src/audio/azure_speech.py</code> e <code>src/audio/azure_text.py</code>.
    Sem chaves, o sistema degrada para Whisper e léxico locais e marca o provedor em cada resultado.</p>
  </div>
  <div class="card"><h4>Exemplo de resposta (Text Analytics)</h4><pre>{esc(sample)}</pre></div>
</div>"""


def section_scenario() -> str:
    cards = []
    for p in SCN["patients"]:
        pid = p["patient_id"]
        sm = S["cenario"].get(pid, {})
        chart = chart_bed(pid)
        counts = sm.get("eventos_por_modalidade", {})
        rows = [[esc(MOD_PT.get(mod, mod)), *(str(c.get(s, 0)) for s in ("CRITICO", "ALTO", "ATENCAO", "INFO"))]
                for mod, c in counts.items()]
        fus = "".join(f"<li>{sev_badge(f['severidade'])} {esc(f['hora'])} — {esc(f['titulo'])}</li>" for f in sm.get("alertas_fusao", [])) \
            or "<li>Sem corroboração multimodal.</li>"
        cards.append(f"""
<div class="card">
  <div class="mh"><h4>{esc(pid)} · {esc(p['nome'])}</h4><span class="risk" style="--r:{sm.get('risco_max', 0)}">risco máx. {100 * sm.get('risco_max', 0):.0f}% às {esc(sm.get('hora_risco_max', '–'))}</span></div>
  <p class="foot">MIMIC-IV internação {p['stay_id']} · áudio {esc(sm.get('hora_audio') or '–')} · vídeo {esc(sm.get('hora_video') or '–')}</p>
  {f'<img src="{chart}" alt="linha do tempo {esc(pid)}">' if chart else ''}
  <div class="grid2 tight"><div>{table(rows, ['modalidade', 'crít.', 'alto', 'atenção', 'info'], 'compact')}</div>
  <div><b>Alertas de fusão</b><ul class="ev small">{fus}</ul></div></div>
</div>""")
    dash = OUTPUTS / "dashboard.png"
    dash_html = f'<img class="shot" src="{img_b64(dash)}" alt="Central de Alertas">' if dash.exists() else ""
    return f"""
<div class="card">
  <h4>Do sinal ao plantonista</h4>
  <ol class="steps row">
    <li><b>Detecção</b>: cada módulo emite eventos com severidade e evidências.</li>
    <li><b>Fusão</b>: risco do leito = noisy-OR entre modalidades (maior evento vigente de cada uma, com decaimento). Se ≥ 2 modalidades concordam em 6h, o alerta é escalonado.</li>
    <li><b>Alert Bus</b>: registro auditável (<code>alerts.jsonl</code>), deduplicação e roteamento por severidade.</li>
    <li><b>Equipe</b>: Central de Alertas (Streamlit) + webhook (Teams/Slack). Crítico aciona enfermagem, plantonista e time de resposta rápida; o profissional confirma o recebimento.</li>
  </ol>
  {dash_html}
</div>
<div class="grid2">{''.join(cards)}</div>"""


def section_results() -> str:
    mt = S["video"]["teste"]
    v = pd.DataFrame(S["vitais"]["avaliacao"]).set_index("detector")
    ens = v.loc["Ensemble (>=2 votos)"]
    news = v.loc["NEWS2 (>=5)"]
    expected = {"controle": set(), "fadiga": {"fadiga"},
                "dispneia": {"dificuldade_respiratoria", "insuficiencia_respiratoria_dor_toracica"},
                "disartria": {"suspeita_avc", "disartria"}}
    n_ok = 0
    for stem, a in S["audio"].items():
        exp = expected.get(stem.split("_", 2)[-1])
        order = {"INFO": 0, "ATENCAO": 1, "ALTO": 2, "CRITICO": 3}
        relevant = [f for f in a["achados"] if f["severidade"] != "INFO"]
        top = max(relevant, key=lambda f: order[f["severidade"]])["condicao"] if relevant else None
        n_ok += (top is None) if exp == set() else (top in (exp or set()))
    n_audio = len(S["audio"])
    f3 = OUTPUTS / "cenario" / "Leito_03_vitais.parquet"
    sbp3 = f"{pd.read_parquet(f3)['sbp'].max():.0f} mmHg" if f3.exists() else "elevada"
    return f"""
<div class="kpis">
  <div class="kpi"><div class="n">{mt['sensibilidade']:.0%}</div><div class="l">sensibilidade em quedas<br><span>teste, vídeos não vistos</span></div></div>
  <div class="kpi"><div class="n">{mt['especificidade']:.0%}</div><div class="l">especificidade em quedas<br><span>atividades diárias normais</span></div></div>
  <div class="kpi"><div class="n">{ens['taxa_alarme_base']:.0%}</div><div class="l">carga de alarmes do ensemble<br><span>vs {news['taxa_alarme_base']:.0%} só com NEWS2</span></div></div>
  <div class="kpi"><div class="n">{n_ok}/{n_audio}</div><div class="l">consultas classificadas corretamente<br><span>controle, fadiga, dispneia, disartria</span></div></div>
</div>
<div class="grid2">
  <div class="card"><h4>Exemplos de anomalias detectadas</h4><ul class="ev">
    <li>{sev_badge('CRITICO')} <b>Queda</b> com permanência no piso, detectada por OpenPose + YOLOv8 (Leito 01, à noite). Ocorreu horas depois de o áudio registrar fadiga e os vitais mostrarem NEWS2 crescente.</li>
    <li>{sev_badge('CRITICO')} <b>Suspeita de AVC</b>: disartria na voz + "fala enrolada", "braço fraco", "boca torta" (Leito 03), junto com pico hipertensivo (PAS {sbp3}).</li>
    <li>{sev_badge('CRITICO')} <b>Dispneia com dor torácica</b>: arquejo/tosse detectados pelo AudioSet + "falta de ar" e "dor no peito", 30 min antes da menor SpO₂ do paciente (Leito 02).</li>
    <li>{sev_badge('ALTO')} <b>Escalonamento de opioide/vasopressor</b> ×4 ou mais em prescrições reais do MIMIC-IV.</li>
    <li>{sev_badge('ATENCAO')} <b>Paciente acamado sem mudança de decúbito</b> por mais de 3h (risco de lesão por pressão).</li>
  </ul></div>
  <div class="card"><h4>Limitações e próximos passos</h4><ul class="ev">
    <li>As modalidades vêm de bases públicas distintas. A associação ao mesmo leito é ilustrativa.</li>
    <li>As consultas em áudio são simuladas (TTS + degradações controladas + sons reais do ESC-50), porque não há base pública PT-BR de fala patológica. O próximo passo é validar com gravações clínicas reais e consentidas.</li>
    <li>O UR Fall tem câmera lateral em ambiente controlado. Para outros ângulos é preciso recalibrar a zona crítica, e o YOLOv8 pode ser ajustado com dados do hospital (ex.: instrumentais cirúrgicos).</li>
    <li>Produção: Azure Event Hubs/IoT Hub para os monitores, Azure Functions para o Alert Bus e Azure ML para retreino. Requer conformidade com LGPD e validação clínica.</li>
  </ul></div>
</div>"""


def section_script() -> str:
    rows = [
        ["0:00–1:00", "Capa e desafio", "Problema: monitorar pacientes com dados multimodais e alertar cedo."],
        ["1:00–2:30", "Fluxo multimodal", "Percorrer o diagrama: entradas → modelos → fusão → alerta."],
        ["2:30–5:00", "Vídeo", "Tocar o vídeo anotado da queda (Leito 01); mostrar as matrizes de confusão e o vídeo normal (Leito 04)."],
        ["5:00–8:00", "Áudio + Azure", "Tocar dispneia e disartria; mostrar transcrição do Azure, termos destacados e indicadores."],
        ["8:00–10:30", "Anomalias", "Vitais (NEWS2 × z-score × IF e carga de alarmes), prescrições, movimentação."],
        ["10:30–13:30", "Alerta à equipe (ao vivo)", "Rodar stream_demo + dashboard: alertas chegando, risco subindo, confirmação."],
        ["13:30–15:00", "Resultados e limitações", "KPIs, exemplos, próximos passos."],
    ]
    return table(rows, ["tempo", "bloco", "o que mostrar"], "compact") + """
<pre class="cmd"># terminal 1: Central de Alertas
.venv\\Scripts\\streamlit run dashboard/app.py
# terminal 2: simulação em tempo real (1h clínica a cada 2s)
.venv\\Scripts\\python -m scripts.stream_demo --speed 600   # 24h clínicas em ~2,5 min
# opcional: --live-media processa vídeo e áudio com os modelos no momento do evento</pre>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1f2933;--muted:#6b7785;--line:#e3e7ec;--accent:#0b6e79;--accent2:#e76f51;--mark:#ffe08a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1418;--card:#171e24;--ink:#e6ebef;--muted:#9aa7b3;--line:#2a333b;--accent:#4fb3bf;--mark:#7a5c00}}
:root[data-theme=dark]{--bg:#0f1418;--card:#171e24;--ink:#e6ebef;--muted:#9aa7b3;--line:#2a333b;--accent:#4fb3bf;--mark:#7a5c00}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header.hero{padding:56px 24px 40px;background:linear-gradient(135deg,#0b3c49,#0b6e79 60%,#2a9d8f);color:#fff}
.hero .in{max-width:1180px;margin:0 auto}.hero .tag{opacity:.85;letter-spacing:.08em;text-transform:uppercase;font-size:12px}
.hero h1{font-size:38px;line-height:1.15;margin:8px 0 10px;max-width:900px}.hero p{max-width:820px;opacity:.92;margin:0}
nav{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);overflow-x:auto;white-space:nowrap}
nav .in{max-width:1180px;margin:0 auto;padding:0 16px}nav a{display:inline-block;padding:12px 10px;color:var(--muted);text-decoration:none;font-size:14px}
nav a:hover{color:var(--accent)}
main{max-width:1180px;margin:0 auto;padding:8px 16px 60px}
section{padding-top:36px}section>h2{font-size:24px;margin:0 0 4px}section>h2 span{color:var(--accent);font-weight:600;margin-right:8px}
section>.lead{color:var(--muted);margin:0 0 16px;max-width:860px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:16px;min-width:0}
.card h4{margin:0 0 10px;font-size:16px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid2.tight{gap:12px;margin-top:8px}
@media (max-width:860px){.grid2{grid-template-columns:1fr}.hero h1{font-size:28px}}
img{max-width:100%;height:auto;display:block;border-radius:8px}video,audio{width:100%;border-radius:8px;margin:6px 0 10px;background:#000}audio{background:none}
.mh{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}.mh h4{margin:0}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;color:var(--c);border:1px solid var(--c);background:color-mix(in srgb,var(--c) 12%,transparent);white-space:nowrap}
.risk{font-weight:700;color:color-mix(in srgb,#d62828 calc(var(--r)*100%),#2a9d8f)}
.steps{padding-left:20px;margin:0}.steps li{margin:6px 0}.steps.row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:0;list-style:none;counter-reset:s}
.steps.row li{background:var(--bg);border-radius:10px;padding:12px;counter-increment:s}.steps.row li::before{content:counter(s);display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--accent);color:#fff;font-size:12px;margin-right:6px}
@media (max-width:860px){.steps.row{grid-template-columns:1fr}}
.ev{padding-left:18px;margin:8px 0}.ev li{margin:6px 0}.ev.small{font-size:13.5px}.muted{color:var(--muted);font-size:13px}
.foot{color:var(--muted);font-size:13px;margin:8px 0 0}.quote{font-style:italic;margin:4px 0 6px}mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px;font-style:normal;font-weight:600}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}table.compact td,table.compact th{padding:5px 8px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;font-size:12.5px;white-space:pre-wrap}code{font-size:13px;background:var(--bg);padding:1px 5px;border-radius:5px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}@media (max-width:860px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}.kpi .n{font-size:34px;font-weight:800;color:var(--accent)}.kpi .l{font-size:13.5px}.kpi .l span{color:var(--muted)}
.ok{color:#2a9d8f;font-weight:600}.warn{color:#b26a00;font-weight:600}
.diagram{width:100%;height:auto}.diagram rect{fill:var(--card);stroke:var(--line);stroke-width:1.5}.diagram .proc rect{stroke:var(--accent)}
.diagram rect.az{stroke:#0078d4;stroke-width:2.5}.diagram .fuse rect{fill:color-mix(in srgb,var(--accent) 12%,var(--card));stroke:var(--accent);stroke-width:2}
.diagram .out rect{stroke:var(--accent2)}.diagram text{text-anchor:middle;fill:var(--ink);font-size:14px;font-weight:600}.diagram text.s{font-size:11.5px;font-weight:400;fill:var(--muted)}
.diagram .links path{fill:none;stroke:var(--muted);stroke-width:1.6}
.data{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}@media (max-width:860px){.data{grid-template-columns:1fr 1fr}}
.data .card h4{font-size:15px}.data p{margin:0;font-size:13.5px;color:var(--muted)}
.shot{border:1px solid var(--line);margin-top:12px}.cmd{font-family:ui-monospace,Consolas,monospace}
footer{color:var(--muted);font-size:13px;text-align:center;padding:30px 16px}
"""


def build() -> Path:
    sections = [
        ("fluxo", "1", "Fluxo multimodal", "Três tipos de dado (vídeo, áudio, texto/tabular) processados por modelos especializados, fundidos por leito e convertidos em alertas escalonados.", section_flow()
         + """<div class="data">
<div class="card"><h4>🎥 UR Fall Detection</h4><p>70 vídeos reais (30 quedas, 40 atividades diárias), Univ. de Rzeszów.</p></div>
<div class="card"><h4>💓 MIMIC-IV Demo · PhysioNet</h4><p>100 pacientes: sinais vitais, 14 mil prescrições e registros de mobilidade (ODbL).</p></div>
<div class="card"><h4>🎙️ Consultas pt-BR + ESC-50</h4><p>4 consultas simuladas; sons reais de tosse e respiração (ontologia AudioSet).</p></div>
<div class="card"><h4>🧠 Modelos</h4><p>OpenPose (CMU) · YOLOv8s · AST-AudioSet · Azure Speech · Azure Language · Isolation Forest.</p></div></div>"""),
        ("video", "2", "Análise de vídeo", "Movimentos e eventos fora do padrão em vídeos do quarto e de fisioterapia, com relatório automático.", section_video()),
        ("audio", "3", "Análise de áudio", "Alterações vocais (fadiga, dificuldade respiratória, disartria) e termos críticos nas consultas.", section_audio()),
        ("azure", "4", "Integração Azure", "Serviços cognitivos gerenciados ampliam a capacidade de transcrição e compreensão clínica.", section_azure()),
        ("anomalias", "5", "Detecção de anomalias", "Séries temporais de sinais vitais, evolução de prescrições e padrões de movimentação.", section_anomaly()),
        ("alerta", "6", "Fusão e fluxo do alerta", "Cenário de 24h com 4 leitos de UTI: como os sinais se somam até chegar à equipe médica.", section_scenario()),
        ("resultados", "7", "Resultados", "Indicadores obtidos e exemplos de anomalias detectadas.", section_results()),
        ("roteiro", "8", "Roteiro sugerido do vídeo", "Até 15 minutos.", section_script()),
    ]
    nav = "".join(f"<a href='#{k}'>{n}. {t}</a>" for k, n, t, *_ in sections)
    body = "".join(f"<section id='{k}'><h2><span>{n}</span>{t}</h2><p class='lead'>{l}</p>{c}</section>" for k, n, t, l, c in sections)
    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitoramento Multimodal UTI</title><style>{CSS}</style></head><body>
<header class="hero"><div class="in"><div class="tag">Tech Challenge · Fase 4 · IA para Devs</div>
<h1>Monitoramento multimodal de pacientes com detecção precoce de risco</h1>
<p>Vídeo (OpenPose + YOLOv8), áudio (Azure Speech + Azure Text Analytics + AudioSet) e sinais clínicos (MIMIC-IV) fundidos em tempo real para alertar a equipe médica.</p></div></header>
<nav><div class="in">{nav}</div></nav><main>{body}</main>
<footer>Gerado automaticamente por <code>scripts/build_report.py</code> em {esc(S['gerado_em'])} · pipeline executado em {S['tempo_total_s']:.0f}s.
Uso acadêmico — não é um dispositivo médico.</footer></body></html>"""
    out = OUTPUTS / "relatorio_apresentacao.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
