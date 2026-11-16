"""Central de Alertas — monitoramento multimodal da UTI (Streamlit).

Execução:  streamlit run dashboard/app.py
Ao vivo:   em outro terminal, python -m scripts.stream_demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.alerts import ROUTING, load_alerts  # noqa: E402
from src.config import AZURE, OUTPUTS  # noqa: E402

st.set_page_config(page_title="Central de Alertas — UTI", page_icon="🏥", layout="wide")

SEV_COLOR = {"CRITICO": "#d62828", "ALTO": "#f77f00", "ATENCAO": "#e9c46a", "INFO": "#8d99ae"}
SEV_LABEL = {"CRITICO": "CRÍTICO", "ALTO": "ALTO", "ATENCAO": "ATENÇÃO", "INFO": "INFO"}
MOD_ICON = {"vitais": "💓", "video": "🎥", "audio": "🎙️", "prescricao": "💊", "movimentacao": "🛏️", "fusao": "🧠"}
ACKS = OUTPUTS / "acks.json"
STATE = OUTPUTS / "stream_state.json"
SC = OUTPUTS / "cenario"

st.markdown("""
<style>
.alert-card{border-left:6px solid;padding:8px 12px;margin:6px 0;border-radius:6px;background:rgba(127,127,127,.08)}
.alert-card small{opacity:.75}
.bed{border-radius:10px;padding:10px 14px;background:rgba(127,127,127,.08);border-top:6px solid}
.bed h4{margin:0 0 2px 0}
.kpi{font-size:13px;opacity:.85}
.chip{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;margin-right:6px;background:rgba(127,127,127,.15)}
</style>""", unsafe_allow_html=True)


def load_state() -> dict | None:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_acks() -> set:
    try:
        return set(json.loads(ACKS.read_text(encoding="utf-8")))
    except Exception:
        return set()


def ack(alert_id: str) -> None:
    a = load_acks()
    a.add(alert_id)
    ACKS.write_text(json.dumps(sorted(a)), encoding="utf-8")


def risk_color(r: float) -> str:
    return "#d62828" if r >= 0.85 else "#f77f00" if r >= 0.6 else "#e9c46a" if r >= 0.3 else "#2a9d8f"


# ------------------------------------------------------------------ cabeçalho
st.title("🏥 Central de Alertas — Monitoramento Multimodal da UTI")
azure_chips = (f"<span class='chip'>Azure Speech: {'✅ ' + (AZURE.speech_region or '') if AZURE.speech_enabled else '⚠️ fallback local'}</span>"
               f"<span class='chip'>Azure Text Analytics: {'✅' if AZURE.language_enabled else '⚠️ fallback local'}</span>"
               "<span class='chip'>OpenPose + YOLOv8</span><span class='chip'>AST/AudioSet</span>"
               "<span class='chip'>NEWS2 · z-score · Isolation Forest</span>")
st.markdown(azure_chips, unsafe_allow_html=True)

with st.sidebar:
    st.header("Filtros")
    sev_sel = st.multiselect("Severidade", list(SEV_LABEL), default=["CRITICO", "ALTO", "ATENCAO"],
                             format_func=lambda s: SEV_LABEL[s])
    mod_sel = st.multiselect("Modalidade", list(MOD_ICON), default=list(MOD_ICON),
                             format_func=lambda m: f"{MOD_ICON[m]} {m}")
    hide_ack = st.checkbox("Ocultar alertas confirmados", value=False)
    st.divider()
    st.caption("Protocolo de escalonamento")
    for s, who in ROUTING.items():
        st.markdown(f"<span style='color:{SEV_COLOR[s]}'>●</span> **{SEV_LABEL[s]}** → {', '.join(who)}",
                    unsafe_allow_html=True)


@st.fragment(run_every="2s")
def live_panel() -> None:
    state = load_state()
    alerts = load_alerts()
    sim_time = pd.Timestamp(state["sim_time"]) if state and state.get("sim_time") else None
    df = pd.DataFrame(alerts)
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"])
        if sim_time is not None:
            df = df[df["event_time"] <= sim_time]
    status = ("🟢 AO VIVO" if state and state.get("running") else "⏹️ simulação concluída") if state else "📼 replay (pipeline)"
    st.markdown(f"**{status}** — horário clínico: **{sim_time:%d/%m/%Y %H:%M}**" if sim_time is not None else f"**{status}**")

    # ---------------- leitos
    scn = json.loads((ROOT / "config" / "scenario.json").read_text(encoding="utf-8"))
    cols = st.columns(len(scn["patients"]))
    for col, p in zip(cols, scn["patients"]):
        pid = p["patient_id"]
        pst = (state or {}).get("patients", {}).get(pid, {})
        risk = pst.get("risk")
        if risk is None:  # sem simulação: usa risco final do pipeline
            f = SC / f"{pid.replace(' ', '_')}_risco.parquet"
            risk = float(pd.read_parquet(f)["risk"].iloc[-1]) if f.exists() else 0.0
        v = {k: x for k, x in pst.get("vitais", {}).items() if x is not None}
        if not v:  # sem simulação: último valor registrado no cenário
            fv = SC / f"{pid.replace(' ', '_')}_vitais.parquet"
            if fv.exists():
                last_v = pd.read_parquet(fv).ffill().iloc[-1]
                v = {s: round(float(last_v[s]), 1) for s in ("hr", "spo2", "sbp", "rr") if pd.notna(last_v[s])}
                pst = {**pst, "news2": int(last_v["news2"])}
        sub = df[df["patient_id"] == pid] if not df.empty else df
        n_crit = int((sub["severity"] == "CRITICO").sum()) if not sub.empty else 0
        n_high = int((sub["severity"] == "ALTO").sum()) if not sub.empty else 0
        last = sub.iloc[-1] if not sub.empty else None
        with col:
            st.markdown(
                f"<div class='bed' style='border-top-color:{risk_color(risk)}'>"
                f"<h4>{pid}</h4><div class='kpi'>{p['nome']}</div>"
                f"<div style='font-size:28px;font-weight:700;color:{risk_color(risk)}'>{100 * risk:.0f}% <span style='font-size:13px'>risco fundido</span></div>"
                f"<div class='kpi'>NEWS2 <b>{pst.get('news2', '–')}</b> · FC <b>{v.get('hr', '–')}</b> · SpO2 <b>{v.get('spo2', '–')}</b> · "
                f"PAS <b>{v.get('sbp', '–')}</b> · FR <b>{v.get('rr', '–')}</b></div>"
                f"<div class='kpi'>🔴 {n_crit} críticos · 🟠 {n_high} altos</div>"
                + (f"<div class='kpi' style='margin-top:4px'>Último: {MOD_ICON.get(last['modality'], '')} {last['title']}</div>" if last is not None else "")
                + "</div>", unsafe_allow_html=True)

    # ---------------- feed
    st.subheader("📣 Alertas para a equipe")
    if df.empty:
        st.info("Nenhum alerta ainda. Rode `python -m scripts.run_pipeline` ou `python -m scripts.stream_demo`.")
        return
    acks = load_acks()
    f = df[df["severity"].isin(sev_sel) & df["modality"].isin(mod_sel)]
    if hide_ack:
        f = f[~f["id"].isin(acks)]
    pinned = f[(f["severity"] == "CRITICO") & ~f["id"].isin(acks)].sort_values("event_time", ascending=False)
    if not pinned.empty:
        st.markdown(f"#### 🚨 Críticos aguardando confirmação ({len(pinned)})")
        for _, a in pinned.head(6).iterrows():
            render_alert(a, acks, prefix="pin")
        st.markdown("#### Todos os alertas (mais recentes primeiro)")
    for _, a in f.sort_values("event_time", ascending=False).head(40).iterrows():
        render_alert(a, acks, prefix="feed")


def render_alert(a, acks: set, prefix: str) -> None:
    c1, c2 = st.columns([12, 1])
    with c1:
        ok = "✔️ confirmado · " if a["id"] in acks else ""
        st.markdown(
            f"<div class='alert-card' style='border-left-color:{SEV_COLOR[a['severity']]}'>"
            f"<b style='color:{SEV_COLOR[a['severity']]}'>{SEV_LABEL[a['severity']]}</b> · "
            f"<b>{a['patient_id']}</b> · {MOD_ICON.get(a['modality'], '')} {a['title']}<br>"
            f"{a['description']}<br><small>{ok}{a['event_time']:%H:%M} · origem: {a['source']} · "
            f"acionar: <b>{', '.join(a['notify'])}</b></small></div>", unsafe_allow_html=True)
    with c2:
        if a["id"] not in acks and a["severity"] in ("CRITICO", "ALTO"):
            if st.button("OK", key=f"ack_{prefix}_{a['id']}", help="Confirmar recebimento"):
                ack(a["id"])
                st.rerun(scope="fragment")


live_panel()

# ------------------------------------------------------------------ detalhe do leito
st.divider()
st.subheader("🔎 Detalhe do leito")
scn = json.loads((ROOT / "config" / "scenario.json").read_text(encoding="utf-8"))
pid = st.selectbox("Leito", [p["patient_id"] for p in scn["patients"]])
spec = next(p for p in scn["patients"] if p["patient_id"] == pid)
key = pid.replace(" ", "_")
tabs = st.tabs(["💓 Sinais vitais", "🧠 Risco fundido", "🎙️ Áudio", "🎥 Vídeo", "📋 Linha do tempo"])

ev_all = pd.read_parquet(SC / "eventos.parquet") if (SC / "eventos.parquet").exists() else pd.DataFrame()
ev = ev_all[ev_all["patient_id"] == pid] if not ev_all.empty else ev_all

with tabs[0]:
    f = SC / f"{key}_vitais.parquet"
    if f.exists():
        vt = pd.read_parquet(f)
        sigs = [("hr", "FC (bpm)"), ("sbp", "PAS (mmHg)"), ("spo2", "SpO2 (%)"), ("rr", "FR (irpm)"), ("news2", "NEWS2")]
        fig = make_subplots(rows=len(sigs), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=[s[1] for s in sigs])
        va = ev[ev["modality"] == "vitais"] if not ev.empty else ev
        for i, (s, lab) in enumerate(sigs, 1):
            fig.add_trace(go.Scatter(x=vt["charttime"], y=vt[s], mode="lines+markers", name=lab,
                                     line=dict(width=1.5), marker=dict(size=4)), row=i, col=1)
            for sev in ("CRITICO", "ALTO"):
                m = va[va["severity"] == sev]
                if not m.empty:
                    y = vt.set_index("charttime")[s].reindex(m["time"]).values
                    fig.add_trace(go.Scatter(x=m["time"], y=y, mode="markers", name=SEV_LABEL[sev], showlegend=(i == 1),
                                             marker=dict(color=SEV_COLOR[sev], size=10, symbol="x")), row=i, col=1)
        for t, label in ((ev[ev["modality"] == "audio"]["time"].min() if not ev.empty else None, "consulta (áudio)"),
                         (ev[ev["modality"] == "video"]["time"].min() if not ev.empty else None, "vídeo")):
            if t is not None and pd.notna(t):
                fig.add_vline(x=t, line_dash="dot", line_color="#6c757d")
                fig.add_annotation(x=t, y=1.02, yref="paper", text=label, showarrow=False, font=dict(size=11))
        fig.update_layout(height=780, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Fonte: MIMIC-IV Demo (PhysioNet), internação {spec['stay_id']}, deslocada para a data da demonstração.")

with tabs[1]:
    f = SC / f"{key}_risco.parquet"
    if f.exists():
        rk = pd.read_parquet(f)
        fig = go.Figure()
        for c in [c for c in rk.columns if c.startswith("risk_")]:
            fig.add_trace(go.Scatter(x=rk["time"], y=rk[c], name=c[5:], stackgroup=None, line=dict(width=1)))
        fig.add_trace(go.Scatter(x=rk["time"], y=rk["risk"], name="RISCO FUNDIDO", line=dict(width=4, color="#d62828")))
        fus = ev[ev["modality"] == "fusao"] if not ev.empty else ev
        for _, r in fus.iterrows():
            fig.add_vline(x=r["time"], line_color=SEV_COLOR[r["severity"]], line_dash="dash")
        fig.update_layout(height=420, yaxis=dict(range=[0, 1], title="risco"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Noisy-OR das modalidades com decaimento temporal; linhas tracejadas = alertas de corroboração multimodal.")

with tabs[2]:
    if "audio" in spec:
        stem = Path(spec["audio"]["file"]).stem
        st.audio(str(ROOT / spec["audio"]["file"]))
        rep_f = OUTPUTS / "audio" / f"{stem}.json"
        if rep_f.exists():
            rep = json.loads(rep_f.read_text(encoding="utf-8"))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Transcrição** ({rep['transcricao']['provider']})")
                text = rep["transcricao"]["text"]
                for t in sorted({t["termo"] for t in rep["nlp"].get("termos_criticos", [])}, key=len, reverse=True):
                    text = text.replace(t, f"**:red[{t}]**")
                st.markdown(text)
                s = rep["nlp"].get("sentimento", {})
                st.markdown(f"**Sentimento** ({rep['nlp'].get('provider')}): `{s.get('geral')}` "
                            f"— negativo {s.get('scores', {}).get('negativo', 0):.2f}")
                if rep["nlp"].get("frases_chave"):
                    st.markdown("**Frases-chave:** " + ", ".join(rep["nlp"]["frases_chave"]))
                if rep.get("sintomas_azure_health"):
                    st.markdown("**Entidades de saúde (Azure):** " + ", ".join(rep["sintomas_azure_health"]))
            with c2:
                ind = rep["indicadores_vocais"]
                fig = go.Figure(go.Bar(x=[ind[k]["score"] for k in ind], y=list(ind), orientation="h",
                                       marker_color=[risk_color(ind[k]["score"]) for k in ind]))
                fig.update_layout(height=220, xaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                for a in rep["achados"]:
                    st.markdown(f"<span style='color:{SEV_COLOR[a['severidade']]}'>●</span> **{a['condicao']}** — "
                                f"{a['descricao']}<br><small>{'; '.join(a['evidencias'])}</small>", unsafe_allow_html=True)
                rs = {k: v for k, v in rep["audioset"]["max_prob"].items() if v > 0.1 and k != "Speech"}
                if rs:
                    st.markdown("**AudioSet (AST):** " + ", ".join(f"{k} {v:.2f}" for k, v in rs.items()))
    else:
        st.info("Sem gravação para este leito.")

with tabs[3]:
    if "video" in spec:
        stem = Path(spec["video"]["file"]).stem
        mp4 = OUTPUTS / "video" / f"{stem}_anotado.mp4"
        if mp4.exists():
            st.video(str(mp4))
        rep_f = OUTPUTS / "video" / f"{stem}_relatorio.md"
        if rep_f.exists():
            st.markdown(rep_f.read_text(encoding="utf-8"))
    else:
        st.info("Sem vídeo para este leito.")

with tabs[4]:
    if not ev.empty:
        t = ev[["time", "severity", "modality", "title", "description", "source"]].copy()
        t["time"] = t["time"].dt.strftime("%H:%M")
        st.dataframe(t, width="stretch", hide_index=True)
