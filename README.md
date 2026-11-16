# Monitoramento Multimodal de Pacientes — Tech Challenge Fase 4

Sistema que monitora pacientes internados a partir de **vídeo, áudio e dados clínicos** e alerta a equipe médica sobre sinais precoces de risco.

| Requisito do desafio | Implementação |
|---|---|
| Análise de vídeo (OpenPose + YOLOv8) | `src/video/`: OpenPose (rede original CMU, 18 keypoints) + YOLOv8s para objetos e a **zona crítica de piso**. Detecta queda, imobilidade pós-queda e assimetria postural, com vídeo anotado e relatório automático |
| Análise de áudio (Azure Speech to Text + Azure Text Analytics) | `src/audio/`: transcrição pt-BR com tempo por palavra; sentimento, frases-chave e **Text Analytics for Health**; biomarcadores vocais (Praat); AST treinado no **AudioSet** |
| Anomalias em sinais vitais | `src/anomaly/vitals.py`: NEWS2, z-score robusto por paciente e Isolation Forest, em **streaming** |
| Anomalias em prescrições | `src/anomaly/prescriptions.py`: regras ISMP (escalonamento de dose, duplicidade, via, rajadas) + Isolation Forest |
| Anomalias de movimentação | `src/anomaly/movement.py` (MIMIC-IV: decúbito, atividade) + eventos de vídeo |
| Alertas automáticos à equipe | `src/common/alerts.py` (Alert Bus + webhook Teams/Slack) + `dashboard/app.py` (Central de Alertas) |
| Fusão multimodal | `src/fusion/risk.py`: noisy-OR com decaimento temporal + regra de corroboração entre modalidades |
| Datasets sugeridos | **PhysioNet** (MIMIC-IV Clinical Database Demo) e **AudioSet** (modelo AST e classes do ESC-50) |

📄 **Relatório técnico:** [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md) · 🖥️ **Relatório de apresentação:** `outputs/relatorio_apresentacao.html`

## Arquitetura

```
 Vídeo do quarto ──► OpenPose + YOLOv8 ─────────────────┐
 Áudio consulta  ──► Azure STT ─► Azure Text Analytics ─┤
                 └─► Praat/librosa + AST (AudioSet) ────┤     ┌─────────────┐    ┌───────────┐   ┌──────────────────┐
 Sinais vitais   ──► NEWS2 · z-score · Isolation Forest ┼───► │ Fusão tardia│──► │ Alert Bus │──►│ Central de Alertas│
 Prescrições     ──► Regras ISMP · Isolation Forest ────┤     │ (por leito) │    │ roteamento│   │ Webhook (Teams)   │
 Mobilidade      ──► Protocolo 2h · declínio funcional ─┘     └─────────────┘    └───────────┘   └──────────────────┘
```

## Como executar

```bash
# 1) Ambiente (Python 3.12). GPU NVIDIA série 50xx exige CUDA 12.8:
python -m venv .venv
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt

# 2) Credenciais Azure: copie .env.example para .env e preencha
#    (recursos "Speech" e "Language", plano gratuito F0)

# 3) Dados públicos e pesos dos modelos
.venv\Scripts\python -m scripts.download_data
.venv\Scripts\python -m scripts.generate_audio_cases

# 4) Pipeline completo (extração de pose em 70 vídeos leva ~20 min na GPU)
.venv\Scripts\python -m scripts.extract_video_features
.venv\Scripts\python -m scripts.run_pipeline
.venv\Scripts\python -m scripts.build_report

# 5) Demonstração em tempo real
.venv\Scripts\streamlit run dashboard/app.py          # terminal 1
.venv\Scripts\python -m scripts.stream_demo           # terminal 2 (24h clínicas em ~2,5 min) (--live-media processa vídeo/áudio ao vivo)

# Testes
.venv\Scripts\python -m pytest -q
```

## Estrutura

```
src/
  video/      openpose.py · analyzer.py · pose_features.py · evaluation.py
  audio/      azure_speech.py · azure_text.py · acoustic.py · audioset.py · analyzer.py
  anomaly/    vitals.py · prescriptions.py · movement.py
  fusion/     risk.py
  common/     alerts.py
  scenario.py · config.py
scripts/      download_data · generate_audio_cases · extract_video_features · run_pipeline · stream_demo · build_report
dashboard/    app.py (Central de Alertas — Streamlit)
config/       scenario.json (4 leitos da demonstração)
outputs/      vídeos anotados, relatórios automáticos, métricas, alerts.jsonl, summary.json, relatório HTML
tests/        testes unitários
```

## Dados e licenças

- **MIMIC-IV Clinical Database Demo v2.2**: Johnson et al., PhysioNet. Licença ODbL.
- **UR Fall Detection Dataset**: Kwolek & Kepski, 2014. Uso acadêmico.
- **ESC-50**: Piczak, 2015. CC BY-NC 3.0 (sons de tosse e respiração).
- **AST AudioSet**: Gong et al., 2021 (`MIT/ast-finetuned-audioset-10-10-0.4593`).
- **OpenPose**: Cao et al., CVPR 2017. Pesos COCO convertidos para PyTorch.
- **YOLOv8**: Ultralytics (AGPL-3.0).

> Projeto acadêmico. Não é dispositivo médico e não substitui avaliação clínica.
