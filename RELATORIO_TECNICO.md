# Relatório Técnico: Monitoramento Multimodal de Pacientes

**Tech Challenge, Fase 4.** O sistema analisa vídeo, áudio e texto/dados clínicos, detecta anomalias em tempo real e alerta a equipe médica.

> Todos os números deste relatório são gerados pelo pipeline (`python -m scripts.run_pipeline`) e ficam em `outputs/summary.json`. A versão visual, para apresentação, está em `outputs/relatorio_apresentacao.html`.

---

## 1. Descrição do fluxo multimodal

```
┌──────────────────┐   ┌───────────────────────────────────────────┐
│ Vídeo do quarto  │──►│ OpenPose (pose 2D) + YOLOv8 (objetos/zona) │──┐
└──────────────────┘   └───────────────────────────────────────────┘  │
┌──────────────────┐   ┌───────────────────────────────────────────┐  │   ┌──────────────┐   ┌──────────┐   ┌───────────────────┐
│ Áudio da consulta│──►│ Azure Speech to Text ─► Azure Text Analytics│──┼──►│ Fusão tardia │──►│ AlertBus │──►│ Central de Alertas │
│                  │──►│ Biomarcadores (Praat) + AST/AudioSet       │  │   │ por leito    │   │ roteia   │   │ Webhook (Teams)    │
└──────────────────┘   └───────────────────────────────────────────┘  │   └──────────────┘   └──────────┘   └───────────────────┘
┌──────────────────┐   ┌───────────────────────────────────────────┐  │
│ Sinais vitais    │──►│ NEWS2 · z-score robusto · Isolation Forest │──┤
│ Prescrições      │──►│ Regras ISMP · Isolation Forest de ordens   │──┤
│ Mobilidade       │──►│ Protocolo de decúbito 2h · declínio        │──┘
└──────────────────┘   └───────────────────────────────────────────┘
```

1. **Ingestão.** Cada leito tem fontes contínuas (monitor de sinais vitais, prontuário de prescrições, registros de enfermagem) e fontes por evento (câmera do quarto, gravação de consulta).
2. **Análise por modalidade.** Cada módulo devolve *eventos* padronizados com horário, modalidade, severidade (INFO, ATENÇÃO, ALTO, CRÍTICO), título, descrição, técnica de origem e evidências.
3. **Fusão tardia** (`src/fusion/risk.py`):
   - Cada severidade vira um peso (0,05 · 0,25 · 0,55 · 0,85), que decai exponencialmente com uma meia-vida própria de cada modalidade.
   - Dentro de uma modalidade vale o *maior* risco vigente, porque alertas repetidos da mesma fonte não são evidências independentes.
   - Entre modalidades, o risco do leito é combinado por **noisy-OR**.
   - **Regra de corroboração** (janela de 6h): ≥ 2 modalidades com ALTO+ geram um alerta de fusão CRÍTICO; 1 modalidade ALTO+ corroborada por outra ATENÇÃO+ gera um alerta de fusão ALTO.
4. **Alertas** (`src/common/alerts.py`):
   - O AlertBus grava cada alerta em `outputs/alerts.jsonl` (trilha auditável).
   - O alerta é roteado por severidade: ATENÇÃO → enfermagem; ALTO → + médico assistente; CRÍTICO → + plantonista e time de resposta rápida.
   - Alertas ALTO+ também são enviados a um webhook (Teams/Slack/Discord).
5. **Central de Alertas** (`dashboard/app.py`, Streamlit):
   - Mostra os leitos com risco fundido e os alertas em tempo real, com os críticos não confirmados fixados no topo e confirmação de recebimento.
   - No detalhe de cada leito: gráficos de sinais vitais, curva de risco, áudio com transcrição e vídeo anotado.
6. **Tempo real** (`scripts/stream_demo.py`): reproduz 24h clínicas de forma acelerada. Os sinais vitais são processados observação a observação pelo `VitalsStreamMonitor`. Com `--live-media`, vídeo e áudio são processados pelos modelos no momento do evento.

### Dados utilizados

| Modalidade | Fonte | Observação |
|---|---|---|
| Vídeo | **UR Fall Detection Dataset** (Kwolek & Kepski, 2014): 30 quedas + 40 atividades diárias, câmera frontal | Vídeos reais, metade RGB |
| Sinais vitais | **MIMIC-IV Clinical Database Demo v2.2** (PhysioNet): 19.827 medições de 140 internações de UTI | FC, PAS/PAD/PAM (invasiva ou não), SpO₂, FR, temperatura |
| Prescrições | MIMIC-IV Demo: 14.391 ordens de 250 internações | Dose, unidade, via, frequência |
| Movimentação | MIMIC-IV Demo: 12.166 registros de enfermagem | Mudança de decúbito (224082), posição (224093), atividade (224084) |
| Áudio | 4 consultas pt-BR **simuladas**: voz neural + degradações controladas + sons reais de tosse e respiração do **ESC-50** (ontologia AudioSet) | Não existe base pública pt-BR de fala patológica rotulada |

**Cenário de demonstração** (`config/scenario.json`): 4 leitos. Cada um recebe uma janela real de 24h de uma internação do MIMIC-IV (vitais, prescrições, mobilidade), deslocada para a data da demonstração, mais um vídeo e/ou uma consulta em áudio. As modalidades vêm de bases distintas, então a associação ao mesmo leito é ilustrativa. O Isolation Forest de vitais usado no cenário foi treinado **sem** essas 4 internações.

---

## 2. Modelos aplicados em cada tipo de dado

### 2.1 Vídeo (`src/video/`)

| Componente | Modelo / técnica | Função |
|---|---|---|
| Pose | **OpenPose** (Cao et al., CVPR 2017): rede original da CMU com os pesos COCO-18 oficiais, reimplementada em PyTorch (`openpose.py`). Pós-processamento de heatmaps na GPU; associação por *Part Affinity Fields* | Esqueleto de 18 keypoints, ~25 fps (RTX 5060 Ti) |
| Objetos e áreas críticas | **YOLOv8s** (Ultralytics, COCO) | Pessoa, cama, sofá, cadeira, objetos. Define a **zona crítica "piso"** e distingue deitar na cama/sofá (esperado) de deitar no chão |
| Features | `pose_features.py` | Inclinação do tronco, altura da cabeça, velocidade de descida, ângulos de joelho/quadril/cotovelo/ombro, energia de movimento, proporção da caixa |
| Eventos | Regras temporais calibradas | **Queda** (descida rápida da cabeça + postura horizontal/no piso fora de cama/sofá); **imobilidade pós-queda**; **assimetria D/E** > 25° em exercício (fisioterapia) |
| Relatório automático | `build_report` / `report_markdown` | Status, tempo por postura, eventos com horário, amplitude articular (p5–p95), vídeo anotado e keyframe |

**Calibração sem vazamento.** Treino = quedas 01–15 + ADL 01–20; teste = quedas 16–30 + ADL 21–40. Os limiares foram escolhidos por busca em grade (270 combinações), maximizando a acurácia balanceada **apenas no treino**, e depois aplicados uma única vez no teste. Limiares finais:
- descida da cabeça ≥ 0,35 altura/s em 0,5 s;
- ≥ 0,5 s no piso;
- tronco > 50°;
- zona de piso y > 0,66.

### 2.2 Áudio (`src/audio/`)

| Componente | Modelo / serviço | Função |
|---|---|---|
| Transcrição | **Azure AI Speech: Speech to Text** (pt-BR, reconhecimento contínuo, timestamps por palavra, `PushAudioInputStream`) | Texto da consulta + ritmo de fala (palavras/min, pausas entre palavras) |
| Texto | **Azure AI Language (Text Analytics)**: sentimento com *opinion mining*, *key phrases*, **Text Analytics for Health** | Polaridade por frase, frases-chave, sintomas/diagnósticos com detecção de negação |
| Termos críticos | Léxico clínico pt-BR (`azure_text.py`) | Mapeia termos em síndromes: respiratória, cardíaca, **neurológica/AVC (FAST)**, fadiga, sinais de alarme, sofrimento emocional |
| Biomarcadores vocais | Praat (parselmouth) + librosa (`acoustic.py`) | Taxa de fala e articulação (núcleos silábicos), proporção de pausas, F0 e variabilidade, jitter, shimmer, HNR, irregularidade do ritmo, energia de consoantes > 2 kHz |
| Eventos sonoros | **AST** (Audio Spectrogram Transformer) treinado no **AudioSet** (527 classes) | Respiração, arquejo, ofego, sibilo, tosse, pigarro |
| Indicadores | Comparação com a **linha de base** (gravação estável do paciente ou perfil populacional) | **Fadiga**: fala mais lenta, mais pausas, menos energia, prosódia monótona. **Dificuldade respiratória**: sons respiratórios + frases entrecortadas. **Disartria**: perda de nitidez + ritmo irregular + articulação lenta |
| Regras de decisão | `analyzer.py` | Disartria + sinais focais → **suspeita de AVC (CRÍTICO)**; desconforto respiratório + dor torácica → **CRÍTICO**; fadiga voz + relato → ALTO |

Sem credenciais Azure, o sistema usa Whisper-small e o léxico local como fallback, e o provedor usado fica registrado em cada resultado.

### 2.3 Sinais vitais (`src/anomaly/vitals.py`)

| Detector | Técnica | Papel |
|---|---|---|
| Limpeza | Faixas fisiologicamente plausíveis | Removeu **89 artefatos** de sensor (ex.: FC = 0, PAM = 801) |
| Clínico | **NEWS2** (Royal College of Physicians) | Escore de alerta precoce; ≥ 7 CRÍTICO, ≥ 5 ou parâmetro vermelho ALTO |
| Estatístico | **z-score robusto** (mediana/MAD móvel das 12 observações anteriores) | Mudança abrupta em relação ao *próprio* paciente |
| ML | **Isolation Forest** (300 árvores): valores + desvio da linha de base + variações + índice de choque | Combinações multivariadas atípicas |
| Streaming | `VitalsStreamMonitor` | Uma observação por vez, cooldown por tipo de alerta; o NEWS2 só realerta quando piora ou como lembrete |

### 2.4 Prescrições (`src/anomaly/prescriptions.py`)

- **Regras de farmacovigilância** (lista ISMP de medicamentos de alta vigilância):
  - escalonamento (≥ 2,5×) ou redução abrupta (≤ 0,3×) da dose diária do mesmo fármaco, via e unidade;
  - dose acima do p99,5 populacional;
  - duplicidade terapêutica concomitante (anticoagulantes, opioides IV, sedativos), com exceção da ponte heparina→varfarina;
  - escalonamento de via oral → IV;
  - "rajada" de ≥ 4 novos medicamentos de alta vigilância em 6h.
- **Isolation Forest** no nível da ordem, com as features: dose z-score populacional, razão em relação à dose anterior, nº de alterações em 24h, alta vigilância e via IV.
- Deduplicação: no máximo 1 alerta por fármaco/regra a cada 24h.

### 2.5 Movimentação (`src/anomaly/movement.py` + vídeo)

- **Sem reposicionamento** por > 3h em paciente restrito ao leito (> 4h = ALTO). O protocolo de prevenção de lesão por pressão prevê 2h; o z-score compara com o intervalo habitual do próprio paciente.
- **Declínio funcional:** o melhor nível de atividade do dia cai ≥ 2 níveis em relação ao dia anterior (escala: leito → beira do leito → cadeira → deambulação).
- **Perda de autonomia:** o paciente que se virava sozinho passa a depender da equipe de forma sustentada.
- **Eventos do vídeo:** queda e imobilidade pós-queda.

---

## 3. Resultados obtidos

### 3.1 Vídeo: detecção de quedas (UR Fall, 70 vídeos reais)

| Conjunto | TP | FP | FN | TN | Sensibilidade | Especificidade | Precisão | F1 |
|---|---|---|---|---|---|---|---|---|
| Treino (35) | 13 | 1 | 2 | 19 | 86,7% | 95,0% | 92,9% | 0,90 |
| **Teste (35, não vistos)** | **13** | **3** | **2** | **17** | **86,7%** | **85,0%** | **81,2%** | **0,84** |

- Antes de restringir as "superfícies de apoio" a cama e sofá, a sensibilidade era de 20%. Quedas perto de cadeiras eram descartadas porque a caixa da pessoa se sobrepunha à cadeira.
- Os 3 falsos positivos do teste (adl-30, 38, 40) são pessoas que **se abaixam deliberadamente até o chão** para olhar embaixo da cama. Clinicamente, um paciente internado no chão também merece verificação.
- Os 2 falsos negativos são quedas com descida lenta ou com o corpo parcialmente fora de quadro.

### 3.2 Áudio: 4 consultas

| Caso | Fadiga | Dif. respiratória | Disartria | AudioSet (AST) | Resultado |
|---|---|---|---|---|---|
| Controle | 0,00 | 0,01 | 0,00 | fala | **Normal** |
| Fadiga | **0,77** | 0,02 | 0,18 | fala | **ALTO**: fadiga (voz + "cansado", "fraqueza", "esgotamento") |
| Dispneia | 0,12 | **0,70** | 0,07 | arquejo 0,47 · pigarro 0,63 · tosse 0,14 | **CRÍTICO**: desconforto respiratório + dor torácica |
| Disartria | 0,31 | 0,02 | **0,56** | fala | **CRÍTICO**: suspeita de AVC (perda de 98% da energia de consoantes + "fala enrolada", "braço fraco", "boca torta") |

As 4 consultas foram classificadas corretamente. Uma observação: a fala disártrica degradou a própria transcrição ("formigando" virou "me ligando"). Isso mostra por que os biomarcadores acústicos complementam a análise do texto.

### 3.3 Sinais vitais: avaliação com eventos injetados

Isolation Forest treinado em 70% das internações. Nos outros 30% foram injetados **102 eventos clínicos rotulados** (taquicardia, dessaturação, hipotensão, taquipneia, febre).

| Detector | Eventos detectados | Carga de alarmes* | Precisão pontual |
|---|---|---|---|
| NEWS2 ≥ 5 | 67,6% (69/102) | 27,3% | 0,08 |
| z-score robusto | 64,7% (66/102) | 6,4% | 0,23 |
| Isolation Forest | 43,1% (44/102) | 2,1% | 0,32 |
| **Ensemble (≥ 2 votos)** | **58,8% (60/102)** | **3,2%** | **0,34** |

\* Fração das medições *sem* evento injetado que são sinalizadas. Inclui anomalias reais de pacientes de UTI, então não é "falso positivo" puro.

Em UTI, o NEWS2 isolado fica ≥ 5 em 25% do tempo, o que gera fadiga de alarmes. Combinar a regra clínica com a linha de base individual e o ML **reduz a carga de alarmes ~9×** e ainda detecta 59% dos eventos.

### 3.4 Prescrições e movimentação (base MIMIC-IV Demo completa)

- **Prescrições.** Os eventos por severidade (ALTO / ATENÇÃO / INFO):
  - escalonamento de dose: 79 / 147 / 100;
  - redução abrupta: 36 / 82 / 91;
  - duplicidade terapêutica: 3 / 50 / 169;
  - rajada de alta vigilância: 0 / 92 / 0;
  - dose acima do p99,5: 5 / 12 / 0;
  - escalonamento de via: só INFO (279).

  O Isolation Forest marcou 264 ordens atípicas, das quais 159 (60%) também foram sinalizadas pelas regras.
- **Movimentação.** 479 alertas de lacuna de reposicionamento (184 ALTO), 14 de declínio funcional e 11 de perda de autonomia, em 130 internações.

### 3.5 Cenário multimodal: 24h, 4 leitos

| Leito | Composição | Alertas críticos | Risco máximo | Alertas de fusão |
|---|---|---|---|---|
| 01 | Fadiga (09h) → NEWS2 crescente → **queda no vídeo (21h30)** | 10 | 98% | 3 críticos (prescrição+vitais; áudio+prescrição+vitais; **vídeo+vitais**) |
| 02 | Dessaturação real (SpO₂ 84%) + **dispneia com dor torácica** 30 min antes da menor SpO₂ | 8 | 99% | 2 altos, 2 críticos (incluindo **áudio+vitais**) |
| 03 | Pico hipertensivo (PAS 184 mmHg) + **suspeita de AVC** no áudio | 8 | 99% | 3 altos, 2 críticos (incluindo **áudio+vitais**) |
| 04 | Controle: vídeo normal, consulta normal | **0** | 51% | **nenhum** |

### Exemplos de anomalias detectadas

- **Queda** (vídeo, CRÍTICO): "Queda detectada: descida rápida da cabeça (0,36 alturas/s) seguida de postura horizontal/no piso por 2,1s fora de cama/sofá". Junto com o NEWS2 = 11 vigente, gera fusão CRÍTICA "Deterioração multimodal (vídeo + vitais)".
- **Suspeita de AVC** (áudio, CRÍTICO): disartria na voz + relato de sinais neurológicos focais.
- **SpO₂ fora do padrão do paciente** (vitais, ALTO, Leito 02, 18h): z = −8,8 com SpO₂ 84%, FC 133 e FR 27.
- **Escalonamento de vasopressor** (prescrição, ALTO): fenilefrina IV 12 → 60 mg/dia (×5), medicamento de alta vigilância.
- **Sem mudança de decúbito por 8,0h** (movimentação, ALTO, Leito 02): padrão habitual do paciente de 2,0h, z = +6,7, risco de lesão por pressão.

---

## 4. Integração com Azure

| Serviço | Arquivo | Configuração |
|---|---|---|
| Azure AI Speech: Speech to Text | `src/audio/azure_speech.py` | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, idioma `pt-BR` |
| Azure AI Language: Sentimento, Key Phrases, Text Analytics for Health | `src/audio/azure_text.py` | `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT` |

As credenciais ficam no `.env`, que nunca é versionado (veja `.env.example`). O status de cada serviço aparece na Central de Alertas e no relatório HTML, e cada resultado registra o provedor usado.

**Evolução para produção:**
- Azure IoT Hub/Event Hubs para os monitores;
- Azure Functions para o AlertBus;
- Azure ML para retreino e monitoramento de deriva;
- Azure Communication Services para notificar o plantonista.

---

## 5. Limitações

- As modalidades vêm de bases públicas distintas, então a associação ao mesmo leito é ilustrativa.
- O áudio é simulado de forma controlada. O próximo passo é validar com gravações clínicas reais e consentidas, e com dados de pacientes com disartria e dispneia.
- O UR Fall tem câmera lateral e ambiente controlado. Outros ângulos exigem recalibrar a zona crítica. Para cirurgias, o YOLOv8 precisaria de ajuste fino com dados de instrumentais (ex.: m2cai16-tool).
- A avaliação dos sinais vitais usa eventos injetados, porque o MIMIC-IV Demo não tem rótulos de deterioração.
- Projeto acadêmico: não é dispositivo médico. Uso real exige validação clínica, conformidade com a LGPD e governança de IA.
