"""Barramento de alertas para a equipe médica.

Todo módulo (vídeo, áudio, sinais vitais, prescrições, movimentação, fusão)
publica `Alert`s aqui. O barramento:
  1. persiste o alerta em `outputs/alerts.jsonl` (consumido pelo dashboard);
  2. imprime no console com cor por severidade;
  3. opcionalmente envia para um webhook (Microsoft Teams / Slack / Discord)
     definido em ALERT_WEBHOOK_URL, simulando o acionamento da equipe.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ALERT_WEBHOOK_URL, ALERTS_FILE

SEVERITY_ORDER = {"INFO": 0, "ATENCAO": 1, "ALTO": 2, "CRITICO": 3}

# Quem é acionado em cada nível (protocolo de escalonamento simulado)
ROUTING = {
    "INFO": ["Registro no prontuário"],
    "ATENCAO": ["Enfermagem do leito"],
    "ALTO": ["Enfermagem do leito", "Médico assistente"],
    "CRITICO": ["Enfermagem do leito", "Médico plantonista", "Time de resposta rápida"],
}

_COLORS = {"INFO": "\033[36m", "ATENCAO": "\033[33m", "ALTO": "\033[35m", "CRITICO": "\033[31m"}


@dataclass
class Alert:
    patient_id: str
    modality: str  # video | audio | vitais | prescricao | movimentacao | fusao
    severity: str  # INFO | ATENCAO | ALTO | CRITICO
    title: str
    description: str
    source: str  # modelo/técnica que gerou o alerta
    event_time: str | None = None  # horário clínico do evento (quando aplicável)
    evidence: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notify: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Severidade inválida: {self.severity}")
        if not self.notify:
            self.notify = list(ROUTING[self.severity])


class AlertBus:
    def __init__(self, path: Path = ALERTS_FILE, webhook_url: str | None = ALERT_WEBHOOK_URL,
                 echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self.echo = echo
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")

    def publish(self, alert: Alert) -> Alert:
        record = asdict(alert)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.echo:
            c = _COLORS.get(alert.severity, "")
            print(f"{c}[ALERTA {alert.severity}]\033[0m {alert.patient_id} | {alert.modality} | "
                  f"{alert.title} -> {', '.join(alert.notify)}")
        if self.webhook_url and SEVERITY_ORDER[alert.severity] >= SEVERITY_ORDER["ALTO"]:
            self._send_webhook(alert)
        return alert

    def _send_webhook(self, alert: Alert) -> None:
        import requests

        text = (f"🚨 [{alert.severity}] {alert.patient_id} — {alert.title}\n"
                f"{alert.description}\nOrigem: {alert.modality} / {alert.source}\n"
                f"Acionar: {', '.join(alert.notify)}")
        try:
            # Formato compatível com Teams (Incoming Webhook), Slack e Discord
            requests.post(self.webhook_url, json={"text": text, "content": text}, timeout=5)
        except requests.RequestException as exc:  # alerta continua registrado localmente
            print(f"[webhook] falha ao enviar: {exc}")


def load_alerts(path: Path = ALERTS_FILE) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
