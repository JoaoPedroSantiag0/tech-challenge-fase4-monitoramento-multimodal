"""Configuração central do sistema de monitoramento multimodal.

Carrega variáveis do arquivo `.env` (credenciais Azure, webhook de alertas)
e expõe caminhos padrão do projeto.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
ALERTS_FILE = OUTPUTS / "alerts.jsonl"

for _p in (DATA_PROCESSED, MODELS_DIR, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AzureSettings:
    speech_key: str | None = os.getenv("AZURE_SPEECH_KEY") or None
    speech_region: str | None = os.getenv("AZURE_SPEECH_REGION") or None
    language_key: str | None = os.getenv("AZURE_LANGUAGE_KEY") or None
    language_endpoint: str | None = os.getenv("AZURE_LANGUAGE_ENDPOINT") or None
    speech_language: str = os.getenv("AZURE_SPEECH_LANGUAGE", "pt-BR")

    @property
    def speech_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_enabled(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


AZURE = AzureSettings()
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL") or None
