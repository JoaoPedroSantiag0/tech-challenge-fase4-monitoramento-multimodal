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


def _env(name: str, default: str | None = None) -> str | None:
    """Lê variável de ambiente ignorando espaços e comentários colados ao valor."""
    v = (os.getenv(name) or "").split(" #")[0].strip()
    return v or default


@dataclass(frozen=True)
class AzureSettings:
    speech_key: str | None = _env("AZURE_SPEECH_KEY")
    speech_region: str | None = _env("AZURE_SPEECH_REGION")
    language_key: str | None = _env("AZURE_LANGUAGE_KEY")
    language_endpoint: str | None = _env("AZURE_LANGUAGE_ENDPOINT")
    speech_language: str = _env("AZURE_SPEECH_LANGUAGE", "pt-BR")

    @property
    def speech_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_enabled(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


AZURE = AzureSettings()
ALERT_WEBHOOK_URL = _env("ALERT_WEBHOOK_URL")
