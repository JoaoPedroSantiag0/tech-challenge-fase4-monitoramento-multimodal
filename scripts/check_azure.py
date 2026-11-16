"""Testa as credenciais Azure configuradas no .env.

Uso: python -m scripts.check_azure
"""
from __future__ import annotations

from src.config import AZURE, DATA_RAW


def main() -> None:
    ok = True
    print(f"Speech:   chave {'OK' if AZURE.speech_key else 'AUSENTE'} · região {AZURE.speech_region}")
    print(f"Language: chave {'OK' if AZURE.language_key else 'AUSENTE'} · endpoint {AZURE.language_endpoint}")
    if AZURE.speech_enabled:
        from src.audio.azure_speech import transcribe_azure

        try:
            r = transcribe_azure(DATA_RAW / "audio" / "consultas" / "caso_01_controle.wav")
            print(f"[OK] Azure Speech to Text ({r['latency_s']}s): \"{r['text'][:90]}...\"")
        except Exception as exc:
            ok = False
            print(f"[ERRO] Azure Speech: {exc}")
    if AZURE.language_enabled:
        from src.audio.azure_text import analyze_azure

        r = analyze_azure("Estou com falta de ar e uma dor no peito desde ontem.")
        if r["errors"]:
            ok = False
            print(f"[ERRO] Azure Language: {r['errors']}")
        else:
            print(f"[OK] Azure Text Analytics: sentimento={r['sentimento']['geral']} · "
                  f"frases-chave={r.get('frases_chave')} · saúde={[e['texto'] for e in r.get('entidades_saude', [])]}")
    if not (AZURE.speech_enabled and AZURE.language_enabled):
        ok = False
    print("\nTudo certo!" if ok else "\nAinda há pendências acima.")


if __name__ == "__main__":
    main()
