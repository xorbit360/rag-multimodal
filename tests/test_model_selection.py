"""Tests de seleccion automatica del modelo mas economico.

Logica pura: sin red, sin claves, sin proveedores reales.
"""
from app.domain.model_selection import (
    ModelCandidate, select_model, select_transcription_model, supports_vision,
)


def _c(model_id, price=None, modality=None):
    return ModelCandidate(id=model_id, prompt_price=price, modality=modality)


def test_precio_mas_bajo_gana():
    cands = [_c("gpt-4o", 0.005), _c("gpt-4o-mini", 0.00015),
             _c("gpt-4.1", 0.002)]
    assert select_model(cands) == "gpt-4o-mini"


def test_vision_filtra_sin_vision():
    cands = [_c("llama-3-8b", 0.0001), _c("gpt-4o", 0.005)]
    # El barato no tiene vision: gana gpt-4o aunque cueste mas.
    assert select_model(cands, need_vision=True) == "gpt-4o"


def test_vision_sin_candidatos_devuelve_none():
    cands = [_c("llama-3-8b", 0.0001)]
    assert select_model(cands, need_vision=True) is None


def test_modalidad_openrouter_cuenta_como_vision():
    cands = [_c("proveedor/modelo-raro", 0.0001, "text+image->text"),
             _c("gpt-4o", 0.005)]
    assert supports_vision(cands[0]) is True
    assert select_model(cands, need_vision=True) == "proveedor/modelo-raro"


def test_sin_precios_usa_tabla_mini_primero():
    cands = [_c("gpt-4o"), _c("gpt-4o-mini"), _c("gpt-4.1")]
    assert select_model(cands) == "gpt-4o-mini"


def test_desconocidos_respetan_orden_del_proveedor():
    cands = [_c("modelo-zeta"), _c("modelo-alfa")]
    assert select_model(cands) == "modelo-zeta"


def test_sin_candidatos_none():
    assert select_model([]) is None
    assert select_model([], need_vision=True) is None


def test_transcripcion_prefiere_whisper_mas_barato():
    cands = [_c("whisper-large", 0.006), _c("whisper-1", 0.001),
             _c("gpt-4o-mini", 0.00015)]
    assert select_transcription_model(cands) == "whisper-1"


def test_transcripcion_sin_whisper_fallback():
    assert select_transcription_model([_c("gpt-4o-mini")]) == "whisper-1"
    assert select_transcription_model([]) == "whisper-1"


def test_automatico_excluye_modelos_no_chat():
    # Caso real 2026-11-04: una variante -tts compartia la subcadena
    # "gpt-4o-mini" y ganaba por aparecer antes, y el chat daba 404.
    cands = [_c("gpt-4o-mini-tts"), _c("whisper-1"),
             _c("text-embedding-3-small"), _c("gpt-4o-mini")]
    assert select_model(cands) == "gpt-4o-mini"


def test_automatico_prefiere_modelo_base_exacto_ante_variantes():
    cands = [_c("gpt-4o-mini-2024-07-18"), _c("gpt-4o-mini"),
             _c("gpt-4o-mini-search-preview")]
    assert select_model(cands) == "gpt-4o-mini"
