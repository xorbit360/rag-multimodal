"""Endpoints propios (p. ej. Ollama local) sin token.

Ollama no exige clave: la deteccion y el chat deben funcionar con solo
la base_url personalizada. Todo mockeado: ninguna prueba llama a un
proveedor real ni usa una clave real.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import _resolve_key_and_selection, app

client = TestClient(app)

CUSTOM_URL = "http://localhost:11434/v1"


def test_detect_custom_sin_token(monkeypatch):
    def fake_detect(api_key, base_url=None):
        assert api_key == ""
        assert base_url == CUSTOM_URL
        return ("custom", [{"id": "llama3.1", "prompt_price": None,
                            "modality": None}])

    monkeypatch.setattr(main_module, "detect_provider_info", fake_detect)
    resp = client.post("/llm/detect",
                       json={"api_key": "", "base_url": CUSTOM_URL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected"] is True
    assert body["provider"] == "custom"
    assert "llama3.1" in body["models"]


def test_detect_sin_token_ni_url_sigue_400():
    resp = client.post("/llm/detect", json={"api_key": ""})
    assert resp.status_code == 400


def test_resolver_custom_vacio_no_toca_la_boveda(monkeypatch):
    def _boom(_browser_id):
        raise AssertionError("la boveda no debe consultarse para custom")

    monkeypatch.setattr(main_module.container.token_vault, "resolve", _boom)
    out = _resolve_key_and_selection("custom", CUSTOM_URL, None, True,
                                     None, "", None)
    assert out[0] == ""
    assert out[1] == "custom"
    assert out[2] == CUSTOM_URL
    assert out[6] == "request"


def test_ask_custom_sin_token_usa_el_endpoint(monkeypatch):
    seen = {}

    def fake_resolve_llm(provider, model, api_key, base_url, auto_model,
                         models_info_json, need_vision):
        seen.update(provider=provider, model=model, api_key=api_key,
                    base_url=base_url)
        return (object(), "llama3.1", False)

    class FakeUseCase:
        def __init__(self, *args):
            pass

        def ask(self, question, top_k=5):
            return SimpleNamespace(text="respuesta custom", sources=[],
                                   image_id=None, image_page=None)

    monkeypatch.setattr(main_module, "_resolve_llm", fake_resolve_llm)
    monkeypatch.setattr(main_module, "AskUseCase", FakeUseCase)
    resp = client.post("/ask", json={
        "question": "hola",
        "llm": {"provider": "custom", "base_url": CUSTOM_URL,
                "model": "llama3.1", "api_key": "", "auto_model": False}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "respuesta custom"
    assert body["model_used"] == "llama3.1"
    assert seen == {"provider": "custom", "model": "llama3.1", "api_key": "",
                    "base_url": CUSTOM_URL}


def test_ask_image_custom_sin_token_no_exige_token(monkeypatch):
    def fake_resolve_llm(provider, model, api_key, base_url, auto_model,
                         models_info_json, need_vision):
        return (object(), "llava", False)

    class FakeVision:
        def __init__(self, *args, **kwargs):
            pass

    class FakeImageUseCase:
        def __init__(self, *args):
            pass

        def ask_image(self, *args, **kwargs):
            return SimpleNamespace(
                description="un motor",
                answer=SimpleNamespace(text="respuesta", sources=[],
                                       image_id=None, image_page=None))

    monkeypatch.setattr(main_module, "_resolve_llm", fake_resolve_llm)
    monkeypatch.setattr(main_module, "OpenAICompatibleVision", FakeVision)
    monkeypatch.setattr(main_module, "ImageAskUseCase", FakeImageUseCase)
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", b"bytes-falsos", "image/png")},
        data={"provider": "custom", "base_url": CUSTOM_URL,
              "model": "llava", "api_key": "", "question": "que es esto"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "respuesta"


def test_ask_image_sin_token_ni_url_sigue_400():
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", b"bytes-falsos", "image/png")},
        data={"provider": "openai", "model": "gpt-4o-mini",
              "api_key": "", "question": "que es esto"})
    assert resp.status_code == 400
    assert "token" in resp.json()["detail"]
