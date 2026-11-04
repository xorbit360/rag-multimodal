"""Tests de deteccion de proveedor LLM y override por peticion.

Todo con httpx mockeado: ninguna prueba llama a un proveedor real ni
usa una clave real. La clave nunca debe aparecer en una respuesta.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app.adapters.llm import build_override_llm, detect_provider
from app.main import app

client = TestClient(app)


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_detect_openai(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert "api.openai.com" in url
        return FakeResp(200, {"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    provider, models = detect_provider("sk-test")
    assert provider == "openai"
    assert "gpt-4o-mini" in models


def test_detect_openrouter_tras_401_en_openai(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if "api.openai.com" in url:
            return FakeResp(401, {})
        if "openrouter.ai/api/v1/auth/key" in url:
            return FakeResp(200, {"data": {"label": "demo"}})
        if "openrouter.ai" in url:
            return FakeResp(200, {"data": [{"id": "openai/gpt-4o-mini"}]})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", fake_get)
    provider, models = detect_provider("sk-or-test")
    assert provider == "openrouter"
    assert models == ["openai/gpt-4o-mini"]


def test_detect_ninguno(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResp(401, {})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert detect_provider("sk-mala") == (None, [])


def test_detect_custom_base_url(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert url.startswith("http://localhost:11434/v1/models")
        return FakeResp(200, {"data": [{"id": "llama3"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    provider, models = detect_provider("ollama-sin-clave",
                                       "http://localhost:11434/v1")
    assert provider == "custom"
    assert models == ["llama3"]


def test_detect_endpoint_nunca_devuelve_la_clave(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return FakeResp(401, {})

    monkeypatch.setattr(httpx, "get", fake_get)
    resp = client.post("/llm/detect", json={"api_key": "sk-secreto-123"})
    assert resp.status_code == 200
    assert resp.json()["detected"] is False
    assert "sk-secreto-123" not in resp.text


def test_build_override_llm():
    llm = build_override_llm("openai", "gpt-4o-mini", "sk-x")
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.model == "gpt-4o-mini"
    llm2 = build_override_llm("custom", "llama3", "k",
                              "http://localhost:11434/v1")
    assert llm2.base_url == "http://localhost:11434/v1"
    with pytest.raises(ValueError):
        build_override_llm("custom", "m", "k")
    with pytest.raises(ValueError):
        build_override_llm("proveedor-raro", "m", "k")


def test_ask_acepta_campo_llm_sin_romper():
    # Sin contexto, responde la negativa sin llamar al LLM: prueba que el
    # campo llm se acepta y no rompe el flujo.
    resp = client.post("/ask", json={
        "question": "zxqvmlp inexistente",
        "llm": {"provider": "openai", "model": "gpt-4o-mini",
                "api_key": "sk-falsa"}})
    assert resp.status_code == 200
    assert "No tengo información suficiente" in resp.json()["answer"]
