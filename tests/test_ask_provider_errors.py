"""Regresion del 500 en /ask con token (captura en vivo: "Error del LLM:
Internal Server Error").

Causa: /ask no capturaba errores del proveedor (401, 404, timeout,
conexion) ni respuestas con forma inesperada (content null, sin choices),
y FastAPI devolvia un 500 con texto plano "Internal Server Error".
Ahora esos caminos devuelven 502 con un mensaje limpio.

Todo con httpx mockeado: ninguna prueba llama a un proveedor real ni
usa una clave real. Se ingiere un PDF pequeno para que la recuperacion
supere el umbral y el LLM llegue a llamarse (sin contexto, el caso de
uso responde la negativa sin llamar al proveedor y el bug no se ve).
"""
import base64
import json
import time

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

QUESTION = "Cual es el par de apriete de los tornillos de culata?"
OVERRIDE = {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-falsa"}
AUTO = {"provider": "openai", "model": None, "api_key": "sk-falsa",
        "auto_model": True,
        "models_info": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}


def _ensure_doc():
    """Ingiere el PDF de prueba (la segunda vez lo reutiliza por dedupe:
    ya completado, sin duplicar chunks)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Mantenimiento")
    page.insert_text((72, 100), "El par de apriete de los tornillos de culata es de 24 Nm.")
    path = "/tmp/repro-ask-errors.pdf"
    doc.save(path)
    doc.close()
    with open(path, "rb") as f:
        payload = f.read()
    resp = client.post("/ingest", files={"file": ("repro.pdf", payload, "application/pdf")})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job


class FakeResp:
    def __init__(self, status, payload=None, json_error=None):
        self.status_code = status
        self._payload = payload or {}
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("mock error", request=request, response=response)


def _ok(url, **kwargs):
    return FakeResp(200, {"choices": [{"message": {"content": "El par es 24 Nm."}}]})


def _no_call(url, **kwargs):
    raise AssertionError(f"no debe llamar al proveedor: {url}")


def test_demo_sin_override_ok():
    _ensure_doc()
    resp = client.post("/ask", json={"question": QUESTION})
    assert resp.status_code == 200
    assert "24 Nm" in resp.json()["answer"]
    assert resp.json()["model_used"] == "demo"


def test_override_ok(monkeypatch):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", _ok)
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "gpt-4o-mini"
    assert resp.json()["model_auto"] is False


def test_auto_con_datos_ok(monkeypatch):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", _ok)
    resp = client.post("/ask", json={"question": QUESTION, "llm": AUTO})
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "gpt-4o-mini"
    assert resp.json()["model_auto"] is True


def test_auto_sin_datos_400():
    _ensure_doc()
    resp = client.post("/ask", json={"question": QUESTION,
                                     "llm": {**AUTO, "models_info": None}})
    assert resp.status_code == 400
    assert "modelo" in resp.json()["detail"].lower()


def test_modelo_ausente_400():
    _ensure_doc()
    resp = client.post("/ask", json={"question": QUESTION,
                                     "llm": {"provider": "openai", "model": None,
                                            "api_key": "sk-falsa", "auto_model": False}})
    assert resp.status_code == 400


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_proveedor_http_a_502_no_500(monkeypatch, status):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(status, {"error": "mock"}))
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 502, resp.text
    assert "token" in resp.json()["detail"].lower()


def test_proveedor_conexion_a_502_no_500(monkeypatch):
    _ensure_doc()

    def _fail(url, **kwargs):
        raise httpx.ConnectError("mock conexion", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fail)
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 502, resp.text


def test_proveedor_content_null_a_502_no_500(monkeypatch):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(
        200, {"choices": [{"message": {"content": None}}]}))
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 502, resp.text
    # El endpoint usa un mensaje generico para errores de forma (el
    # detalle "respuesta vacia" queda en el log del adaptador).
    assert "inesperada" in resp.json()["detail"].lower()


def test_proveedor_sin_choices_a_502_no_500(monkeypatch):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(200, {"choices": []}))
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 502, resp.text
    assert "inesperada" in resp.json()["detail"].lower()


def test_proveedor_cuerpo_no_json_a_502_no_500(monkeypatch):
    _ensure_doc()
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(
        200, json_error=json.JSONDecodeError("Expecting value", "no-json", 0)))
    resp = client.post("/ask", json={"question": QUESTION, "llm": OVERRIDE})
    assert resp.status_code == 502, resp.text


def test_audio_transcripcion_vacia_a_502_no_500(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(200, {"text": ""}))
    resp = client.post(
        "/ask-audio",
        files={"audio": ("nota.mp3", b"\x00\x01", "audio/mpeg")},
        data={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-falsa",
              "auto_model": "false", "transcription_model": "whisper-1",
              "question": "", "top_k": "5"})
    assert resp.status_code == 502, resp.text


def test_imagen_descripcion_vacia_a_502_no_500(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **k: FakeResp(
        200, {"choices": [{"message": {"content": ""}}]}))
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", png_1x1, "image/png")},
        data={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-falsa",
              "auto_model": "false", "question": "", "top_k": "5"})
    assert resp.status_code == 502, resp.text
