"""Tests de audio, imagen, dedupe, modelo automatico y cableado web.

Todo con httpx mockeado: ninguna prueba llama a un proveedor real ni
usa una clave real. La clave nunca debe aparecer en una respuesta.
"""
import base64
import time

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from app.adapters.multimodal import (
    OpenAICompatibleTranscription, OpenAICompatibleVision,
)
from app.domain.ports import VisionNotSupportedError
from app.main import app, container

client = TestClient(app)

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://test")
            raise httpx.HTTPStatusError(
                "error", request=req,
                response=httpx.Response(self.status_code, request=req))


def _pdf_bytes(tmp_path, text, name="doc.pdf"):
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def _ingest_and_wait(data, name):
    resp = client.post("/ingest", files={"file": (name, data, "application/pdf")})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job
    return job


def _chat_mock(transcription=None, description=None):
    """Mock de httpx.post: enruta transcripcion, vision y chat segun la URL
    y el cuerpo. Devuelve respuestas fijas, sin llamar a ningun proveedor."""
    def fake_post(url, headers=None, json=None, files=None, data=None,
                  timeout=None):
        if "/audio/transcriptions" in url:
            return FakeResp(200, {"text": transcription or ""})
        if "/chat/completions" in url:
            body = json or {}
            text_parts = []
            for msg in body.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            text_parts.append("IMAGE_URL")
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                else:
                    text_parts.append(content)
            joined = " ".join(text_parts)
            if "IMAGE_URL" in joined:
                return FakeResp(200, {"choices": [{"message": {
                    "content": description or ""}}]})
            return FakeResp(200, {"choices": [{"message": {
                "content": "Respuesta simulada basada en el contexto."}}]})
        raise AssertionError(f"URL no mockeada: {url}")
    return fake_post


def test_ask_audio_transcribe_y_responde(monkeypatch, tmp_path):
    data = _pdf_bytes(tmp_path, "El par de apriete de los tornillos de culata es de 24 Nm.",
                      "audio-doc.pdf")
    _ingest_and_wait(data, "audio-doc.pdf")
    monkeypatch.setattr(
        httpx, "post",
        _chat_mock(transcription="Cual es el par de apriete de los tornillos de culata?"))
    resp = client.post(
        "/ask-audio",
        files={"audio": ("pregunta.mp3", b"audio-falso", "audio/mpeg")},
        data={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-falsa",
              "auto_model": "false", "question": "", "top_k": "5"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["transcription"] == "Cual es el par de apriete de los tornillos de culata?"
    assert body["sources"], "la transcripcion debe alimentar la recuperacion"
    assert body["model_used"] == "gpt-4o-mini"
    assert body["model_auto"] is False
    assert body["transcription_model"] == "whisper-1"
    assert "sk-falsa" not in resp.text


def test_ask_audio_sin_token_400():
    resp = client.post(
        "/ask-audio",
        files={"audio": ("pregunta.mp3", b"audio-falso", "audio/mpeg")},
        data={"provider": "openai", "model": "gpt-4o-mini", "api_key": "",
              "auto_model": "false"})
    assert resp.status_code == 400
    assert "token" in resp.json()["detail"].lower()


def test_ask_image_busca_referencias_sin_indexar(monkeypatch, tmp_path):
    data = _pdf_bytes(tmp_path, "El diagrama muestra el piston, la biela y el ciguenal del motor.",
                      "imagen-doc.pdf")
    _ingest_and_wait(data, "imagen-doc.pdf")
    before_chunks = len(container.store.all_chunks())
    before_docs = len(client.get("/documents").json())
    monkeypatch.setattr(
        httpx, "post",
        _chat_mock(description="diagrama de un motor con piston, biela y ciguenal"))
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", TINY_PNG, "image/png")},
        data={"provider": "openai", "model": "gpt-4o", "api_key": "sk-falsa",
              "auto_model": "false", "question": "", "top_k": "5"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "piston" in body["description"]
    assert body["sources"], "la descripcion debe buscar referencias en el RAG"
    assert body["model_used"] == "gpt-4o"
    assert "sk-falsa" not in resp.text
    # La imagen enviada NO se indexa: ni chunks nuevos ni documentos nuevos.
    assert len(container.store.all_chunks()) == before_chunks
    assert len(client.get("/documents").json()) == before_docs


def test_ask_image_modelo_sin_vision_400(monkeypatch):
    def fake_post(url, headers=None, json=None, files=None, data=None,
                  timeout=None):
        return FakeResp(400, {"error": {"message": "image not supported"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", TINY_PNG, "image/png")},
        data={"provider": "openai", "model": "modelo-sin-vision",
              "api_key": "sk-falsa", "auto_model": "false"})
    assert resp.status_code == 400
    assert "visi" in resp.json()["detail"].lower()
    assert "description" not in resp.json(), "sin descripcion inventada"


def test_ask_image_sin_token_400():
    resp = client.post(
        "/ask-image",
        files={"image": ("foto.png", TINY_PNG, "image/png")},
        data={"provider": "openai", "model": "gpt-4o", "api_key": "",
              "auto_model": "false"})
    assert resp.status_code == 400


def test_ingest_dedupe_mismo_pdf(tmp_path):
    data = _pdf_bytes(tmp_path, "Contenido unico para dedupe 987654321.",
                      "dedupe.pdf")
    first = _ingest_and_wait(data, "dedupe.pdf")
    assert first["duplicate"] is False
    chunks_after_first = len(container.store.all_chunks())
    second = _ingest_and_wait(data, "dedupe.pdf")
    assert second["duplicate"] is True
    assert "ya indexado" in second["message"]
    assert second["chunks_count"] == first["chunks_count"]
    assert second["document_id"] == first["document_id"]
    assert len(container.store.all_chunks()) == chunks_after_first
    docs = [d for d in client.get("/documents").json()
            if d["document_name"] == "dedupe.pdf"]
    assert len(docs) == 1, "el documento no debe aparecer duplicado"


def test_ask_auto_elige_mas_barato_y_lo_reporta(monkeypatch, tmp_path):
    data = _pdf_bytes(tmp_path, "El par de apriete de los tornillos de culata es de 24 Nm.",
                      "auto-doc.pdf")
    _ingest_and_wait(data, "auto-doc.pdf")
    seen = {}

    def fake_post(url, headers=None, json=None, files=None, data=None,
                  timeout=None):
        if "/chat/completions" in url:
            seen["model"] = (json or {}).get("model")
            return FakeResp(200, {"choices": [{"message": {
                "content": "Respuesta simulada."}}]})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "post", fake_post)
    resp = client.post("/ask", json={
        "question": "Cual es el par de apriete de los tornillos de culata?",
        "llm": {"provider": "openai", "api_key": "sk-falsa",
                "auto_model": True,
                "models_info": [
                    {"id": "gpt-4o", "prompt_price": 0.005},
                    {"id": "gpt-4o-mini", "prompt_price": 0.00015}]}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_used"] == "gpt-4o-mini"
    assert body["model_auto"] is True
    assert seen["model"] == "gpt-4o-mini"
    assert "sk-falsa" not in resp.text


def test_ask_auto_vision_sin_vision_400(monkeypatch):
    resp = client.post("/ask-image",
                       files={"image": ("foto.png", TINY_PNG, "image/png")},
                       data={"provider": "openai", "api_key": "sk-falsa",
                             "auto_model": "true",
                             "models_info_json": (
                                 '[{"id": "llama-3-8b", "prompt_price": 0.0001}]')})
    assert resp.status_code == 400
    assert "visi" in resp.json()["detail"].lower()


def test_detect_devuelve_precios_y_modalidad(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if "openrouter.ai/api/v1/auth/key" in url:
            return FakeResp(200, {"data": {"label": "demo"}})
        if "openrouter.ai" in url:
            return FakeResp(200, {"data": [
                {"id": "openai/gpt-4o-mini",
                 "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
                 "architecture": {"modality": "text->text"}},
                {"id": "openai/gpt-4o",
                 "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                 "architecture": {"modality": "text+image->text"}}]})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", fake_get)
    resp = client.post("/llm/detect", json={"api_key": "sk-or-test"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected"] is True
    assert body["provider"] == "openrouter"
    info = {m["id"]: m for m in body["models_info"]}
    assert info["openai/gpt-4o-mini"]["prompt_price"] == 0.00000015
    assert info["openai/gpt-4o"]["modality"] == "text+image->text"


def test_adaptadores_multimodales(monkeypatch):
    def fake_post(url, headers=None, json=None, files=None, data=None,
                  timeout=None):
        if "/audio/transcriptions" in url:
            assert data["model"] == "whisper-1"
            return FakeResp(200, {"text": "hola mundo"})
        if "/chat/completions" in url:
            return FakeResp(200, {"choices": [{"message": {
                "content": "un motor con piston"}}]})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "post", fake_post)
    trans = OpenAICompatibleTranscription(api_key="k",
                                          base_url="https://api.openai.com/v1")
    assert trans.transcribe(b"audio", "a.mp3", "whisper-1") == "hola mundo"
    vision = OpenAICompatibleVision(api_key="k",
                                    base_url="https://api.openai.com/v1",
                                    model="gpt-4o")
    assert vision.describe(TINY_PNG, "image/png") == "un motor con piston"


def test_vision_400_es_error_claro_no_invento(monkeypatch):
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: FakeResp(400, {}))
    vision = OpenAICompatibleVision(api_key="k",
                                    base_url="https://api.openai.com/v1",
                                    model="sin-vision")
    with pytest.raises(VisionNotSupportedError):
        vision.describe(TINY_PNG, "image/png")


def test_frontend_cableado():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "app.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    # Chat con audio/imagen y endpoints nuevos.
    assert "/ask-audio" in src
    assert "/ask-image" in src
    assert "accept_audio" in src
    # El chat NO acepta PDF: solo la barra lateral (linea de tipos del chat).
    for line in src.splitlines():
        if line.strip().startswith("CHAT_FILE_TYPES"):
            assert "pdf" not in line
    assert 'type=["pdf"]' in src  # uploader de la barra lateral
    # Token guardable/borrable solo en el navegador.
    assert "Guardar token en este navegador" in src
    assert "Borrar token guardado" in src
    assert "localStorage" in src
    # Modelo automatico predeterminado y modelo usado visible.
    assert "Automático (el más económico)" in src
    assert "Modelo usado" in src
    # Responsivo movil.
    assert "@media" in src
    assert "initial_sidebar_state" in src
    # Aviso de duplicado en la ingesta.
    assert "ya estaba indexado" in src
