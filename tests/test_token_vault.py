"""Tests de la boveda de tokens: guardado cifrado en el servidor,
predeterminada compartida, borrado y que ninguna respuesta devuelve
el valor del token.

Todo con tokens falsos y httpx mockeado: ninguna prueba llama a un
proveedor real ni usa una clave real.
"""
import json
import os
import subprocess
import sys

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import app.main as main
from app.adapters.token_vault import (
    TokenVault,
    TokenVaultUnavailable,
    decrypt_config,
    encrypt_config,
    public_config,
    validate_browser_id,
)

FAKE_TOKEN = "sk-test-fake-token-12345"
FAKE_DEFAULT = "sk-test-fake-default-67890"
BROWSER_ID = "browser-test-abc-123"

client = TestClient(main.app)


def _key():
    return Fernet.generate_key().decode()


def _vault(tmp_path, key=None):
    return TokenVault(store_dir=str(tmp_path / "tokens"),
                      encryption_key=key if key is not None else _key(),
                      default_path=str(tmp_path / "default_token.enc"))


def _config(token=FAKE_TOKEN):
    return {"api_key": token, "provider": "openai", "base_url": None,
            "model": "gpt-4o-mini", "auto_model": False,
            "models_info": [{"id": "gpt-4o-mini", "prompt_price": None,
                             "modality": None}]}


def _use_vault(monkeypatch, vault):
    monkeypatch.setattr(main.container, "token_vault", vault)


# --- Unidad: boveda ---

def test_guardar_y_recuperar_por_identificador(tmp_path):
    vault = _vault(tmp_path)
    vault.save_visitor(BROWSER_ID, _config())
    loaded = vault.load_visitor(BROWSER_ID)
    assert loaded["api_key"] == FAKE_TOKEN
    assert loaded["provider"] == "openai"
    assert loaded["model"] == "gpt-4o-mini"
    assert vault.load_visitor("otro-navegador-999") is None


def test_archivo_guardado_no_contiene_el_token_en_plano(tmp_path):
    vault = _vault(tmp_path)
    vault.save_visitor(BROWSER_ID, _config())
    blob = open(vault._visitor_path(BROWSER_ID), "rb").read()
    assert FAKE_TOKEN.encode() not in blob
    assert decrypt_config(blob, vault.encryption_key)["api_key"] == FAKE_TOKEN


def test_borrado_elimina_y_vuelve_al_predeterminado(tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    vault.save_visitor(BROWSER_ID, _config())
    assert vault.resolve(BROWSER_ID) == (_config(), "visitor")
    assert vault.delete_visitor(BROWSER_ID) is True
    assert vault.load_visitor(BROWSER_ID) is None
    config, source = vault.resolve(BROWSER_ID)
    assert config["api_key"] == FAKE_DEFAULT
    assert source == "default"
    assert vault.delete_visitor(BROWSER_ID) is False


def test_predeterminado_sin_token_propio_y_sin_nada(tmp_path):
    vault = _vault(tmp_path)
    config, source = vault.resolve(BROWSER_ID)
    assert config is None and source == "demo"
    vault.save_default(_config(FAKE_DEFAULT))
    config, source = vault.resolve(BROWSER_ID)
    assert config["api_key"] == FAKE_DEFAULT and source == "default"
    config, source = vault.resolve(None)
    assert config["api_key"] == FAKE_DEFAULT and source == "default"


def test_token_propio_tiene_preferencia_sobre_predeterminado(tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    vault.save_visitor(BROWSER_ID, _config())
    config, source = vault.resolve(BROWSER_ID)
    assert config["api_key"] == FAKE_TOKEN and source == "visitor"


def test_sin_clave_de_cifrado_falla_claro_y_nunca_en_plano(tmp_path):
    vault = _vault(tmp_path, key="")
    assert vault.configured is False
    with pytest.raises(TokenVaultUnavailable):
        vault.save_visitor(BROWSER_ID, _config())
    # Sin archivos, resolver no necesita la clave: demo sin error.
    config, source = vault.resolve(BROWSER_ID)
    assert config is None and source == "demo"
    # Con archivo existente pero sin clave, usarlo falla claro.
    good = _vault(tmp_path)
    good.save_visitor(BROWSER_ID, _config())
    broken = _vault(tmp_path, key="")
    with pytest.raises(TokenVaultUnavailable):
        broken.load_visitor(BROWSER_ID)
    # El archivo si existe (lo escribio la boveda con clave): el fallo
    # es por falta de clave al descifrar, no por ausencia.
    assert list((tmp_path / "tokens").glob("*.token")) != []


def test_clave_de_cifrado_invalida_no_configurada(tmp_path):
    vault = _vault(tmp_path, key="no-es-una-clave-fernet")
    assert vault.configured is False
    with pytest.raises(TokenVaultUnavailable):
        vault.save_visitor(BROWSER_ID, _config())


def test_identificador_invalido_rechazado(tmp_path):
    vault = _vault(tmp_path)
    for bad in ["../escape", "a/b", "corto", "", "con espacios", "x" * 200]:
        with pytest.raises(ValueError):
            vault.save_visitor(bad, _config())
        with pytest.raises(ValueError):
            vault.load_visitor(bad)
    assert validate_browser_id(BROWSER_ID) == BROWSER_ID


def test_config_publica_sin_secreto():
    pub = public_config(_config())
    assert "api_key" not in pub
    assert pub["provider"] == "openai"
    assert pub["model"] == "gpt-4o-mini"
    assert public_config(None) is None


def test_encrypt_decrypt_roundtrip_sin_vault():
    key = _key()
    blob = encrypt_config(_config(), key)
    assert FAKE_TOKEN.encode() not in blob
    assert decrypt_config(blob, key)["api_key"] == FAKE_TOKEN
    with pytest.raises(TokenVaultUnavailable):
        encrypt_config(_config(), "")
    with pytest.raises(TokenVaultUnavailable):
        decrypt_config(blob, _key())  # otra clave: no descifra


# --- API: guardar, estado, borrar, detectar, preguntar ---

def test_api_guardar_y_estado_sin_devolver_el_token(monkeypatch, tmp_path):
    _use_vault(monkeypatch, _vault(tmp_path))
    resp = client.post("/llm/token", json={
        "browser_id": BROWSER_ID, "api_key": FAKE_TOKEN,
        "provider": "openai", "model": "gpt-4o-mini", "auto_model": False})
    assert resp.status_code == 200, resp.text
    assert FAKE_TOKEN not in resp.text
    body = resp.json()
    assert body["has_visitor_token"] is True
    assert body["active"] == "visitor"
    status = client.get("/llm/token/status",
                        params={"browser_id": BROWSER_ID}).json()
    assert status["active"] == "visitor"
    assert status["provider"] == "openai"
    assert status["model"] == "gpt-4o-mini"
    assert FAKE_TOKEN not in json.dumps(status)


def test_api_borrado_vuelve_al_predeterminado(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    vault.save_visitor(BROWSER_ID, _config())
    _use_vault(monkeypatch, vault)
    resp = client.delete("/llm/token", params={"browser_id": BROWSER_ID})
    assert resp.status_code == 200
    assert resp.json()["active"] == "default"
    assert resp.json()["has_visitor_token"] is False
    assert FAKE_TOKEN not in resp.text
    assert FAKE_DEFAULT not in resp.text


def test_api_borrado_sin_predeterminado_vuelve_a_demo(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_visitor(BROWSER_ID, _config())
    _use_vault(monkeypatch, vault)
    resp = client.delete("/llm/token", params={"browser_id": BROWSER_ID})
    assert resp.json()["active"] == "demo"


def test_api_estado_sin_identificador_muestra_predeterminado(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    _use_vault(monkeypatch, vault)
    body = client.get("/llm/token/status").json()
    assert body["active"] == "default"
    assert body["has_default"] is True
    assert FAKE_DEFAULT not in json.dumps(body)


def test_api_guardar_sin_clave_falla_503_claro(monkeypatch, tmp_path):
    _use_vault(monkeypatch, _vault(tmp_path, key=""))
    resp = client.post("/llm/token", json={
        "browser_id": BROWSER_ID, "api_key": FAKE_TOKEN})
    assert resp.status_code == 503
    assert "TOKEN_ENCRYPTION_KEY" in resp.text
    assert FAKE_TOKEN not in resp.text


def test_api_guardar_identificador_invalido_400(monkeypatch, tmp_path):
    _use_vault(monkeypatch, _vault(tmp_path))
    resp = client.post("/llm/token", json={
        "browser_id": "../escape", "api_key": FAKE_TOKEN})
    assert resp.status_code == 400


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_api_detect_por_identificador_sin_devolver_el_token(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_visitor(BROWSER_ID, _config())
    _use_vault(monkeypatch, vault)

    def fake_get(url, headers=None, timeout=None):
        assert "api.openai.com" in url
        assert FAKE_TOKEN not in url
        return FakeResp(200, {"data": [{"id": "gpt-4o-mini"},
                                       {"id": "gpt-4o"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    resp = client.post("/llm/detect", json={"browser_id": BROWSER_ID})
    assert resp.status_code == 200
    assert resp.json()["detected"] is True
    assert resp.json()["provider"] == "openai"
    assert "gpt-4o-mini" in resp.json()["models"]
    assert FAKE_TOKEN not in resp.text


def test_api_detect_por_identificador_con_predeterminado(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    _use_vault(monkeypatch, vault)

    def fake_get(url, headers=None, timeout=None):
        return FakeResp(200, {"data": [{"id": "gpt-4o-mini"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    resp = client.post("/llm/detect", json={"browser_id": BROWSER_ID})
    assert resp.json()["detected"] is True
    assert FAKE_DEFAULT not in resp.text


class _FakeAnswer:
    def __init__(self):
        self.text = "respuesta de prueba"
        self.sources = []
        self.image_id = None
        self.image_page = None


class _FakeAskUseCase:
    def __init__(self, *args, **kwargs):
        pass

    def ask(self, question, top_k=5):
        return _FakeAnswer()


def test_api_ask_con_token_guardado_usa_visitante(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_default(_config(FAKE_DEFAULT))
    vault.save_visitor(BROWSER_ID, _config())
    _use_vault(monkeypatch, vault)
    monkeypatch.setattr(main, "AskUseCase", _FakeAskUseCase)
    resp = client.post("/ask", json={
        "question": "hola",
        "browser_id": BROWSER_ID,
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "",
                "auto_model": False}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["token_source"] == "visitor"
    assert FAKE_TOKEN not in resp.text
    assert FAKE_DEFAULT not in resp.text


def test_api_ask_sin_token_propio_usa_predeterminado(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_default({"api_key": FAKE_DEFAULT, "provider": "openai",
                        "base_url": None, "model": "gpt-4o-mini",
                        "auto_model": False, "models_info": None})
    _use_vault(monkeypatch, vault)
    monkeypatch.setattr(main, "AskUseCase", _FakeAskUseCase)
    resp = client.post("/ask", json={"question": "hola"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["token_source"] == "default"
    assert FAKE_DEFAULT not in resp.text


def test_api_ask_predeterminado_auto_sin_lista_mantiene_400(monkeypatch, tmp_path):
    # El automatico exige lista de modelos (regla existente de
    # select_model): una predeterminada auto sin lista no inventa un
    # modelo, devuelve el 400 habitual. Por eso el script escribe un
    # modelo fijo por defecto (gpt-4o-mini).
    vault = _vault(tmp_path)
    vault.save_default({"api_key": FAKE_DEFAULT, "provider": "openai",
                        "base_url": None, "model": None, "auto_model": True,
                        "models_info": None})
    _use_vault(monkeypatch, vault)
    resp = client.post("/ask", json={"question": "hola"})
    assert resp.status_code == 400
    assert FAKE_DEFAULT not in resp.text


def test_api_ask_token_explicito_gana_a_guardado(monkeypatch, tmp_path):
    vault = _vault(tmp_path)
    vault.save_visitor(BROWSER_ID, _config())
    _use_vault(monkeypatch, vault)
    monkeypatch.setattr(main, "AskUseCase", _FakeAskUseCase)
    resp = client.post("/ask", json={
        "question": "hola",
        "browser_id": BROWSER_ID,
        "llm": {"provider": "openai", "model": "gpt-4o-mini",
                "api_key": "sk-test-explicito-999", "auto_model": False}})
    assert resp.json()["token_source"] == "request"
    assert "sk-test-explicito-999" not in resp.text


# --- Script de la predeterminada ---

def _run_script(tmp_path, env_extra, stdin_text):
    script = os.path.join(os.getcwd(), "scripts", "set_default_token.py")
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, script, "--provider", "openai"],
        input=stdin_text, capture_output=True, text=True, timeout=30,
        env=env, cwd=str(tmp_path))


def test_script_guarda_predeterminada_sin_imprimir_el_token(tmp_path):
    key = _key()
    target = tmp_path / "default_token.enc"
    proc = _run_script(tmp_path, {
        "TOKEN_ENCRYPTION_KEY": key,
        "DEFAULT_TOKEN_PATH": str(target)}, FAKE_TOKEN + "\n")
    assert proc.returncode == 0, proc.stderr
    assert target.exists()
    assert FAKE_TOKEN not in proc.stdout
    assert FAKE_TOKEN not in proc.stderr
    assert decrypt_config(target.read_bytes(), key)["api_key"] == FAKE_TOKEN
    assert decrypt_config(target.read_bytes(), key)["model"] == "gpt-4o-mini"
    mode = oct(os.stat(target).st_mode & 0o777)
    assert mode == "0o600"


def test_script_sin_clave_no_guarda_nada(tmp_path):
    target = tmp_path / "default_token.enc"
    proc = _run_script(tmp_path, {
        "TOKEN_ENCRYPTION_KEY": "",
        "DEFAULT_TOKEN_PATH": str(target)}, FAKE_TOKEN + "\n")
    assert proc.returncode != 0
    assert not target.exists()
    assert FAKE_TOKEN not in proc.stdout
