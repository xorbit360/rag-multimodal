"""Guardado de tokens LLM en el servidor, cifrado en reposo (Fernet).

- Un visitante guarda su token: queda cifrado en un archivo asociado a
  un identificador aleatorio de su navegador (el demo no tiene login).
- El servidor puede tener UN token predeterminado compartido, en otro
  archivo cifrado que SOLO escribe el script del dueno (nunca la API
  y nunca la web publica).
- El navegador guarda solo el identificador, nunca el token.
- Sin clave de cifrado configurada, guardar y usar un token guardado
  fallan con un error claro: nunca se guarda ni se usa nada en plano.
- El valor del token nunca se registra en logs ni se devuelve en
  ninguna respuesta: solo presencia y origen (visitante, predeterminado
  o demo). Los archivos se escriben con permisos 0600 y reemplazo
  atomico.
"""
from __future__ import annotations

import json
import os
import re

from cryptography.fernet import Fernet, InvalidToken

# Identificador de navegador: solo caracteres seguros para usarlo como
# nombre de archivo (nada de rutas, puntos ni separadores).
BROWSER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


class TokenVaultUnavailable(Exception):
    """El guardado no esta usable: falta la clave de cifrado, es
    invalida o un archivo cifrado no se pudo descifrar."""


def validate_browser_id(browser_id: str) -> str:
    """Valida el identificador o lanza ValueError (el llamador lo
    traduce a 400)."""
    if not browser_id or not BROWSER_ID_RE.match(browser_id):
        raise ValueError("identificador de navegador invalido")
    return browser_id


def encrypt_config(config: dict, encryption_key: str) -> bytes:
    """Cifra un dict de configuracion de token (incluye la clave) y
    devuelve los bytes a guardar. Usado tambien por el script del
    dueno para la predeterminada."""
    if not encryption_key:
        raise TokenVaultUnavailable(
            "falta la clave de cifrado del servidor (TOKEN_ENCRYPTION_KEY)")
    try:
        fernet = Fernet(encryption_key.encode())
    except Exception as exc:
        raise TokenVaultUnavailable(
            "la clave de cifrado del servidor es invalida") from exc
    payload = json.dumps(config, ensure_ascii=False).encode()
    return fernet.encrypt(payload)


def decrypt_config(blob: bytes, encryption_key: str) -> dict:
    """Descifra un blob guardado por encrypt_config."""
    if not encryption_key:
        raise TokenVaultUnavailable(
            "falta la clave de cifrado del servidor (TOKEN_ENCRYPTION_KEY)")
    try:
        fernet = Fernet(encryption_key.encode())
        raw = fernet.decrypt(blob)
    except InvalidToken as exc:
        raise TokenVaultUnavailable(
            "no se pudo descifrar el token guardado (archivo danado o "
            "clave distinta)") from exc
    except Exception as exc:
        raise TokenVaultUnavailable(
            "la clave de cifrado del servidor es invalida") from exc
    try:
        data = json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError) as exc:
        raise TokenVaultUnavailable(
            "el token guardado tiene un formato inesperado") from exc
    if not isinstance(data, dict) or not data.get("api_key"):
        raise TokenVaultUnavailable(
            "el token guardado tiene un formato inesperado")
    return data


def public_config(config: dict | None) -> dict | None:
    """Configuracion sin el valor secreto: lo que la API puede devolver
    para reconstruir la UI (proveedor, modelo, seleccion)."""
    if not config:
        return None
    return {
        "provider": config.get("provider", "openai"),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "auto_model": bool(config.get("auto_model", False)),
        "models_info": config.get("models_info"),
    }


class TokenVault:
    """Almacen cifrado en disco para tokens de visitante y la
    predeterminada compartida. La construccion nunca falla por falta
    de clave: el fallo claro ocurre al guardar o usar."""

    def __init__(self, store_dir: str, encryption_key: str,
                 default_path: str):
        self.store_dir = store_dir
        self.encryption_key = encryption_key
        self.default_path = default_path

    @property
    def configured(self) -> bool:
        """True si hay una clave de cifrado valida (sin probarla
        contra ningun archivo)."""
        if not self.encryption_key:
            return False
        try:
            Fernet(self.encryption_key.encode())
        except Exception:
            return False
        return True

    def _visitor_path(self, browser_id: str) -> str:
        validate_browser_id(browser_id)
        return os.path.join(self.store_dir, f"{browser_id}.token")

    def _write_blob(self, path: str, blob: bytes) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "wb") as out:
            out.write(blob)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def save_visitor(self, browser_id: str, config: dict) -> None:
        """Guarda (o reemplaza) el token del visitante, cifrado."""
        if not config.get("api_key"):
            raise ValueError("falta el token a guardar")
        blob = encrypt_config(config, self.encryption_key)
        self._write_blob(self._visitor_path(browser_id), blob)

    def load_visitor(self, browser_id: str) -> dict | None:
        """Token del visitante o None si no guardo ninguno. Lanza
        TokenVaultUnavailable si el archivo existe pero no se puede
        descifrar (clave ausente, invalida o archivo danado)."""
        path = self._visitor_path(browser_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            return decrypt_config(handle.read(), self.encryption_key)

    def delete_visitor(self, browser_id: str) -> bool:
        """Borra el token del visitante. True si existia."""
        path = self._visitor_path(browser_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def visitor_exists(self, browser_id: str) -> bool:
        """Solo existencia del archivo, sin descifrar."""
        return os.path.exists(self._visitor_path(browser_id))

    def load_default(self) -> dict | None:
        """Token predeterminado compartido o None si el dueno no lo
        establecio. Lanza TokenVaultUnavailable si el archivo existe
        pero no se puede descifrar."""
        if not os.path.exists(self.default_path):
            return None
        with open(self.default_path, "rb") as handle:
            return decrypt_config(handle.read(), self.encryption_key)

    def has_default(self) -> bool:
        """Solo existencia del archivo, sin descifrar."""
        return os.path.exists(self.default_path)

    def save_default(self, config: dict) -> None:
        """Escribe la predeterminada cifrada. Solo la usa el script
        del dueno, nunca la API ni la web."""
        if not config.get("api_key"):
            raise ValueError("falta el token a guardar")
        blob = encrypt_config(config, self.encryption_key)
        self._write_blob(self.default_path, blob)

    def resolve(self, browser_id: str | None) -> tuple[dict | None, str]:
        """Token efectivo para un visitante: el suyo si lo guardo,
        si no el predeterminado compartido, si no ninguno (demo).
        Devuelve (config, origen) con origen en visitor, default o
        demo. Un browser_id invalido lanza ValueError."""
        if browser_id:
            visitor = self.load_visitor(browser_id)
            if visitor is not None:
                return visitor, "visitor"
        default = self.load_default()
        if default is not None:
            return default, "default"
        return None, "demo"
