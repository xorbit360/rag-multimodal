#!/usr/bin/env python3
"""Establece el token LLM predeterminado compartido del demo.

Uso (en el servidor, con la misma clave de cifrado que usa la API):

    TOKEN_ENCRYPTION_KEY=<clave> python3 scripts/set_default_token.py \
        [--provider openai] [--base-url URL] [--model MODELO] [--path RUTA]

- Pide el token con getpass: no se muestra, no queda en el historial
  de shell ni en argumentos, y NUNCA se imprime.
- Lo guarda CIFRADO en DEFAULT_TOKEN_PATH (por defecto
  storage/default_token.enc, dentro del volumen persistente). Solo
  el dueno puede poner o cambiar la predeterminada, desde el servidor:
  la web publica no puede ponerla ni borrarla.
- La API usara esa predeterminada para cualquier visitante que no
  tenga token propio guardado; si el visitante guarda el suyo, usa
  el suyo. Borrar el token propio vuelve a la predeterminada.

Generar la clave de cifrado (una sola vez, guardarla en el entorno
del servidor, nunca en Git):

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Dentro del contenedor de la API (recomendado, porque ahi vive el
volumen persistente):

    docker compose exec -e TOKEN_ENCRYPTION_KEY=<clave> api \
        python scripts/set_default_token.py
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.token_vault import TokenVault  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Establece el token LLM predeterminado del demo.")
    parser.add_argument("--provider", default="openai",
                        help="openai, openrouter o custom (por defecto openai)")
    parser.add_argument("--base-url", default=None,
                        help="solo para provider custom")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="modelo fijo (por defecto gpt-4o-mini; vacio = "
                             "automatico, que exige lista de modelos en la "
                             "peticion)")
    parser.add_argument("--path", default=None,
                        help="ruta del archivo cifrado (por defecto "
                             "DEFAULT_TOKEN_PATH o storage/default_token.enc)")
    args = parser.parse_args()

    encryption_key = os.environ.get("TOKEN_ENCRYPTION_KEY", "")
    if not encryption_key:
        print("ERROR: falta TOKEN_ENCRYPTION_KEY en el entorno. "
              "Genera una clave y vuelve a intentarlo (ver la ayuda del "
              "script). No se guardo nada.", file=sys.stderr)
        return 2
    if args.provider == "custom" and not args.base_url:
        print("ERROR: provider custom requiere --base-url.", file=sys.stderr)
        return 2
    if args.provider not in ("openai", "openrouter", "custom"):
        print("ERROR: provider desconocido (openai, openrouter o custom).",
              file=sys.stderr)
        return 2

    token = getpass.getpass("Token LLM predeterminado: ")
    if not token or not token.strip():
        print("ERROR: token vacio. No se guardo nada.", file=sys.stderr)
        return 2

    store_dir = os.environ.get("TOKEN_STORE_DIR", "")
    default_path = (args.path or os.environ.get("DEFAULT_TOKEN_PATH")
                    or "storage/default_token.enc")
    vault = TokenVault(store_dir=store_dir or "storage/tokens",
                       encryption_key=encryption_key,
                       default_path=default_path)
    if not vault.configured:
        print("ERROR: TOKEN_ENCRYPTION_KEY es invalida. No se guardo nada.",
              file=sys.stderr)
        return 2
    config = {
        "api_key": token.strip(),
        "provider": args.provider,
        "base_url": args.base_url,
        "model": args.model or None,
        "auto_model": not args.model,
        "models_info": None,
    }
    try:
        vault.save_default(config)
    except Exception as exc:
        print(f"ERROR: no se pudo guardar la predeterminada: {exc}",
              file=sys.stderr)
        return 1
    # El valor del token no se muestra jamas.
    print(f"Predeterminada guardada en {default_path} (valor no mostrado). "
          "La API la usara para visitantes sin token propio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
