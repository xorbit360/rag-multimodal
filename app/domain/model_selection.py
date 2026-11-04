"""Seleccion automatica del modelo mas economico.

Logica pura, sin llamadas a proveedores: recibe la lista de modelos que
el proveedor devolvio en /models (con precio y modalidad cuando los trae)
y elige segun lo que requiera la pregunta.

Reglas, en orden:
1. Si hay imagen, solo valen modelos con vision. Si ninguno la tiene, no
   se elige ninguno (el endpoint devuelve un error claro, nunca una
   descripcion inventada).
2. Si los modelos traen precio (OpenRouter lo trae en /models), gana el
   precio de prompt mas bajo.
3. Si no hay precios, gana el orden de costo conocido: mini/flash y
   modelos economicos primero, segun la tabla CHEAP_ORDER.
4. Si no hay datos de precio ni coincidencia en la tabla, se respeta el
   orden en que el proveedor listo los modelos.
5. Sin candidatos, no se elige ninguno: el llamador debe pedir al usuario
   que elija un modelo manual.

La tabla CHEAP_ORDER es un ORDEN aproximado de costo conocido, no precios
exactos: los precios reales cambian y solo el proveedor los confirma.
Los modelos desconocidos quedan al final, en el orden del proveedor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelCandidate:
    """Un modelo tal como lo lista el proveedor."""
    id: str
    prompt_price: float | None = None  # USD por token, si el proveedor lo da
    modality: str | None = None  # ej. "text+image->text" en OpenRouter


# Orden de costo conocido, del mas economico al mas caro. Se compara por
# subcadena en minusculas: un modelo desconocido no coincide y queda al
# final. Mini y flash primero, tal como pide el encargo.
CHEAP_ORDER = [
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4.1",
    "claude-haiku",
    "gemini-flash",
    "gemini-pro",
    "llama-3",
    "qwen",
    "gpt-5-mini",
    "gpt-5",
    "claude-sonnet",
    "claude-opus",
]

# Subcadenas que indican que un modelo acepta imagenes. Se suma a la
# modalidad "image" que OpenRouter devuelve en architecture.modality.
VISION_HINTS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "vision",
    "claude-3",
    "claude-sonnet-4",
    "claude-opus-4",
    "gemini",
    "llama-3.2",
    "qwen-vl",
    "qwen2-vl",
    "pixtral",
]


def supports_vision(candidate: ModelCandidate) -> bool:
    if candidate.modality and "image" in candidate.modality.lower():
        return True
    lowered = candidate.id.lower()
    return any(hint in lowered for hint in VISION_HINTS)


def _cheap_rank(model_id: str) -> int:
    lowered = model_id.lower()
    for rank, pattern in enumerate(CHEAP_ORDER):
        if pattern in lowered:
            return rank
    return len(CHEAP_ORDER)


def select_model(candidates: list[ModelCandidate],
                 need_vision: bool = False) -> str | None:
    """Elige el modelo mas economico. Devuelve None si no hay candidatos
    o si se pide vision y ninguno la soporta."""
    if not candidates:
        return None
    pool = [c for c in candidates if supports_vision(c)] if need_vision else list(candidates)
    if not pool:
        return None
    priced = [c for c in pool if c.prompt_price is not None]
    if priced:
        # Precio mas bajo; empate: orden de costo conocido; luego orden
        # original del proveedor (el sort es estable).
        return min(priced,
                   key=lambda c: (c.prompt_price, _cheap_rank(c.id))).id
    return min(pool, key=lambda c: _cheap_rank(c.id)).id


def select_transcription_model(candidates: list[ModelCandidate]) -> str:
    """Modelo de transcripcion para el modo automatico: el whisper mas
    economico si el proveedor lista alguno; si no, whisper-1."""
    whispers = [c for c in candidates if "whisper" in c.id.lower()]
    if whispers:
        priced = [c for c in whispers if c.prompt_price is not None]
        if priced:
            return min(priced, key=lambda c: c.prompt_price).id
        return whispers[0].id
    return "whisper-1"
