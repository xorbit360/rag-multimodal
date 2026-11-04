"""Pagina aparte 'Trabajos': lo que hay en Redis, solo lectura.

Esta pagina NO es el chat y no contamina su visual: Streamlit la
muestra como un menu aparte en la barra lateral. Aqui el evaluador
ve la ficha tecnica de la ingesta: que trabajos hay en el JobStore
(Redis en docker-compose), en que estado estan, cuantos chunks e
imagenes genero cada uno y si alguno fallo, mas la ficha del vector
store (coleccion, puntos, dimension, embedding).

No hay ningun boton que cree, borre o edite nada: es solo lectura.
Para probar una ingesta nueva, se sube un PDF desde la pagina del
chat (menu Ingesta); aqui solo se consulta su progreso.
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Trabajos — RAG Multimodal", layout="wide")
st.title("Trabajos en Redis — solo lectura")
st.caption(
    "Lo que hay en el JobStore (Redis en docker-compose, memoria en "
    "local) y en la base de vectores. Solo lectura: aqui no se crea, "
    "no se borra y no se edita nada. Para ingerir un PDF, usa el menu "
    "de Ingesta en la pagina del chat."
)

if st.button("Actualizar"):
    st.rerun()

try:
    health = httpx.get(f"{API_URL}/health", timeout=10.0).json()
    st.caption(
        f"Cola: {health.get('queue')} · Vector store: "
        f"{health.get('vector_store')}"
    )
except Exception as exc:
    st.error(f"No se pudo consultar la API: {exc}")
    st.stop()

try:
    jobs = httpx.get(f"{API_URL}/jobs", timeout=10.0).json()
except Exception as exc:
    st.error(f"No se pudieron listar los trabajos: {exc}")
    jobs = []

if not jobs:
    st.info(
        "No hay trabajos registrados en el JobStore en este momento. "
        "Los trabajos aparecen aqui cuando se ingiere un PDF nuevo; "
        "los documentos ya indexados siguen abajo, en la seccion de "
        "documentos, aunque el registro de trabajos se haya reiniciado."
    )
else:
    rows = [
        {
            "Documento": j["document_name"],
            "Estado": j["status"],
            "Progreso": f"{round(j['progress'] * 100)}%",
            "Mensaje": j["message"],
            "Chunks": j["chunks_count"],
            "Imagenes": j["images_count"],
            "Duplicado": "si" if j["duplicate"] else "no",
            "Error": j["error"] or "",
            "Job": j["job_id"],
        }
        for j in jobs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.header("Documentos indexados")
try:
    docs = httpx.get(f"{API_URL}/documents", timeout=10.0).json()
    if not docs:
        st.info("No hay documentos indexados.")
    else:
        st.dataframe(
            [
                {
                    "Documento": d["document_name"],
                    "Chunks": d["chunks"],
                    "Imagenes": d["images"],
                    "Estado": d["status"],
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )
except Exception as exc:
    st.error(f"No se pudieron listar los documentos: {exc}")

st.header("Ficha tecnica del vector store")
try:
    overview = httpx.get(f"{API_URL}/data/overview", timeout=10.0).json()
    st.write(f"- Coleccion: {overview['collection']}")
    st.write(f"- Puntos: {overview['points']}")
    st.write(f"- Dimension: {overview['dimension']}")
    st.write(f"- Embedding: {overview['embedding']}")
    st.write(f"- Vector store: {overview['vector_store']}")
    for sample in overview.get("samples", []):
        st.write(
            f"- Punto guardado: {sample['document_name']}, pagina "
            f"{sample['page']} ({sample['kind']}): {sample['text'][:200]}"
        )
except Exception:
    st.caption(
        "La ficha del vector store solo esta disponible en esta "
        "configuracion (/data/overview)."
    )
