"""Pagina aparte 'Trabajos': lo que hay en Redis y en el vector store.

Esta pagina NO es el chat y no contamina su visual: Streamlit la
muestra como un menu aparte en la barra lateral. Aqui el evaluador
ve la ficha tecnica de la ingesta: que trabajos hay en el JobStore
(Redis en docker-compose), en que estado estan, cuantos chunks e
imagenes genero cada uno y si alguno fallo, mas la ficha del vector
store (coleccion, puntos, dimension, embedding).

Los trabajos son solo lectura; los documentos SI se pueden eliminar
(con confirmacion, desde la seccion Documentos indexados y desde el
chat). Para probar una ingesta nueva, se sube un PDF desde la pagina
del chat (menu Ingesta); aqui se consulta su progreso.
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Trabajos — RAG Multimodal", layout="wide")
st.title("Trabajos en Redis")
st.caption(
    "Lo que hay en el JobStore (Redis en docker-compose, memoria en "
    "local) y en la base de vectores. Los trabajos son solo lectura; "
    "los documentos se pueden eliminar con confirmacion abajo. Para "
    "ingerir un PDF, usa el menu de Ingesta en la pagina del chat."
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
st.caption("Eliminar borra el documento completo: sus chunks en el "
           "vector store, sus trabajos y sus archivos. Es irreversible; "
           "para volver a tenerlo, subelo de nuevo desde el chat.")
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
        for d in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- {d['document_name']} ({d['chunks']} chunks, "
                         f"{d['images']} imagenes)")
            with col2:
                if st.button("Eliminar", key=f"del-{d['document_id']}"):
                    st.session_state["confirm_delete"] = d["document_id"]
            if st.session_state.get("confirm_delete") == d["document_id"]:
                st.warning("¿Borrar este PDF? No se puede deshacer.")
                if st.button("Si, borrar", key=f"confirm-{d['document_id']}"):
                    resp = httpx.delete(
                        f"{API_URL}/documents/{d['document_id']}",
                        timeout=30.0)
                    if resp.status_code == 200:
                        st.success(
                            f"Borrado: {d['document_name']} "
                            f"({resp.json()['chunks_removed']} chunks)")
                    else:
                        st.error(f"No se pudo borrar: {resp.text}")
                    st.session_state["confirm_delete"] = None
                    st.rerun()
                if st.button("Cancelar", key=f"cancel-{d['document_id']}"):
                    st.session_state["confirm_delete"] = None
                    st.rerun()
except Exception as exc:
    st.error(f"No se pudieron listar los documentos: {exc}")

st.header("Ficha tecnica del vector store")
st.caption("Para ver la coleccion real en Qdrant, fuera de esta pantalla: "
           "el acceso pide usuario y contrasena, que se comparten solo "
           "por correo privado, nunca aqui ni en el repositorio. "
           "Uso solo de consulta: no borres ni edites puntos.")
st.link_button("Ver base vectorial (Qdrant)",
               "https://qdrant.rag.xorbit360.com/dashboard")
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
