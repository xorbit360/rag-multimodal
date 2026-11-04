"""Frontend Streamlit: chat con la API RAG.

Muestra: historial, Markdown, fuentes explicitas "Fuente: X, Pagina N" y,
cuando la respuesta usa un diagrama/imagen, la imagen mas cercana.
La clave del LLM, si se pega, solo vive en la sesion: no se guarda.
"""
import os
import time

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Multimodal", layout="wide")
st.title("RAG Multimodal — PDFs tecnicos")

with st.sidebar:
    st.header("Ingesta")
    uploaded = st.file_uploader("Sube un PDF tecnico", type=["pdf"])
    if uploaded is not None and st.button("Ingerir PDF"):
        with st.spinner("Subiendo e ingiriendo..."):
            resp = httpx.post(f"{API_URL}/ingest",
                              files={"file": (uploaded.name, uploaded.getvalue(),
                                              "application/pdf")},
                              timeout=60.0)
            if resp.status_code != 200:
                st.error(f"Error al subir: {resp.text}")
            else:
                job_id = resp.json()["job_id"]
                st.session_state["last_job"] = job_id
                for _ in range(60):
                    job = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=10.0).json()
                    if job["status"] in ("completed", "failed"):
                        break
                    time.sleep(2)
                if job["status"] == "completed":
                    st.success(f"Listo: {job['chunks_count']} chunks, "
                               f"{job['images_count']} imagenes")
                else:
                    st.error(f"Fallo: {job.get('error')}")
    if st.button("Ver documentos indexados"):
        docs = httpx.get(f"{API_URL}/documents", timeout=10.0).json()
        st.session_state["docs"] = docs
    for doc in st.session_state.get("docs", []):
        st.write(f"- {doc['document_name']} ({doc['chunks']} chunks, "
                 f"{doc['images']} imagenes)")

    st.header("LLM (opcional)")
    st.caption("Sin clave, responde en modo demo sin costo. Pega tu token y "
               "se detecta el proveedor solo. La clave solo vive en esta "
               "sesion, no se guarda.")
    token = st.text_input("Token del LLM", type="password")
    if st.button("Detectar proveedor") and token:
        with st.spinner("Detectando proveedor..."):
            det = httpx.post(f"{API_URL}/llm/detect", json={"api_key": token},
                             timeout=30.0).json()
        if det["detected"]:
            st.session_state["llm_detected"] = det
            st.success(f"Proveedor detectado: {det['provider']}")
        else:
            st.session_state["llm_detected"] = None
            st.warning("No se reconoció el token. Configúralo manual abajo.")
    det = st.session_state.get("llm_detected")
    provider = None
    base_url = None
    model = None
    if det:
        provider = det["provider"]
        base_url = det["base_url"]
        models = det["models"] or []
        if models:
            default = "gpt-4o-mini" if "gpt-4o-mini" in models else models[0]
            model = st.selectbox("Modelo", models, index=models.index(default))
        else:
            model = st.text_input("Modelo")
    else:
        provider = st.selectbox("Proveedor (manual)",
                                ["openai", "openrouter", "custom"])
        if provider == "custom":
            base_url = st.text_input("Base URL (ej. http://localhost:11434/v1)")
        model = st.text_input("Modelo")
    if token and model:
        st.session_state["llm_override"] = {"provider": provider,
                                            "base_url": base_url,
                                            "model": model,
                                            "api_key": token}
    elif not token:
        st.session_state.pop("llm_override", None)

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_url"):
            st.image(f"{API_URL}{msg['image_url']}",
                     caption=f"Diagrama / imagen — Pagina {msg.get('image_page')}")
        for src in msg.get("sources", []):
            st.caption(f"Fuente: {src['document_name']}, Pagina {src['page']} "
                       f"({src['kind']})")

question = st.chat_input("Pregunta sobre tus documentos...")
if question:
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            payload = {"question": question}
            if st.session_state.get("llm_override"):
                payload["llm"] = st.session_state["llm_override"]
            resp = httpx.post(f"{API_URL}/ask", json=payload, timeout=120.0)
            if resp.status_code != 200:
                st.error(f"Error del LLM: {resp.text[:300]}")
                st.stop()
            data = resp.json()
            st.markdown(data["answer"])
            if data.get("image_url"):
                st.image(f"{API_URL}{data['image_url']}",
                         caption=f"Diagrama / imagen — Pagina {data['image_page']}")
            for src in data.get("sources", []):
                st.caption(f"Fuente: {src['document_name']}, Pagina {src['page']} "
                           f"({src['kind']})")
            st.session_state["history"].append({
                "role": "assistant", "content": data["answer"],
                "image_url": data.get("image_url"),
                "image_page": data.get("image_page"),
                "sources": data.get("sources", [])})
