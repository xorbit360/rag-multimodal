"""Frontend Streamlit: chat con la API RAG.

Muestra: historial, Markdown, fuentes explicitas "Fuente: X, Pagina N" y,
cuando la respuesta usa un diagrama/imagen, la imagen mas cercana.
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
            resp = httpx.post(f"{API_URL}/ask", json={"question": question},
                              timeout=120.0)
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
