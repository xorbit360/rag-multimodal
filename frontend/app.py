"""Frontend Streamlit: chat con la API RAG.

Muestra: historial, Markdown, fuentes explicitas "Fuente: X, Pagina N" y,
cuando la respuesta usa un diagrama/imagen, la imagen mas cercana.
La clave del LLM, si se pega, solo vive en la sesion: no se guarda en
el servidor. Si el usuario pulsa "Guardar token en este navegador", se
persiste SOLO en el localStorage de su navegador (nunca en el backend,
ni en disco, ni en Git, ni en logs); "Borrar token guardado" lo elimina
de localStorage y de la sesion. Al cargar, un token guardado se restaura
en el campo y se detecta el proveedor automaticamente.

El chat es SOLO para preguntar: acepta texto, imagen o audio, pero NO
PDF (los PDF solo se suben por la barra lateral). El modo de modelo
predeterminado es "Automatico (el mas economico)": el backend elige el
modelo segun lo que requiera la pregunta (vision si hay imagen,
transcripcion si hay audio, chat si es solo texto). Si el usuario elige
un modelo manual, se respeta su eleccion. Cada respuesta muestra el
modelo usado.
"""
import json
import os
import time

import httpx
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TOKEN_LS_KEY = "rag_llm_token"
AUTO_LABEL = "Automático (el más económico)"
CHAT_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg", "webm"]
IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_MIMES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4",
               "audio/ogg", "audio/webm"}

st.set_page_config(page_title="RAG Multimodal", layout="wide",
                   initial_sidebar_state="auto")
st.title("RAG Multimodal — PDFs tecnicos")

# Ajustes responsivos (sin redisenar la app): en celular el chat va a
# ancho completo, los controles son grandes y tactiles, y la barra
# lateral queda colapsable (initial_sidebar_state="auto" ya la oculta
# en pantallas pequenas). El font-size 16px en el input evita el zoom
# automatico de iOS al enfocar.
st.markdown("""
<style>
button { min-height: 44px; }
@media (max-width: 768px) {
  div[data-testid="stMainBlockContainer"] {
    padding-left: 1rem !important; padding-right: 1rem !important;
  }
  div[data-testid="stChatInput"] textarea { font-size: 16px !important; }
  div[data-testid="stChatInput"] button { min-height: 48px !important; }
  button[kind="secondary"], button[kind="primary"] {
    min-height: 48px !important; font-size: 16px !important; width: 100%;
  }
  div[data-testid="stFileUploader"] { font-size: 16px; }
  img { max-width: 100% !important; height: auto !important; }
}
</style>
""", unsafe_allow_html=True)


def _error_detail(resp):
    """Mensaje limpio de un error de la API: el detail si viene JSON,
    si no el texto crudo. Antes se mostraba el JSON completo."""
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return resp.text


def show_model_caption(data):
    used = data.get("model_used")
    if not used:
        return
    if used == "demo":
        st.caption("Modelo usado: demo (sin costo)")
    elif data.get("model_auto"):
        st.caption(f"Modelo usado: {used} (automático)")
    else:
        st.caption(f"Modelo usado: {used}")


def show_answer(data):
    st.markdown(data["answer"])
    if data.get("image_url"):
        st.image(f"{API_URL}{data['image_url']}",
                 caption=f"Diagrama / imagen — Pagina {data['image_page']}")
    for src in data.get("sources", []):
        st.caption(f"Fuente: {src['document_name']}, Pagina {src['page']} "
                   f"({src['kind']})")
    show_model_caption(data)


def _detect(token, base_url=None):
    payload = {"api_key": token}
    if base_url:
        payload["base_url"] = base_url
    return httpx.post(f"{API_URL}/llm/detect", json=payload,
                      timeout=30.0).json()


with st.sidebar:
    st.header("Ingesta")
    st.caption("Los PDF solo se suben aqui. El chat es solo para preguntar.")
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
                    if job.get("duplicate"):
                        st.info("Este PDF ya estaba indexado: se reutilizo "
                                "la ingesta anterior, sin duplicar chunks "
                                f"({job['chunks_count']} chunks, "
                                f"{job['images_count']} imagenes).")
                    else:
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
               "sesion, no se guarda en el servidor.")
    token = st.text_input("Token del LLM", type="password", key="llm_token")
    if st.button("Detectar proveedor") and token:
        with st.spinner("Detectando proveedor..."):
            det = _detect(token)
        if det["detected"]:
            st.session_state["llm_detected"] = det
            st.session_state["auto_detected_for"] = token
            st.success(f"Proveedor detectado: {det['provider']}")
        else:
            st.session_state["llm_detected"] = None
            st.warning("No se reconoció el token. Configúralo manual abajo.")
    if token and st.session_state.get("auto_detected_for") != token:
        # Al cargar con un token guardado (restaurado desde localStorage)
        # o al cambiarlo: detectar automaticamente, sin pulsar nada.
        try:
            det = _detect(token)
            if det["detected"]:
                st.session_state["llm_detected"] = det
        except Exception:
            pass
        st.session_state["auto_detected_for"] = token
    if not token:
        st.session_state.pop("llm_detected", None)
        st.session_state.pop("llm_override", None)
        st.session_state["auto_detected_for"] = None

    if st.button("Guardar token en este navegador") and token:
        # Solo marca la peticion: el JavaScript del puente lee el campo y
        # escribe en localStorage. El token NO viaja a ningun otro sitio.
        st.session_state["save_token_requested"] = True
        st.success("Token guardado en este navegador (solo aqui, no en el "
                   "servidor).")
    if st.button("Borrar token guardado"):
        st.session_state["delete_token_requested"] = True
        st.session_state.pop("llm_detected", None)
        st.session_state.pop("llm_override", None)
        st.session_state["auto_detected_for"] = None
        st.success("Token borrado del navegador y de la sesion.")

    det = st.session_state.get("llm_detected")
    llm_override = None
    if token:
        if det:
            provider = det["provider"]
            base_url = det["base_url"]
            models = det["models"] or []
            choice = st.selectbox("Modelo", [AUTO_LABEL] + models, index=0)
            if choice == AUTO_LABEL:
                llm_override = {"provider": provider, "base_url": base_url,
                                "model": None, "api_key": token,
                                "auto_model": True,
                                "models_info": det.get("models_info")}
            else:
                llm_override = {"provider": provider, "base_url": base_url,
                                "model": choice, "api_key": token,
                                "auto_model": False, "models_info": None}
        else:
            provider = st.selectbox("Proveedor (manual)",
                                    ["openai", "openrouter", "custom"])
            if provider == "custom":
                base_url = st.text_input("Base URL (ej. http://localhost:11434/v1)")
            else:
                base_url = None
            manual_model = st.text_input("Modelo (vacío = automático)")
            if manual_model:
                llm_override = {"provider": provider, "base_url": base_url,
                                "model": manual_model, "api_key": token,
                                "auto_model": False, "models_info": None}
            else:
                llm_override = {"provider": provider, "base_url": base_url,
                                "model": None, "api_key": token,
                                "auto_model": True, "models_info": None}
        st.session_state["llm_override"] = llm_override
    else:
        st.session_state.pop("llm_override", None)

    # Puente a localStorage: se ejecuta en el navegador del usuario.
    # - Guardar: lee el campo y escribe SOLO en localStorage.
    # - Borrar: elimina de localStorage y vacia el campo.
    # - Cargar: si hay un token guardado y el campo esta vacio, lo restaura
    #   (el evento dispara un rerun y la deteccion automatica de arriba).
    # Si el navegador bloquea el almacenamiento, se ignora en silencio y
    # queda el flujo manual (pegar token + Detectar).
    save_flag = "true" if st.session_state.get("save_token_requested") else "false"
    delete_flag = "true" if st.session_state.get("delete_token_requested") else "false"
    bridge = """
<script>
(function() {
  var KEY = "rag_llm_token";
  var SAVE = __SAVE__;
  var DEL = __DELETE__;
  function findInput() {
    var doc = window.parent.document;
    var el = doc.querySelector('input[aria-label="Token del LLM"]');
    if (!el) { el = doc.querySelector('input[type="password"]'); }
    return el;
  }
  function setNative(el, value) {
    var proto = window.parent.HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new window.parent.Event("input", {bubbles: true}));
    el.dispatchEvent(new window.parent.Event("change", {bubbles: true}));
  }
  var tries = 0;
  var timer = setInterval(function() {
    var input = findInput();
    tries++;
    if (!input && tries < 15) { return; }
    clearInterval(timer);
    if (!input) { return; }
    try {
      if (DEL) {
        localStorage.removeItem(KEY);
        if (input.value) { setNative(input, ""); }
      } else if (SAVE) {
        if (input.value) { localStorage.setItem(KEY, input.value); }
      } else {
        var saved = localStorage.getItem(KEY);
        if (saved && !input.value) { setNative(input, saved); }
      }
    } catch (e) { /* almacenamiento no disponible: se ignora */ }
  }, 200);
})();
</script>
""".replace("__SAVE__", save_flag).replace("__DELETE__", delete_flag)
    components.html(bridge, height=0)
    st.session_state.pop("save_token_requested", None)
    st.session_state.pop("delete_token_requested", None)

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("transcription"):
            st.caption(f"Transcripción: {msg['transcription']}")
        if msg.get("description"):
            st.caption(f"Descripción de la imagen: {msg['description']}")
        if msg.get("image_url"):
            st.image(f"{API_URL}{msg['image_url']}",
                     caption=f"Diagrama / imagen — Pagina {msg.get('image_page')}")
        for src in msg.get("sources", []):
            st.caption(f"Fuente: {src['document_name']}, Pagina {src['page']} "
                       f"({src['kind']})")
        show_model_caption(msg)

chat_value = st.chat_input(
    "Pregunta sobre tus documentos... (PDF solo en la barra lateral)",
    accept_file=True, file_type=CHAT_FILE_TYPES, accept_audio=True)
if chat_value:
    text = (chat_value.text or "").strip()
    audio_file = getattr(chat_value, "audio", None)
    files = list(getattr(chat_value, "files", []) or [])
    audio_upload = None
    image_upload = None
    for f in files:
        if f.type in IMAGE_MIMES and image_upload is None:
            image_upload = f
        elif f.type in AUDIO_MIMES and audio_upload is None:
            audio_upload = f
    if audio_file is not None:
        audio_upload = audio_file
    if len(files) > 1:
        st.caption("Se usó solo el primer adjunto.")
    if audio_upload is not None and image_upload is not None:
        st.warning("Envía solo un adjunto por pregunta: audio o imagen.")
        st.stop()
    override = st.session_state.get("llm_override")

    if audio_upload is not None:
        if not override:
            st.warning("El audio necesita un token: pégalo en la sección LLM. "
                       "En modo demo sin token no se puede transcribir.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(text if text else "🎤 Audio enviado")
        with st.chat_message("assistant"):
            with st.spinner("Transcribiendo y buscando en los documentos..."):
                resp = httpx.post(
                    f"{API_URL}/ask-audio",
                    files={"audio": (audio_upload.name, audio_upload.getvalue(),
                                     audio_upload.type)},
                    data={"provider": override["provider"],
                          "model": override.get("model") or "",
                          "api_key": override["api_key"],
                          "base_url": override.get("base_url") or "",
                          "auto_model": "true" if override.get("auto_model") else "false",
                          "models_info_json": json.dumps(override.get("models_info"))
                          if override.get("models_info") else "",
                          "transcription_model": "whisper-1",
                          "question": text,
                          "top_k": "5"},
                    timeout=300.0)
                if resp.status_code != 200:
                    st.error(f"Error: {_error_detail(resp)[:300]}")
                    st.stop()
                data = resp.json()
                st.caption(f"Transcripción: {data['transcription']}")
                show_answer(data)
                if data.get("transcription_model"):
                    st.caption(f"Transcripción con: {data['transcription_model']}")
                st.session_state["history"].append({
                    "role": "user",
                    "content": text if text else "🎤 Audio enviado"})
                st.session_state["history"].append({
                    "role": "assistant", "content": data["answer"],
                    "transcription": data["transcription"],
                    "image_url": data.get("image_url"),
                    "image_page": data.get("image_page"),
                    "sources": data.get("sources", []),
                    "model_used": data.get("model_used"),
                    "model_auto": data.get("model_auto"),
                    "transcription_model": data.get("transcription_model")})
    elif image_upload is not None:
        if not override:
            st.warning("La pregunta con imagen necesita un token: pégalo en "
                       "la sección LLM. En modo demo sin token no se puede "
                       "describir imágenes.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(text if text else "🖼️ Imagen adjunta")
        with st.chat_message("assistant"):
            with st.spinner("Describiendo la imagen y buscando referencias..."):
                resp = httpx.post(
                    f"{API_URL}/ask-image",
                    files={"image": (image_upload.name, image_upload.getvalue(),
                                     image_upload.type)},
                    data={"provider": override["provider"],
                          "model": override.get("model") or "",
                          "api_key": override["api_key"],
                          "base_url": override.get("base_url") or "",
                          "auto_model": "true" if override.get("auto_model") else "false",
                          "models_info_json": json.dumps(override.get("models_info"))
                          if override.get("models_info") else "",
                          "question": text,
                          "top_k": "5"},
                    timeout=180.0)
                if resp.status_code != 200:
                    st.error(f"Error: {_error_detail(resp)[:300]}")
                    st.stop()
                data = resp.json()
                st.caption(f"Descripción de la imagen: {data['description']}")
                show_answer(data)
                st.session_state["history"].append({
                    "role": "user",
                    "content": text if text else "🖼️ Imagen adjunta"})
                st.session_state["history"].append({
                    "role": "assistant", "content": data["answer"],
                    "description": data["description"],
                    "image_url": data.get("image_url"),
                    "image_page": data.get("image_page"),
                    "sources": data.get("sources", []),
                    "model_used": data.get("model_used"),
                    "model_auto": data.get("model_auto")})
    else:
        if not text:
            st.stop()
        st.session_state["history"].append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos..."):
                payload = {"question": text}
                if override:
                    payload["llm"] = override
                resp = httpx.post(f"{API_URL}/ask", json=payload, timeout=120.0)
                if resp.status_code != 200:
                    st.error(f"Error del LLM: {_error_detail(resp)[:300]}")
                    st.stop()
                data = resp.json()
                show_answer(data)
                st.session_state["history"].append({
                    "role": "assistant", "content": data["answer"],
                    "image_url": data.get("image_url"),
                    "image_page": data.get("image_page"),
                    "sources": data.get("sources", []),
                    "model_used": data.get("model_used"),
                    "model_auto": data.get("model_auto")})
