"""Frontend Streamlit: chat con la API RAG.

Muestra: historial, Markdown, fuentes explicitas "Fuente: X, Pagina N" y,
cuando la respuesta usa un diagrama/imagen, la imagen mas cercana.

El token del LLM se puede guardar EN EL SERVIDOR, cifrado: al pulsar
"Guardar token en el servidor", el backend lo cifra y lo asocia a un
identificador aleatorio del navegador (el demo no tiene login). El
navegador guarda solo ese identificador, nunca el token. Al volver, el
frontend recupera el estado (proveedor, modelos) sin volver a pegar el
token, y las preguntas resuelven la clave en el servidor. "Borrar mi
token guardado" elimina el token del visitante y vuelve al token
predeterminado del demo (si lo hay) o al modo demo. La predeterminada
solo la establece el dueno en el servidor: la web publica no puede
ponerla ni borrarla. Sin token propio guardado, el demo usa el
predeterminado si existe; si no, responde en modo demo sin costo. Cada
respuesta indica que token se uso (tu token guardado, predeterminado
del demo o demo).

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
# URL publica del dashboard de Qdrant. En el VPS la fija el override
# de compose; en local es el Qdrant del propio docker-compose.
QDRANT_URL = os.environ.get("QDRANT_DASHBOARD_URL",
                            "http://localhost:6333/dashboard")
# Solo el identificador aleatorio del navegador vive en localStorage.
# El token nunca se guarda en el navegador: queda cifrado en el servidor.
BROWSER_ID_LS_KEY = "rag_browser_id"
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
    suffix = ""
    if data.get("token_source") == "visitor":
        suffix = " (tu token guardado)"
    elif data.get("token_source") == "default":
        suffix = " (token predeterminado del demo)"
    if used == "demo":
        st.caption("Modelo usado: demo (sin costo)")
    elif data.get("model_auto"):
        st.caption(f"Modelo usado: {used} (automático){suffix}")
    else:
        st.caption(f"Modelo usado: {used}{suffix}")


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


def _detect_by_id(browser_id):
    """Detecta con el token guardado (visitante o predeterminado) sin
    que el valor viaje: el backend lo resuelve y devuelve solo
    proveedor y modelos."""
    return httpx.post(f"{API_URL}/llm/detect", json={"browser_id": browser_id},
                      timeout=30.0).json()


def _token_status(browser_id):
    return httpx.get(f"{API_URL}/llm/token/status",
                     params={"browser_id": browser_id},
                     timeout=10.0).json()


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
    st.caption("Sin token propio, usa el token predeterminado del demo si "
               "lo hay; si no, responde en modo demo sin costo. Pega tu "
               "token y se detecta el proveedor solo. O escribe una Base "
               "URL personalizada (p. ej. Ollama local) y se detectan sus "
               "modelos automaticamente, sin token. Si guardas un token, "
               "queda encriptado en el servidor asociado a tu navegador "
               "(no se guarda en este navegador) y puedes borrarlo.")
    token = st.text_input("Token del LLM", type="password", key="llm_token")
    custom_url = st.text_input(
        "Base URL personalizada (opcional)",
        help="Endpoint compatible con OpenAI, p. ej. Ollama local: "
             "http://localhost:11434/v1. Ollama no exige token: con la URL "
             "sola se detectan sus modelos automaticamente. Si hay URL, "
             "manda sobre el token y el token guardado.")
    if st.button("Detectar proveedor") and (token or custom_url):
        with st.spinner("Detectando proveedor..."):
            det = _detect(token, custom_url or None)
        if det["detected"]:
            st.session_state["llm_detected"] = det
            st.session_state["auto_detected_for"] = token
            st.session_state["auto_detected_for_url"] = custom_url
            st.success(f"Proveedor detectado: {det['provider']}")
        else:
            st.session_state["llm_detected"] = None
            st.warning("No se reconoció el token ni la URL. Configúralo manual abajo.")
    if token and st.session_state.get("auto_detected_for") != token:
        # Al pegar un token o al cambiarlo: detectar automaticamente,
        # sin pulsar nada. (El token guardado en el servidor no pasa
        # por aqui: se resuelve con el ID del navegador mas abajo.)
        try:
            det = _detect(token)
            if det["detected"]:
                st.session_state["llm_detected"] = det
        except Exception:
            pass
        st.session_state["auto_detected_for"] = token
    if custom_url and st.session_state.get("auto_detected_for_url") != custom_url:
        # Al escribir o cambiar la URL personalizada: detectar sus
        # modelos automaticamente, aunque no haya token (Ollama no
        # exige clave). Si la URL no responde, el bloque manual abajo
        # permite escribir el modelo a mano.
        try:
            det = _detect(token, custom_url)
            if det["detected"]:
                st.session_state["llm_detected"] = det
        except Exception:
            pass
        st.session_state["auto_detected_for_url"] = custom_url
    if not token and not custom_url:
        st.session_state.pop("llm_detected", None)
        st.session_state.pop("llm_override", None)
        st.session_state["auto_detected_for"] = None
        st.session_state["auto_detected_for_url"] = None

    # Identificador del navegador + estado del token guardado. El puente
    # JavaScript genera el ID en localStorage y lo escribe en este campo;
    # Python nunca genera uno propio para no dejar guardados huerfanos.
    browser_id = st.text_input(
        "ID del navegador", key="browser_id",
        help="Solo un identificador aleatorio para tu token guardado en el "
             "servidor. No es el token.")
    if browser_id and not token:
        if (st.session_state.get("status_for") != browser_id
                or st.session_state.get("refresh_token_status")):
            try:
                st.session_state["token_status"] = _token_status(browser_id)
            except Exception:
                pass
            st.session_state["status_for"] = browser_id
            st.session_state.pop("refresh_token_status", None)
        status = st.session_state.get("token_status") or {}
        if status.get("vault_error"):
            st.warning("Guardado en el servidor no disponible: "
                       f"{status['vault_error']}")
        elif status.get("active") == "visitor":
            st.caption("Usando: tu token guardado (encriptado en el servidor).")
        elif status.get("active") == "default":
            st.caption("Usando: token predeterminado del demo.")
        else:
            st.caption("Modo demo (sin token).")
        # Proveedor y modelos del token guardado, sin volver a pegar el
        # token (la clave nunca vuelve del servidor).
        if (status.get("active") in ("visitor", "default")
                and st.session_state.get("server_detected_for") != browser_id):
            fallback = {"provider": status.get("provider") or "openai",
                        "base_url": status.get("base_url"),
                        "models": [],
                        "models_info": status.get("models_info")}
            try:
                det2 = _detect_by_id(browser_id)
                if det2.get("detected"):
                    st.session_state["server_det"] = det2
                else:
                    st.session_state["server_det"] = fallback
            except Exception:
                st.session_state["server_det"] = fallback
            st.session_state["server_detected_for"] = browser_id
    if token:
        st.session_state.pop("server_det", None)
        st.session_state["server_detected_for"] = None

    if st.button("Guardar token en el servidor") and token:
        browser_id_now = st.session_state.get("browser_id") or ""
        if not browser_id_now:
            st.warning("Espera a que se genere el ID del navegador y reintenta.")
        else:
            # La configuracion actual (detectada o manual) queda guardada
            # junto al token para reconstruirla al volver sin repegarlo.
            cfg = st.session_state.get("llm_override") or {}
            payload = {"browser_id": browser_id_now, "api_key": token,
                       "provider": cfg.get("provider", "openai"),
                       "base_url": cfg.get("base_url"),
                       "model": cfg.get("model"),
                       "auto_model": bool(cfg.get("auto_model", False)),
                       "models_info": cfg.get("models_info")}
            try:
                resp = httpx.post(f"{API_URL}/llm/token", json=payload,
                                  timeout=30.0)
                if resp.status_code != 200:
                    st.error(f"No se pudo guardar: {_error_detail(resp)[:300]}")
                else:
                    st.session_state["clear_token_field"] = True
                    st.session_state["refresh_token_status"] = True
                    st.session_state.pop("llm_detected", None)
                    st.session_state["auto_detected_for"] = None
                    st.success("Token guardado encriptado en el servidor.")
            except Exception as exc:
                st.error(f"No se pudo guardar: {exc}")
    if st.button("Borrar mi token guardado"):
        browser_id_now = st.session_state.get("browser_id") or ""
        if not browser_id_now:
            st.warning("No hay ID del navegador para borrar.")
        else:
            try:
                resp = httpx.delete(f"{API_URL}/llm/token",
                                    params={"browser_id": browser_id_now},
                                    timeout=30.0)
                if resp.status_code != 200:
                    st.error(f"No se pudo borrar: {_error_detail(resp)[:300]}")
                else:
                    st.session_state["refresh_token_status"] = True
                    st.session_state.pop("server_det", None)
                    st.session_state["server_detected_for"] = None
                    st.session_state["clear_token_field"] = True
                    st.success("Tu token guardado fue borrado. Vuelve al "
                               "predeterminado del demo o al modo demo.")
            except Exception as exc:
                st.error(f"No se pudo borrar: {exc}")

    det = st.session_state.get("llm_detected")
    server_det = st.session_state.get("server_det")
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
    elif server_det:
        # Token guardado (visitante o predeterminado): la clave se
        # resuelve en el servidor con browser_id; aqui solo viaja la
        # seleccion (proveedor, modelo), nunca el valor.
        provider = server_det["provider"]
        base_url = server_det["base_url"]
        models = server_det["models"] or []
        choice = st.selectbox("Modelo", [AUTO_LABEL] + models, index=0)
        if choice == AUTO_LABEL:
            llm_override = {"provider": provider, "base_url": base_url,
                            "model": None, "api_key": "",
                            "auto_model": True,
                            "models_info": server_det.get("models_info")}
        else:
            llm_override = {"provider": provider, "base_url": base_url,
                            "model": choice, "api_key": "",
                            "auto_model": False, "models_info": None}
        st.session_state["llm_override"] = llm_override
    else:
        st.session_state.pop("llm_override", None)

    if custom_url:
        # URL personalizada presente: manda sobre token y token guardado.
        # Un endpoint propio (Ollama local) funciona sin token.
        det = st.session_state.get("llm_detected")
        if det and det.get("provider") == "custom":
            models = det["models"] or []
            choice = st.selectbox("Modelo (personalizado)",
                                  [AUTO_LABEL] + models, index=0)
            if choice == AUTO_LABEL:
                st.session_state["llm_override"] = {
                    "provider": "custom", "base_url": custom_url,
                    "model": None, "api_key": token,
                    "auto_model": True,
                    "models_info": det.get("models_info")}
            else:
                st.session_state["llm_override"] = {
                    "provider": "custom", "base_url": custom_url,
                    "model": choice, "api_key": token,
                    "auto_model": False, "models_info": None}
        else:
            manual_model = st.text_input(
                "Modelo personalizado (si la URL no listó modelos)")
            if manual_model:
                st.session_state["llm_override"] = {
                    "provider": "custom", "base_url": custom_url,
                    "model": manual_model, "api_key": token,
                    "auto_model": False, "models_info": None}
            else:
                st.session_state["llm_override"] = {
                    "provider": "custom", "base_url": custom_url,
                    "model": None, "api_key": token,
                    "auto_model": True, "models_info": None}

    # Puente a localStorage: se ejecuta en el navegador del usuario.
    # Solo gestiona el ID del navegador (no el token, que nunca se
    # guarda aqui): si no hay ID, genera uno aleatorio y lo escribe en
    # el campo para que Python lo use en el estado del token. Si se
    # pidio vaciar el campo del token (tras guardar o borrar), lo vacia.
    # Si el navegador bloquea el almacenamiento, se ignora en silencio
    # y queda el flujo manual (pegar token + Detectar).
    clear_flag = "true" if st.session_state.get("clear_token_field") else "false"
    bridge = """
<script>
(function() {
  var KEY = "rag_browser_id";
  var CLEAR = __CLEAR__;
  function findInput(label) {
    var doc = window.parent.document;
    return doc.querySelector('input[aria-label="' + label + '"]');
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
    var idInput = findInput("ID del navegador");
    var tokenInput = findInput("Token del LLM");
    if (!tokenInput) { tokenInput = window.parent.document.querySelector('input[type="password"]'); }
    tries++;
    if (!idInput && tries < 15) { return; }
    clearInterval(timer);
    try {
      var id = localStorage.getItem(KEY);
      if (!id) {
        id = (window.parent.crypto && window.parent.crypto.randomUUID)
          ? window.parent.crypto.randomUUID()
          : ("id-" + Math.random().toString(36).slice(2) + Date.now().toString(36));
        localStorage.setItem(KEY, id);
      }
      if (idInput && !idInput.value) { setNative(idInput, id); }
      if (CLEAR && tokenInput && tokenInput.value) { setNative(tokenInput, ""); }
    } catch (e) { /* almacenamiento no disponible: se ignora */ }
  }, 200);
})();
</script>
""".replace("__CLEAR__", clear_flag)
    components.html(bridge, height=0)
    st.session_state.pop("clear_token_field", None)

    st.header("Visualizar RAG")
    st.caption("Abre el dashboard real de Qdrant, la base de vectores del demo: colecciones, puntos y vectores. Pide usuario y contraseña; solo quien tenga la clave puede entrar.")
    st.link_button("Visualizar RAG", QDRANT_URL)

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
                          "top_k": "5",
                          "browser_id": st.session_state.get("browser_id") or ""},
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
                    "transcription_model": data.get("transcription_model"),
                    "token_source": data.get("token_source")})
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
                          "top_k": "5",
                          "browser_id": st.session_state.get("browser_id") or ""},
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
                    "model_auto": data.get("model_auto"),
                    "token_source": data.get("token_source")})
    else:
        if not text:
            st.stop()
        st.session_state["history"].append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los documentos..."):
                payload = {"question": text}
                if st.session_state.get("browser_id"):
                    payload["browser_id"] = st.session_state["browser_id"]
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
                    "model_auto": data.get("model_auto"),
                    "token_source": data.get("token_source")})
