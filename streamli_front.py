import streamlit as st
import requests
import os
import random
import time

# 1. CONFIGURACIÓN INICIAL
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Asistente Agrícola", layout="centered", page_icon="🌿")

FRASES_AGRO = [
    "🚜 Consultando con el espantapájaros más sabio...",
    "🌾 Espera un momento, estoy convenciendo a las plantas...",
    "🍎 Revisando el manual... ¡Espero que no tenga gusanos!",
    "🌽 Buscando la respuesta entre los surcos...",
    "🚜 Arrancando el tractor del conocimiento...",
    "🍅 Analizando... Esto me importa un rábano.",
    "💧 Regando las ideas para que florezca tu respuesta...",
    "🐄 Hablando con las vacas para ver qué opinan..."
]

# 2. MEMORIA DEL CHAT Y BIENVENIDA
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "¡Hola! Soy tu asistente agrícola digital. 🚜🌱\n\nRecordaré lo que hablemos para ayudarte mejor. **¿En qué puedo ayudarte hoy?**"
        }
    ]

# Botón en la barra lateral para resetear la conversación
if st.sidebar.button("🗑️ Limpiar Conversación"):
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Memoria limpia! El campo está listo para nuevas preguntas. 🌾"}
    ]
    st.rerun()

st.title("🌿 Chat Agrícola Inteligente")

# 3. VERIFICACIÓN DE CONEXIÓN CON LA API
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

if not check_api():
    st.error("🔴 API Offline. Por favor, inicia el servidor FastAPI.")
    st.stop()

# 4. RENDERIZAR EL HISTORIAL (Para que no se borren los mensajes al recargar)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=300)

# 5. LÓGICA DE INTERACCIÓN
prompt = st.chat_input("Escribe aquí tu duda agrícola...")

if prompt:
    # A. Mostrar la pregunta del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Preparar la respuesta del asistente
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info(random.choice(FRASES_AGRO))
        
        response_placeholder = st.empty()
        full_response = ""
        ruta_foto_final = None

        try:
            # Petición a la API enviando el historial completo para que tenga memoria
            response = requests.post(
                f"{API_URL}/query_stream", 
                json={
                    "question": prompt,
                    "history": st.session_state.messages 
                }, 
                stream=True
            )

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    status_placeholder.empty() # Quitar mensaje de carga
                    
                    # Detectar si el chunk contiene la ruta de la imagen (lógica adaptada)
                    if "IMAGE_PATH:" in chunk:
                        # Extraemos la ruta si viene mezclada en el chunk
                        partes = chunk.split("IMAGE_PATH:")
                        chunk_texto = partes[0]
                        ruta_sucia = partes[1].strip()
                        # Limpiamos posibles saltos de línea residuales en la ruta
                        ruta_foto_final = ruta_sucia.split("\n")[0].strip()
                        
                        full_response += chunk_texto
                    else:
                        full_response += chunk

                    # Actualizamos la UI
                    response_placeholder.markdown(full_response + "▌")

            # D. Mostrar imagen de referencia si la hay
            if ruta_foto_final:
                ruta_foto_final = ruta_foto_final.replace("\\", "/")
                if os.path.exists(ruta_foto_final):
                    st.image(ruta_foto_final, width=400, caption="Referencia técnica del manual")

            # E. GUARDAR TODO EN LA MEMORIA (Session State)
            # Guardamos la pregunta del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Guardamos la respuesta del asistente (incluyendo la imagen si existe)
            history_item = {"role": "assistant", "content": full_response}
            if ruta_foto_final and os.path.exists(ruta_foto_final):
                history_item["image"] = ruta_foto_final
            
            st.session_state.messages.append(history_item)

        except Exception as e:
            status_placeholder.empty()
            st.error(f"Hubo un problema al conectar con el tractor de datos: {e}")


