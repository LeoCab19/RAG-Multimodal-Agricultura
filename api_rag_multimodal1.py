import os
import chromadb
import torch
import logging
import time
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from PIL import Image
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from functools import lru_cache
from rank_bm25 import BM25Okapi
import uvicorn

# Cargar variables de entorno
load_dotenv()

# =========================
# CONFIGURACIÓN
# =========================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_multimodalGrande" 
MODELO_LLM = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
MODELO_EMB = "intfloat/multilingual-e5-large" 
MODELO_RERANKER = os.getenv("MODELO_RERANKER", "BAAI/bge-reranker-v2-m3")
TOP_K = int(os.getenv("TOP_K", 10))
MAX_TOKENS = int(os.getenv("MAX_TOKENS_LLM", 10000))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API_AGRICOLA_FINAL")

app = FastAPI()

# =========================
# CARGA DE MODELOS Y VARIABLES GLOBALES
# =========================
logger.info("Cargando Orquesta de Modelos...")
device = "cuda" if torch.cuda.is_available() else "gpu"

embedder = SentenceTransformer(MODELO_EMB, trust_remote_code=True, device=device)
reranker = CrossEncoder(MODELO_RERANKER, device=device, automodel_args={"torch_dtype": torch.float16} if device == "cuda" else {})

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODELO_LLM, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODELO_LLM, trust_remote_code=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

# Variables globales para BM25
bm25_index = None
all_docs_text = []
all_docs_metadata = []
all_ids = []

# =========================
# FUNCIONES AUXILIARES Y TÉCNICAS AVANZADAS
# =========================

@lru_cache(maxsize=1)
def get_collection():
    logger.info("📦 [CACHE MISS] Accediendo a la base de datos Chroma por primera vez...")
    return client.get_collection(COLLECTION_NAME)

def init_bm25():
    """Técnica avanzada: Búsqueda Híbrida (Parte 1 - Indexación)"""
    global bm25_index, all_docs_text, all_docs_metadata, all_ids
    logger.info("🔍 Generando índice BM25 para búsqueda híbrida...")
    
    # Extraemos todos los datos de Chroma para el motor de texto plano
    data = collection.get()
    all_docs_text = data["documents"]
    all_docs_metadata = data["metadatas"]
    all_ids = data["ids"]
    
    if not all_docs_text:
        logger.error("❌ La colección está vacía. No se puede inicializar BM25.")
        return

    tokenized_corpus = [doc.lower().split() for doc in all_docs_text]
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info(f"✅ BM25 listo con {len(all_docs_text)} documentos.")

def expand_query_agricola(question: str) -> str:
    """Técnica avanzada 1: Query Expansion"""
    return f"Información técnica sobre, descripción, síntomas y tratamiento de: {question}"

def apply_rrf(vector_ids: List[str], bm25_ids: List[str], k: int = 60) -> List[str]:
    """Técnica avanzada 2: Reciprocal Rank Fusion (RRF)"""
    scores = {}
    for rank, idx in enumerate(vector_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + k)
    for rank, idx in enumerate(bm25_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + k)
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# =========================
# LÓGICA DE RECUPERACIÓN (HÍBRIDA + RERANK)
# =========================

def retrieve_smart(question):
    start_time = time.time()

    cache_info = get_collection.cache_info()
    logger.info(f"📊 Estado de Caché DB: Hits={cache_info.hits}, Misses={cache_info.misses}")
    db_collection = get_collection()
    
    # 1. Query Expansion
    expanded_query = expand_query_agricola(question)
    logger.info(f"✅ EXPAND_QUERY Ha pasado")
    
    # 2. Búsqueda Vectorial
    q_emb = embedder.encode([f"query: {expanded_query}"], normalize_embeddings=True).tolist()
    res_v = db_collection.query(query_embeddings=q_emb, n_results=TOP_K)
    v_ids = res_v["ids"][0]
    logger.info(f"✅ bUSQUEDA VECTORIAL Ha pasado")
    
    # 3. Búsqueda BM25 (Palabras clave)
    tokenized_query = question.lower().split()
    bm25_res_ids = bm25_index.get_top_n(tokenized_query, all_ids, n=TOP_K)
    logger.info(f"✅ BM25 Ha pasado")
    
    # 4. Fusión con RRF
    rrf_ordered_ids = apply_rrf(v_ids, bm25_res_ids)

    logger.info(f"✅ RRF Ha pasado")
    
    # 5. Preparar candidatos para Re-ranking
    # Creamos diccionarios de búsqueda rápida
    lookup_docs = {idx: txt for idx, txt in zip(all_ids, all_docs_text)}
    lookup_metas = {idx: meta for idx, meta in zip(all_ids, all_docs_metadata)}
    
    candidate_docs = [lookup_docs[idx] for idx in rrf_ordered_ids if idx in lookup_docs]
    candidate_metas = [lookup_metas[idx] for idx in rrf_ordered_ids if idx in lookup_metas]
    
    # 6. Re-ranking Final (Cross-Encoder)
    pairs = [[question, doc] for doc in candidate_docs]
    cross_scores = reranker.predict(pairs)
    
    scored_results = sorted(
        zip(cross_scores, candidate_docs, candidate_metas, rrf_ordered_ids), 
        key=lambda x: x[0], 
        reverse=True
    )
    
    mejor_score, mejor_doc, mejor_meta, mejor_id = scored_results[0]

    # --- LÓGICA DE IMÁGENES ---
    img_path = mejor_meta.get("images", "")
    final_images = []
    if img_path:
        img_path_fixed = os.path.normpath(img_path).replace("\\", "/")
        if os.path.exists(img_path_fixed):
            final_images.append(img_path_fixed)

    end_time = time.time()
    logger.info(f"🏆 GANADOR: {mejor_id} (Score: {mejor_score:.4f}) en {end_time - start_time:.4f}s")
    
    return mejor_doc, final_images

# =========================
# ENDPOINTS Y FLUJO API
# =========================
class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = []

@app.on_event("startup")
async def startup_event():
    init_bm25()
    logger.info("🚀 Don Manual está en línea y el motor híbrido cargado.")

@app.get("/health") 
async def health():
    return {"status": "ok", "device": str(device), "hybrid_search": "enabled"}

# =========================
# SYSTEM PROMPT INTEGRADO (NUEVA VERSIÓN)
# =========================
SYSTEM_PROMPT_BASE = """
Eres un experto en agronomía, directo y servicial. Tu función es resolver dudas del agricultor usando datos precisos de manuales técnicos. Habla de forma clara, profesional y sin rodeos.

REGLAS DE ORO:
1. **Idioma**: Responde siempre en español.
2. **Estructura**: Usa exclusivamente listas numeradas (1., 2., 3.) o puntos (•). Prohibidos los párrafos largos.
3. **Resaltado**: Usa **negritas** obligatoriamente para: **plagas**, **productos químicos**, **dosis** y **plazos de seguridad**.
4. **Falta de datos**: Si el manual no indica una dosis, di exactamente: "No dispongo de la dosis exacta para este producto, por favor revise la etiqueta".
5. **Cierre**: Finaliza siempre preguntando si hay alguna otra duda con el cultivo.

### EJEMPLOS DE RESPUESTA:

USUARIO: Tengo unos bichitos negros en los tomates.
RESPUESTA: Para el control de **Pulgón Negro** en tomate, siga estas indicaciones:
1. **Monitoreo**: Revise si hay presencia de hormigas en los brotes.
2. **Opción Ecológica**: Aplique **Jabón Potásico** al **2%** (20 cc por litro de agua).
3. **Opción Química**: Use **Imidacloprid**, respetando el **Plazo de Seguridad** de **3 días** antes de la cosecha.

USUARIO: ¿Qué dosis de cobre uso para el olivo?
RESPUESTA: Para prevenir el repilo en el olivar, actúe de la siguiente forma:
• **Producto**: **Oxicloruro de Cobre 50%**.
• **Dosis**: **300 gramos** por cada **100 litros** de agua (concentración al 0,3%).
• **Seguridad**: No aplicar en días con viento para evitar la deriva del producto.
"""

@app.post("/query_stream")
async def query_stream(req: QueryRequest):
    # 1. Retrieval
    context, image_paths = retrieve_smart(req.question)
    
    # 2. Carga de imágenes (Igual que antes)
    loaded_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((448, 448))
            loaded_images.append(img)
        except Exception as e:
            logger.error(f"Error imagen: {e}")

    # 3. Construcción del Chat
    messages = []
    
    # Insertamos el System Prompt al inicio
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_BASE}]
    })
    
    historial_reciente = req.history[-6:] if req.history else []
    for m in historial_reciente:
        messages.append({
            "role": m["role"],
            "content": [{"type": "text", "text": m["content"]}]
        })

    # 4. Prompt de Usuario con Contexto
    prompt_usuario = (
        f"CONTEXTO TÉCNICO (PDFs):\n{context}\n\n"
        f"PREGUNTA DEL AGRICULTOR: {req.question}\n"
        "RESPUESTA TÉCNICA:"
    )
    
    contenido_actual = [{"type": "image"}] * len(loaded_images)
    contenido_actual.append({"type": "text", "text": prompt_usuario})
    messages.append({"role": "user", "content": contenido_actual})

    # 5. Generación (Igual que antes)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], 
        images=loaded_images if loaded_images else None, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs, 
        "streamer": streamer, 
        "max_new_tokens": MAX_TOKENS, 
        "do_sample": True, 
        "temperature": 0.1, 
        "repetition_penalty": 1.2
    }
    
    Thread(target=model.generate, kwargs=generation_kwargs).start()

    def generate():
        full_response = ""
        # Definimos frases que indican que el modelo no encontró información
        denial_phrases = [
            "lo siento", "no tengo información", "no dispongo", 
            "no se menciona", "consulte a un experto", "no encontré"
        ]
        
        for chunk in streamer:
            if chunk:
                full_response += chunk.lower()
                yield chunk

        # LÓGICA DE FILTRADO DE IMAGEN
        # Solo enviamos la imagen si hay una ruta Y la respuesta no parece una negativa
        if image_paths:
            # Si el modelo admite que no sabe, omitimos la imagen del manual
            is_denial = any(phrase in full_response for phrase in denial_phrases)
            
            if not is_denial:
                yield f"\nIMAGE_PATH: {image_paths[0]}\n"
            else:
                logger.info("🚫 Imagen bloqueada: El modelo dio una respuesta de desconocimiento.")

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)