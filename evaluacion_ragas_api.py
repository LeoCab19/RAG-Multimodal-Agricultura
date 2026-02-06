import json
import os
import re
import torch
import logging
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from rank_bm25 import BM25Okapi

# ============================================================================
# 1. CONFIGURACIÓN Y CARGA DE MODELOS
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUADOR_COMPLETO")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_multimodalGrande"
GOLDEN_SET_PATH = "./golden_set_manual.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"🚀 Iniciando motor en: {DEVICE}")

# Modelos de Recuperación
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)

# Modelo VLM para Generación y Juez
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Base de Datos
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

# Inicializar BM25
data = collection.get()
all_docs_text = data["documents"]
all_ids = data["ids"]
tokenized_corpus = [doc.lower().split() for doc in all_docs_text]
bm25_index = BM25Okapi(tokenized_corpus)

# ============================================================================
# 2. FUNCIONES DE LÓGICA RAG
# ============================================================================

def motor_busqueda(question):
    # Vectorial
    q_emb = embedder.encode([f"query: {question}"], normalize_embeddings=True).tolist()
    res_v = collection.query(query_embeddings=q_emb, n_results=10)
    v_ids = res_v["ids"][0]
    
    # BM25
    tokenized_query = question.lower().split()
    bm25_ids = bm25_index.get_top_n(tokenized_query, all_ids, n=10)
    
    # RRF (Fusión)
    scores = {}
    for r, idx in enumerate(v_ids): scores[idx] = scores.get(idx, 0) + 1/(r + 60)
    for r, idx in enumerate(bm25_ids): scores[idx] = scores.get(idx, 0) + 1/(r + 60)
    rrf_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Rerank
    lookup = {idx: txt for idx, txt in zip(all_ids, all_docs_text)}
    candidates = [lookup[idx] for idx in rrf_ids if idx in lookup]
    pairs = [[question, doc] for doc in candidates]
    scores_rerank = reranker.predict(pairs)
    
    scored = sorted(zip(scores_rerank, candidates, rrf_ids), key=lambda x: x[0], reverse=True)
    return scored[0][1], [s[2] for s in scored] # (Mejor doc, Lista de IDs)

def preguntar_juez(prompt):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Eres un juez técnico. Responde SOLO con el número solicitado."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    res = processor.batch_decode(out, skip_special_tokens=True)[0]
    nums = re.findall(r'\d+', res)
    return int(nums[-1]) if nums else 0

# ============================================================================
# 3. EJECUCIÓN DE LA EVALUACIÓN
# ============================================================================

def ejecutar_evaluacion():
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: No se encuentra {GOLDEN_SET_PATH}")
        return

    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        casos = [json.loads(line) for line in f]

    resultados = []
    
    for caso in tqdm(casos, desc="Evaluando"):
        query = caso['query']
        ids_esperados = [str(i).strip() for i in caso.get('relevant_ids', [])]

        # RAG
        contexto, ids_recuperados = motor_busqueda(query)
        ids_recuperados = [str(i).strip() for i in ids_recuperados]

        # Generar respuesta para evaluar
        p_gen = f"Contexto: {contexto}\nPregunta: {query}\nRespuesta técnica:"
        in_gen = processor(text=[p_gen], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_gen = model.generate(**in_gen, max_new_tokens=150)
        respuesta_ia = processor.batch_decode(out_gen, skip_special_tokens=True)[0]

        # MÉTRICAS
        # Fidelidad (0 o 1)
        fid = preguntar_juez(f"¿La respuesta alucina? Responde 1 si es TOTALMENTE fiel al contexto, 0 si inventa datos.\nContexto: {contexto}\nRespuesta: {respuesta_ia}")
        fid = 1 if fid >= 1 else 0 # Forzar binario
        
        # Relevancia (1 a 5)
        rel = preguntar_juez(f"Del 1 al 5, ¿qué tan útil es esta respuesta para el agricultor?\nRespuesta: {respuesta_ia}")
        rel = max(1, min(5, rel)) # Asegurar rango 1-5

        # Recall
        encontrados = sum(1 for rid in ids_esperados if rid in ids_recuperados)
        rec = encontrados / len(ids_esperados) if ids_esperados else 0

        resultados.append({"fid": fid, "rel": rel, "rec": rec})

    # CÁLCULOS FINALES
    total = len(resultados)
    avg_fid = (sum(r['fid'] for r in resultados) / total) * 100
    avg_rel = sum(r['rel'] for r in resultados) / total
    avg_rec = (sum(r['rec'] for r in resultados) / total) * 100

    print("\n" + "═════════════════════════════════════════════")
    print("           RESULTADOS DE EVALUACIÓN")
    print("═════════════════════════════════════════════")
    print(f"✅ Fidelidad (No alucinación): {avg_fid:>6.1f}%")
    print(f"🎯 Relevancia (Utilidad):       {avg_rel:>6.2f}/5")
    print(f"🔍 Recall (Precisión Búsqueda): {avg_rec:>6.1f}%")
    print("═════════════════════════════════════════════")

if __name__ == "__main__":
    ejecutar_evaluacion()

