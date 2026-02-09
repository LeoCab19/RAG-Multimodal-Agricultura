import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# Configuración básica igual a tu script
PDF_DIR = "./pdf"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
client = chromadb.PersistentClient(path=CHROMA_DIR)

def split_text_by_chars(text, size, overlap):
    """Divide el texto en fragmentos de tamaño fijo."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

def ingesta_con_configuracion(name, chunk_size, chunk_overlap):
    print(f"--- Creando colección: {name} (Size: {chunk_size}) ---")
    try: client.delete_collection(name)
    except: pass
    
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(PDF_DIR, file))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            # Aplicamos el chunking controlado
            text_chunks = split_text_by_chars(full_text, chunk_size, chunk_overlap)
            
            # Generar Embeddings e Insertar
            if text_chunks:
                prefixes = [f"passage: {c}" for c in text_chunks]
                embeddings = embedder.encode(prefixes, normalize_embeddings=True).tolist()
                collection.add(
                    documents=text_chunks,
                    ids=[f"{file}_{i}" for i in range(len(text_chunks))],
                    embeddings=embeddings
                )
            doc.close()

# 1. GENERAMOS LAS 3 CONFIGURACIONES
configs = [
    {"name": "eval_pequeno", "size": 256, "overlap": 50},
    {"name": "eval_medio", "size": 512, "overlap": 100},
    {"name": "eval_grande", "size": 1024, "overlap": 200}
]

for c in configs:
    ingesta_con_configuracion(c["name"], c["size"], c["overlap"])