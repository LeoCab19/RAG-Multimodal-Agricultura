import os
import fitz  # PyMuPDF
import chromadb
import logging
from PIL import Image
from sentence_transformers import SentenceTransformer
import io
import numpy as np
import torch

# =========================
# CONFIGURACIÓN
# =========================
PDF_DIR = "./pdf"
IMAGE_DIR = "./images"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_multimodalGrande"

# Modelo potente (Dimensión 1024)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 

# --- AJUSTES FILTROS ---
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
UMBRAL_VARIACION = 12
UMBRAL_BRILLO_MIN = 40 
UMBRAL_BRILLO_MAX = 220

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingesta_proximidad_v2")

def es_imagen_valida(img_pil):
    gray = np.array(img_pil.convert("L"))
    std_dev = np.std(gray)
    brillo = gray.mean()
    return std_dev > UMBRAL_VARIACION and UMBRAL_BRILLO_MIN < brillo < UMBRAL_BRILLO_MAX

def procesar_documento(pdf_path, out_dir):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    carpeta_pdf = os.path.join(out_dir, pdf_name)
    os.makedirs(carpeta_pdf, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    datos_finales = []

    for page_num in range(len(doc)): 
        page = doc[page_num]
        lista_imagenes_pagina = []
        
        for item in page.get_images(full=True):
            xref = item[0]
            base = doc.extract_image(xref)
            img_pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
            
            if (img_pil.width < MIN_IMAGE_WIDTH or 
                img_pil.height < MIN_IMAGE_HEIGHT or 
                not es_imagen_valida(img_pil)):
                continue
            
            rects = page.get_image_rects(xref)
            if rects:
                bbox = rects[0]
                y_centro_img = (bbox.y0 + bbox.y1) / 2
                path_img = os.path.join(carpeta_pdf, f"p{page_num+1}_img{xref}.png")
                img_pil.save(path_img)
                lista_imagenes_pagina.append({"path": path_img, "y": y_centro_img})

        page_dict = page.get_text("dict")
        for b_idx, b in enumerate(page_dict["blocks"]):
            if b["type"] == 0: 
                texto_bloque = " ".join([span["text"] for line in b["lines"] for span in line["spans"]]).strip()
                
                if len(texto_bloque) > 30:
                    y_centro_texto = (b["bbox"][1] + b["bbox"][3]) / 2
                    foto_vinculada = "None"
                    if lista_imagenes_pagina:
                        foto_cercana_obj = min(lista_imagenes_pagina, key=lambda x: abs(x["y"] - y_centro_texto))
                        foto_vinculada = foto_cercana_obj["path"]
                    
                    datos_finales.append({
                        "texto": texto_bloque,
                        "page": page_num + 1,
                        "image": foto_vinculada,
                        "id": f"{pdf_name}_p{page_num+1}_b{b_idx}"
                    })
                            
    doc.close()
    return datos_finales

def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try: 
        client.delete_collection(COLLECTION_NAME)
    except: 
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )
    
    logger.info(f"Cargando modelo: {EMBEDDING_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_DIR, file)
            logger.info(f"Procesando {file}...")
            chunks = procesar_documento(path, IMAGE_DIR)
            
            if chunks:
                # ========================================================
                # CAMBIO CLAVE: Prefijo 'passage: ' para el almacenamiento
                # ========================================================
                textos_con_prefijo = [f"passage: {item['texto']}" for item in chunks]
                textos_originales = [item["texto"] for item in chunks] # Guardamos el original en la DB
                
                embeddings = embedder.encode(
                    textos_con_prefijo, # Se genera el vector con el prefijo
                    normalize_embeddings=True, 
                    show_progress_bar=True
                ).tolist()
                
                collection.add(
                    documents=textos_originales, # Se guarda el texto limpio para la respuesta
                    ids=[item["id"] for item in chunks],
                    metadatas=[{
                        "source": file,
                        "page": item["page"],
                        "images": item["image"]
                    } for item in chunks],
                    embeddings=embeddings
                )
    
    logger.info("Base de datos actualizada con prefijos E5 (passage:).")

if __name__ == "__main__":
    main()