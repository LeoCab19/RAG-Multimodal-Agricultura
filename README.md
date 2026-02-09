# Asistente Agrícola RAG Multimodal

Sistema de consulta técnica agrícola con búsqueda híbrida, re-ranking y generación multimodal usando VLM.

## Características

- **Búsqueda Híbrida**: Vectorial (E5) + BM25 fusionados con RRF
- **Re-ranking**: Cross-encoder para máxima precisión
- **Multimodal**: Vincula imágenes con texto por proximidad espacial
- **Streaming**: Respuestas en tiempo real con memoria conversacional
- **Evaluación**: Métricas de Fidelidad, Relevancia y Recall

## Stack

**Modelos**: `intfloat/multilingual-e5-large` (embeddings) • `BAAI/bge-reranker-v2-m3` (reranker) • `Qwen2-VL-2B` (generación)  
**Backend**: FastAPI + ChromaDB + BM25  
**Frontend**: Streamlit

## Instalación

```bash
# Dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Estructura
mkdir -p pdf images chroma_db

# Variables de entorno (.env)
VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
MODELO_RERANKER=BAAI/bge-reranker-v2-m3
TOP_K=10
MAX_TOKENS_LLM=10000
```

## Uso

### 1. Ingestar PDFs
```bash
# Coloca PDFs en ./pdf/
python chunkingV5ImagenMejorado.py
```
Extrae texto, filtra imágenes (>100px, brillo 40-220, variación >12) y genera embeddings con prefijo `passage:`.

### 2. Iniciar API
```bash
python api_rag_multimodal1.py  # http://127.0.0.1:8000
```

### 3. Ejecutar UI
```bash
streamlit run streamli_front.py  # http://localhost:8501
```

## Arquitectura

```
Usuario → Query Expansion → [Búsqueda Vectorial + BM25] → RRF → 
Re-ranking → Mejor Doc + Imagen → VLM (streaming) → Respuesta
```

**Optimizaciones clave**:
- Caché LRU de ChromaDB (`@lru_cache`)
- Índice BM25 pre-cargado en RAM (startup)
- Prefijos E5: `query:` para búsquedas, `passage:` para docs

### Evaluación de Recuperación (Retrieval)
Para garantizar la precisión del asistente, se evaluaron tres configuraciones de segmentación de documentos (chunking). El objetivo fue maximizar la relevancia del primer resultado recuperado.

Primero se ha tenido que utiliza un script para hacer el chunk pequeño y el chunk mediano.
Hemos utilizado el siguiente script:
```bash
python preparan_evaluacion.py
```
Despues para el resultado se tiene que ejecutar el siguiente script:

```bash
python evaluador_retrivel.py
```
### Tabla Comparativa de Configuraciones RAG

| Configuración | Tokens | Overlap | Hit Rate @3 | MRR @3 | Latencia |
|---------------|--------|---------|-------------|--------|----------|
| **Pequeño** | 256 | 20 | 1.0 | 1.0 | Baja |
| **Medio** | 512 | 50 | 1.0 | 1.0 | Media |
| **Grande** | 1024 | 100 | 1.0 | 0.5 | Alta |

**Recomendación**: Configuración **Media** (512 tokens, overlap 50) ofrece el mejor balance precisión/velocidad.

## Evaluación

### Crear Golden Set
```bash
python generar_evaluador.py
```
Genera casos de prueba: pregunta + respuesta esperada + IDs relevantes.

### Ejecutar Tests
```bash
python evaluacion_ragas_api.py
```

**Métricas**:
- **Fidelidad** (0-100%): Sin alucinaciones (meta >90%)
- **Relevancia** (1-5): Utilidad práctica (meta >4.0)
- **Recall** (0-100%): Documentos recuperados (meta >85%)

Ejemplo:
```
✅ Fidelidad:   90.0%
🎯 Relevancia:  1.80/5
🔍 Recall:      100.0%
```

##  Configuración

### Ajustar Rendimiento
```python
# api_rag_multimodal1.py
TOP_K = 10              # Candidatos (5-20)
temperature = 0.1       # Creatividad (0.05-0.7)
max_new_tokens = 10000  # Longitud máxima

# RRF k=60 (balance), k=20 (agresivo), k=100 (conservador)
```

### CPU (sin GPU)
```python
device = "cpu"
torch_dtype = torch.float32  # No float16
TOP_K = 5  # Reducir carga
```
### Tambien se puede utilizar con GPU

## Formato de Datos

**ChromaDB**:
```json
{
  "id": "manual_p23_b5",
  "document": "El pulgón negro...",
  "metadata": {"source": "manual.pdf", "page": 23, "images": "./images/p23.png"},
  "embedding": [...]  // 1024 dims
}
```

**Golden Set** (JSONL):
```json
{"query": "¿Control de pulgón?", "ground_truth": "Jabón 2%...", "relevant_ids": ["manual_p23_b5"]}
```

## Troubleshooting

| Error | Solución |
|-------|----------|
| Collection not found | `python chunkingV5ImagenMejorado.py` |
| CUDA OOM | Reducir `TOP_K=5` o `MAX_TOKENS_LLM=500` |
| API lenta | Índice BM25 cargado? Revisar logs de caché |
| Baja fidelidad | `temperature=0.05`, fortalecer system prompt |
| Bajo recall | Verificar prefijos E5, aumentar `TOP_K=20` |

## Flujo de Búsqueda Híbrida

1. **Query Expansion**: `"araña"` → `"Información técnica sobre... araña"`
2. **Vectorial**: Embedding query → ChromaDB (cosine similarity)
3. **BM25**: Tokenización → Índice invertido (TF-IDF)
4. **RRF**: Fusión con `score = Σ[1/(60+rank)]`
5. **Rerank**: Cross-encoder evalúa pares [query, doc]
6. **Mejor**: Top-1 + imagen vinculada → Prompt VLM




