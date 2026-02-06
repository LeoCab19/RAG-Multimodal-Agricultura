import chromadb
import json
import random
import os

# CONFIGURACIÓN
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_multimodalGrande"
OUTPUT_FILE = "golden_set_manual.jsonl"

def generar_casos():
    # Inicializar cliente de Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error al conectar con la colección: {e}")
        return
    
    # Obtener datos
    data = collection.get()
    ids = data['ids']
    documents = data['documents']
    
    if not ids:
        print("La base de datos está vacía.")
        return

    print(f"\n--- GENERADOR DE GOLDEN SET OPTIMIZADO ---")
    print(f"Total de chunks disponibles: {len(ids)}")
    print("Instrucciones: 's' para saltar, 'salir' para finalizar.\n")

    while True:
        # Selección aleatoria
        idx = random.randint(0, len(ids) - 1)
        chunk_texto = documents[idx]
        # Limpiamos el ID de posibles espacios en blanco
        chunk_id = str(ids[idx]).strip() 

        print("\n" + "═"*60)
        print(f"📍 TRABAJANDO CON ID: {chunk_id}")
        print("-" * 60)
        print(f"CONTENIDO DEL CHUNK:\n{chunk_texto}")
        print("═"*60)

        accion = input("\n¿Usar este chunk? [ENTER para Sí / 's' para Saltar / 'salir']: ").lower()
        
        if accion == 'salir': 
            break
        if accion == 's':
            continue

        # Paso de Pregunta
        pregunta = input(f"\n❓ Escribe la PREGUNTA para el ID [{chunk_id}]:\n> ")
        if pregunta.lower() == 'salir': break
        if pregunta.lower() == 's' or not pregunta: continue 
        
        # Paso de Respuesta (Ground Truth)
        print(f"\n💡 Escribe la RESPUESTA basada únicamente en el texto anterior:")
        respuesta = input("> ")
        if respuesta.lower() == 'salir': break
        if respuesta.lower() == 's' or not respuesta: continue

        # Construcción del objeto JSONL (Aseguramos que el ID vaya en una lista)
        caso = {
            "query": pregunta.strip(),
            "ground_truth": respuesta.strip(),
            "relevant_ids": [chunk_id]
        }

        # Guardado en modo append
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(caso, ensure_ascii=False) + "\n")
            print(f"\n✅ ¡ÉXITO! Caso guardado para el ID: {chunk_id}")
        except Exception as e:
            print(f"❌ Error al guardar: {e}")

    print(f"\nProceso finalizado. Archivo guardado en: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generar_casos()