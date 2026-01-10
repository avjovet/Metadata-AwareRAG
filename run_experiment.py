# run_experiment.py

import json
from typing import List, Dict, Any

from src.pipelines import NaiveRAGPipeline

def run_experiment(pipeline: NaiveRAGPipeline, dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Ejecuta un pipeline RAG sobre un conjunto de datos y recopila los resultados.
    """
    if not dataset:
        print("ADVERTENCIA: El dataset de evaluación está vacío.")
        return []
        
    results = []
    print(f"\n--- Iniciando Experimento con {len(dataset)} preguntas ---")
    
    for i, item in enumerate(dataset):
        question = item.get("question")
        if not question:
            print(f"ADVERTENCIA: Item {i+1} del dataset no tiene 'question'. Saltando.")
            continue

        print(f"Procesando Q{i+1}: {question}")
        
        # El pipeline devuelve un diccionario estructurado
        output = pipeline.invoke(question)
        
        result_item = {
            "question": question,
            "expected_answer": item.get("expected_answer"),
            "generated_answer": output.get("generated_answer"),
            "retrieved_context": output.get("retrieved_context", [])
        }
        results.append(result_item)

    print("--- Experimento Finalizado ---")
    return results

def main():
    """
    Script principal para ejecutar un experimento con un pipeline RAG.
    """
    # 1. Inicializa el pipeline a probar
    try:
        pipeline_to_test = NaiveRAGPipeline()
    except FileNotFoundError as e:
        print(f"\nERROR CRÍTICO: {e}")
        print("Asegúrate de haber creado la base de datos vectorial antes de ejecutar el experimento.")
        print("Puedes hacerlo ejecutando: python -m src.indexing_logic")
        return

    # 2. Define un conjunto de datos para el experimento.
    print("\nDefiniendo el dataset para el experimento...")
    experiment_dataset = [
        {
            "question": "¿Qué es el régimen laboral privado?",
            "expected_answer": "Es el régimen laboral común aplicable a la mayoría de los trabajadores en el sector privado."
        },
        {
            "question": "¿Cuáles son las vacaciones para un trabajador del regimen privado?",
            "expected_answer": "30 días calendario por cada año completo de servicios."
        },
        {
            "question": "Háblame sobre la CTS.",
            "expected_answer": "La Compensación por Tiempo de Servicios (CTS) es un beneficio social que protege al trabajador de las contingencias del cese laboral."
        }
    ]

    # 3. Ejecuta el experimento.
    results = run_experiment(pipeline_to_test, experiment_dataset)

    # 4. Muestra los resultados de una forma legible.
    print("\n\n--- Resultados del Experimento ---")
    if not results:
        print("No se obtuvieron resultados del experimento.")
        return
        
    for i, result in enumerate(results):
        print(f"\n----- Pregunta {i+1} -----")
        print(f"Pregunta: {result['question']}")
        print(f"Respuesta Esperada: {result['expected_answer']}")
        print(f"Respuesta Generada: {result['generated_answer']}")
        print("\nContexto Recuperado:")
        if result.get('retrieved_context'):
            for j, context_doc in enumerate(result['retrieved_context']):
                # Imprime solo los primeros 300 caracteres del contexto para brevedad.
                print(f"  --- Contexto {j+1} ---\n{context_doc[:300]}...\n")
        else:
            print("  No se recuperó contexto.")
        print("--------------------------")

    # 5. Guarda los resultados en un archivo JSON para un análisis más detallado.
    output_filename = "experiment_results.json"
    print(f"\nGuardando resultados detallados en '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Resultados guardados. Puedes revisar el archivo '{output_filename}'.")
    

if __name__ == "__main__":
    main()
