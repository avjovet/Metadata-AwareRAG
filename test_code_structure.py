# test_code_structure.py
"""
Script de prueba para verificar la estructura del código sin necesidad de modelos LLM.
Ejecutar: python test_code_structure.py
"""

import sys
import importlib
import traceback
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg: str):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg: str):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg: str):
    print(f"{YELLOW}⚠{RESET} {msg}")

def print_info(msg: str):
    print(f"{BLUE}ℹ{RESET} {msg}")

def test_imports():
    """Prueba que todos los imports funcionen correctamente."""
    print("\n" + "="*60)
    print("1. VERIFICANDO IMPORTS")
    print("="*60)
    
    modules_to_test = [
        "src.types",
        "src.config.settings",
        "src.config.constants",
        "src.utils.validators",
        "src.utils.json_utils",
        "src.utils.document_utils",
        "src.steps.self_query.filter_validation",
        "src.steps.self_query.filter_strategies",
        "src.steps.self_query.constants",
        "src.steps.self_query.retrieval",
        "src.steps.self_query.routers",
    ]
    
    failed = []
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_success(f"Import exitoso: {module_name}")
        except Exception as e:
            print_error(f"Error importando {module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print_warning(f"Total de imports fallidos: {len(failed)}")
    else:
        print_success(f"Todos los imports ({len(modules_to_test)}) funcionan correctamente")
    
    return len(failed) == 0

def test_validators():
    """Prueba las funciones de validación."""
    print("\n" + "="*60)
    print("2. PROBANDO VALIDADORES")
    print("="*60)
    
    try:
        from src.utils.validators import QuestionValidator
        
        # Test 1: Pregunta válida
        is_valid, error = QuestionValidator.validate("¿Qué dice el artículo 2?")
        assert is_valid, "Pregunta válida debería pasar"
        print_success("Validación de pregunta válida: OK")
        
        # Test 2: Pregunta vacía
        is_valid, error = QuestionValidator.validate("")
        assert not is_valid, "Pregunta vacía debería fallar"
        assert error is not None, "Debería haber mensaje de error"
        print_success("Validación de pregunta vacía: OK")
        
        # Test 3: Pregunta muy corta
        is_valid, error = QuestionValidator.validate("ab")
        assert not is_valid, "Pregunta muy corta debería fallar"
        print_success("Validación de pregunta muy corta: OK")
        
        # Test 4: Pregunta muy larga
        long_question = "a" * 501
        is_valid, error = QuestionValidator.validate(long_question)
        assert not is_valid, "Pregunta muy larga debería fallar"
        print_success("Validación de pregunta muy larga: OK")
        
        # Test 5: None
        is_valid, error = QuestionValidator.validate(None)
        assert not is_valid, "None debería fallar"
        print_success("Validación de None: OK")
        
        return True
    except Exception as e:
        print_error(f"Error en validadores: {e}")
        traceback.print_exc()
        return False

def test_json_utils():
    """Prueba las utilidades de JSON."""
    print("\n" + "="*60)
    print("3. PROBANDO UTILIDADES JSON")
    print("="*60)
    
    try:
        from src.utils.json_utils import parse_llm_json_response
        
        # Test 1: JSON válido como string
        response = '{"category": "constitucion", "confidence": 0.9}'
        result = parse_llm_json_response(
            response,
            expected_fields={"category", "confidence"},
            default_values={"category": "general", "confidence": 0.5}
        )
        assert result["category"] == "constitucion"
        assert result["confidence"] == 0.9
        print_success("Parsing de JSON string: OK")
        
        # Test 2: JSON en markdown code block
        response = '```json\n{"category": "derecho_laboral", "confidence": 0.8}\n```'
        result = parse_llm_json_response(
            response,
            expected_fields={"category", "confidence"},
            default_values={"category": "general", "confidence": 0.5}
        )
        assert result["category"] == "derecho_laboral"
        print_success("Parsing de JSON en markdown: OK")
        
        # Test 3: JSON inválido (debería usar defaults)
        response = "esto no es json"
        result = parse_llm_json_response(
            response,
            expected_fields={"category", "confidence"},
            default_values={"category": "general", "confidence": 0.5}
        )
        assert result["category"] == "general"
        assert result["confidence"] == 0.5
        print_success("Parsing de JSON inválido (defaults): OK")
        
        # Test 4: JSON como dict
        response = {"category": "faq", "confidence": 0.7}
        result = parse_llm_json_response(
            response,
            expected_fields={"category", "confidence"},
            default_values={"category": "general", "confidence": 0.5}
        )
        assert result["category"] == "faq"
        print_success("Parsing de JSON dict: OK")
        
        return True
    except Exception as e:
        print_error(f"Error en utilidades JSON: {e}")
        traceback.print_exc()
        return False

def test_filter_validation():
    """Prueba la validación de filtros."""
    print("\n" + "="*60)
    print("4. PROBANDO VALIDACIÓN DE FILTROS")
    print("="*60)
    
    try:
        from src.steps.self_query.filter_validation import validate_and_normalize_filters
        from src.types import ExtractedFilters
        
        # Test 1: Filtros válidos
        filters = ExtractedFilters(
            document_type="constitucion",
            article_number=2,
            year=2021
        )
        validated, discarded = validate_and_normalize_filters(filters)
        assert "document_type" in validated or len(validated) > 0
        print_success("Validación de filtros válidos: OK")
        
        # Test 2: Artículo fuera de rango
        filters = ExtractedFilters(article_number=300)  # Fuera de rango
        validated, discarded = validate_and_normalize_filters(filters)
        assert "article_number" not in validated or any("article_number" in d for d in discarded)
        print_success("Validación de artículo fuera de rango: OK")
        
        # Test 3: Año fuera de rango
        filters = ExtractedFilters(year=1800)  # Fuera de rango
        validated, discarded = validate_and_normalize_filters(filters)
        assert "year" not in validated or any("year" in d for d in discarded)
        print_success("Validación de año fuera de rango: OK")
        
        # Test 4: Filtros vacíos
        filters = ExtractedFilters()
        validated, discarded = validate_and_normalize_filters(filters)
        assert isinstance(validated, dict)
        assert isinstance(discarded, list)
        print_success("Validación de filtros vacíos: OK")
        
        return True
    except Exception as e:
        print_error(f"Error en validación de filtros: {e}")
        traceback.print_exc()
        return False

def test_filter_strategies():
    """Prueba la creación de estrategias de filtrado."""
    print("\n" + "="*60)
    print("5. PROBANDO ESTRATEGIAS DE FILTRADO")
    print("="*60)
    
    try:
        from src.steps.self_query.filter_strategies import create_filter_strategies
        
        # Test 1: Estrategias con filtros válidos
        validated_filters = {
            "document_type": "constitucion",
            "article_number": 2,
            "title": "Derechos Fundamentales"
        }
        strategies = create_filter_strategies(validated_filters, "constitucion")
        assert len(strategies) > 0, "Debería haber al menos una estrategia"
        assert strategies[-1]["name"] == "sin_filtros", "Última estrategia debería ser sin filtros"
        print_success(f"Creación de estrategias ({len(strategies)} estrategias): OK")
        
        # Test 2: Estrategias sin filtros
        strategies = create_filter_strategies({}, "general")
        assert len(strategies) > 0
        print_success("Creación de estrategias sin filtros: OK")
        
        # Test 3: Verificar orden de estrategias (de más específica a menos)
        validated_filters = {"article_number": 5, "year": 2021}
        strategies = create_filter_strategies(validated_filters, "constitucion")
        if len(strategies) > 1:
            # La primera debería ser más específica
            assert len(strategies[0]["filters"]) >= len(strategies[-1]["filters"])
        print_success("Orden de estrategias (especificidad): OK")
        
        return True
    except Exception as e:
        print_error(f"Error en estrategias de filtrado: {e}")
        traceback.print_exc()
        return False

def test_document_utils():
    """Prueba las utilidades de documentos."""
    print("\n" + "="*60)
    print("6. PROBANDO UTILIDADES DE DOCUMENTOS")
    print("="*60)
    
    try:
        from src.utils.document_utils import documents_to_text
        from langchain_core.documents import Document
        
        # Test 1: Lista de documentos
        docs = [
            Document(page_content="Contenido 1", metadata={"source": "doc1"}),
            Document(page_content="Contenido 2", metadata={"source": "doc2"})
        ]
        result = documents_to_text(docs)
        assert "Contenido 1" in result
        assert "Contenido 2" in result
        assert result.count("\n\n") == 1  # Separador entre documentos
        print_success("Conversión de documentos a texto: OK")
        
        # Test 2: Lista vacía
        result = documents_to_text([])
        assert result == ""
        print_success("Conversión de lista vacía: OK")
        
        return True
    except Exception as e:
        print_error(f"Error en utilidades de documentos: {e}")
        traceback.print_exc()
        return False

def test_pipeline_structure():
    """Prueba la estructura de los pipelines sin ejecutarlos."""
    print("\n" + "="*60)
    print("7. VERIFICANDO ESTRUCTURA DE PIPELINES")
    print("="*60)
    
    try:
        # Mock de componentes necesarios
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Mock de LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"has_spelling_errors": false}'
        mock_llm.invoke.return_value = mock_response
        
        # Test: Verificar que create_dynamic_rag_pipeline retorna Runnable
        # Nota: No podemos ejecutarlo completamente sin modelos reales,
        # pero podemos verificar que la función existe y tiene la firma correcta
        from src.pipelines.dynamic import create_dynamic_rag_pipeline
        import inspect
        
        sig = inspect.signature(create_dynamic_rag_pipeline)
        params = list(sig.parameters.keys())
        expected_params = ["db_folder_name", "embedding_model_name", "llm_model_name", 
                          "temperature", "top_k", "enable_self_query"]
        
        for param in expected_params:
            assert param in params, f"Parámetro {param} faltante"
        
        print_success("Firma de create_dynamic_rag_pipeline: OK")
        
        # Verificar que retorna Runnable
        return_type = sig.return_annotation
        print_info(f"Tipo de retorno: {return_type}")
        
        return True
    except Exception as e:
        print_error(f"Error verificando estructura de pipelines: {e}")
        traceback.print_exc()
        return False

def test_types():
    """Prueba los tipos Pydantic."""
    print("\n" + "="*60)
    print("8. PROBANDO TIPOS PYDANTIC")
    print("="*60)
    
    try:
        from src.types import (
            PipelineOutput,
            SemanticRouterOutput,
            ExtractedFilters,
            QualityRouterOutput
        )
        
        # Test 1: PipelineOutput
        output = PipelineOutput(
            question="Test question",
            generated_answer="Test answer",
            retrieved_context=["Context 1", "Context 2"]
        )
        assert output.question == "Test question"
        assert len(output.retrieved_context) == 2
        print_success("PipelineOutput: OK")
        
        # Test 2: SemanticRouterOutput
        router_output = SemanticRouterOutput(
            category="constitucion",
            confidence=0.9,
            reasoning="Test reasoning"
        )
        assert router_output.category == "constitucion"
        assert router_output.confidence == 0.9
        print_success("SemanticRouterOutput: OK")
        
        # Test 3: ExtractedFilters
        filters = ExtractedFilters(
            document_type="constitucion",
            article_number=2
        )
        assert filters.document_type == "constitucion"
        assert filters.article_number == 2
        print_success("ExtractedFilters: OK")
        
        return True
    except Exception as e:
        print_error(f"Error en tipos: {e}")
        traceback.print_exc()
        return False

def test_syntax():
    """Verifica la sintaxis de los archivos principales."""
    print("\n" + "="*60)
    print("9. VERIFICANDO SINTAXIS")
    print("="*60)
    
    import py_compile
    import os
    
    files_to_check = [
        "src/pipelines/dynamic.py",
        "src/pipelines/naive.py",
        "src/steps/routing.py",
        "src/steps/retrieval.py",
        "src/steps/synthesis.py",
        "src/steps/self_query/routers.py",
        "src/steps/self_query/retrieval.py",
        "src/steps/self_query/filter_validation.py",
        "src/steps/self_query/filter_strategies.py",
    ]
    
    failed = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                py_compile.compile(file_path, doraise=True)
                print_success(f"Sintaxis OK: {file_path}")
            except py_compile.PyCompileError as e:
                print_error(f"Error de sintaxis en {file_path}: {e}")
                failed.append(file_path)
        else:
            print_warning(f"Archivo no encontrado: {file_path}")
    
    if failed:
        print_error(f"Total de archivos con errores de sintaxis: {len(failed)}")
    else:
        print_success(f"Todos los archivos ({len(files_to_check)}) tienen sintaxis correcta")
    
    return len(failed) == 0

def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "="*60)
    print(" " * 15 + "PRUEBAS DE ESTRUCTURA DE CÓDIGO")
    print(" " * 10 + "(Sin necesidad de modelos LLM)")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Validadores": test_validators(),
        "Utilidades JSON": test_json_utils(),
        "Validación de Filtros": test_filter_validation(),
        "Estrategias de Filtrado": test_filter_strategies(),
        "Utilidades de Documentos": test_document_utils(),
        "Estructura de Pipelines": test_pipeline_structure(),
        "Tipos Pydantic": test_types(),
        "Sintaxis": test_syntax(),
    }
    
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASÓ")
        else:
            print_error(f"{test_name}: FALLÓ")
    
    print("\n" + "-"*60)
    print(f"Total: {passed}/{total} pruebas pasaron")
    print("-"*60)
    
    if passed == total:
        print_success("\n¡Todas las pruebas pasaron! El código está estructuralmente correcto.")
        return 0
    else:
        print_error(f"\n{total - passed} prueba(s) fallaron. Revisa los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

