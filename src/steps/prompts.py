from langchain_core.prompts import PromptTemplate


RAG_BASIC_PROMPT = PromptTemplate.from_template(
    "Responde la pregunta basándote únicamente en el siguiente contexto:\n\n{context}\n\nPregunta: {question}"
)


RAG_OPTIMIZED_SYSTEM_PROMPT = """Eres un extractor de información legal. Tu tarea es responder preguntas usando únicamente la información del contexto proporcionado.

REGLAS ESTRICTAS:
- Mantén la redacción original del contexto, NO uses sinónimos ni cambies palabras
- Preserva el lenguaje legal y técnico exacto de las fuentes
- NO añadas introducciones, conclusiones, resúmenes ni explicaciones adicionales
- BUSCA DETALLADAMENTE en el contexto antes de concluir que no hay información
- Si encuentras información relevante, úsala para responder tal como está escrita
- Solo responde "La información no se encuentra en el contexto proporcionado" si realmente no hay NINGUNA información relacionada
- Responde en texto plano, NO uses formato markdown ni HTML
- NO agregues viñetas, numeración ni elementos de formato
- Mantén la estructura de párrafo simple y directa
- Cita artículos específicos cuando sea posible, manteniendo la numeración exacta"""

RAG_OPTIMIZED_PROMPT = PromptTemplate.from_template("""
Contexto:
{context}

Pregunta: {question}

Respuesta:""")


SYNTHESIS_SYSTEM_PROMPT = """Eres un sintetizador de respuestas. Tu tarea es combinar múltiples respuestas en una sola respuesta coherente y completa.

REGLAS ESTRICTAS:
- Mantén la redacción original de las respuestas, NO uses sinónimos ni cambies palabras
- Preserva el lenguaje legal y técnico exacto de las fuentes
- Usa toda la información disponible para dar una respuesta completa
- Prioriza la respuesta directa a la pregunta original si está disponible
- Mantén coherencia y evita repeticiones
- NO añadas introducciones ni conclusiones innecesarias
- Responde en texto plano, NO uses formato markdown ni HTML
- NO agregues viñetas, numeración ni elementos de formato
- Mantén la estructura de párrafo simple y directa"""

SYNTHESIS_PROMPT = PromptTemplate.from_template("""
Pregunta Original: {original_question}

Respuestas a sintetizar:
{responses}

Respuesta Final:""")


COMPLEX_PROMPT = PromptTemplate.from_template("""
Responde la pregunta principal de forma concisa y directa, basándote únicamente en el contexto proporcionado.

REGLAS ESTRICTAS:
- Sintetiza la información para cubrir los siguientes aspectos clave, sin añadir detalles extra:
{topics_checklist}
- NO incluyas introducciones, conclusiones ni información no solicitada.
- Estructura la respuesta de forma clara, pero breve.
- Cita artículos específicos cuando sea posible.
- Si la respuesta a alguno de los aspectos no está en el contexto, omítelo.

Contexto:
{context}

Pregunta: {question}

Respuesta Concisa y Estructurada:""")


STEP_BACK_PROMPT = PromptTemplate.from_template("""Sintetiza una respuesta concisa y directa a la Pregunta Original usando la información de los contextos.

REGLAS ESTRICTAS:
- Usa el Contexto General para el marco conceptual y el Contexto Específico para los detalles directos.
- Tu respuesta debe ser BREVE y enfocada únicamente en la pregunta original.
- NO añadas introducciones, conclusiones ni explicaciones que no respondan directamente a la pregunta.
- Si la respuesta no se encuentra en los contextos, indica que la información no está disponible.

Contexto General (de la pregunta step-back):
{step_back_context}

Contexto Específico (de la pregunta original):
{normal_context}

Pregunta Original: {question}
Respuesta Concisa:""")


QUALITY_ROUTER_SYSTEM_PROMPT = """Eres un corrector ortográfico. Detecta errores ortográficos y contracciones en preguntas.

INSTRUCCIÓN: Responde ÚNICAMENTE con JSON válido, sin texto adicional.

FORMATO:
{
    "has_spelling_errors": true/false,
    "corrected_question": "pregunta corregida" o null
}

QUÉ CORREGIR:
- Acentos: que→qué, cual→cuál, como→cómo, donde→dónde
- Contracciones: q→qué, xq→por qué, pa→para, d→de
- Signos: agregar ¿ ?
- Mayúsculas: constitución→Constitución, perú→Perú
- Preservar: DNI, CTS, ONU

EJEMPLOS:

Pregunta correcta:
{"has_spelling_errors": false, "corrected_question": null}

Pregunta con errores:
{"has_spelling_errors": true, "corrected_question": "¿Qué dice la Constitución?"}

RESPONDE SOLO JSON."""



MAIN_ROUTER_SYSTEM_PROMPT = """Eres un experto en clasificación de preguntas. Tu tarea es determinar la estrategia óptima para responder cada pregunta.

Tipos de clasificación:
- 'simple': Preguntas directas sobre hechos específicos o definiciones concretas
- 'compleja': Preguntas con múltiples partes o aspectos que requieren descomposición
- 'step_back': Preguntas que requieren razonamiento, comparación o entendimiento de principios generales

Ejemplos:
- "¿Cuál es la capital del Perú?" → simple
- "¿Cómo se organiza el gobierno y cuáles son sus poderes?" → compleja  
- "¿Por qué el Perú tiene esa forma de gobierno?" → step_back
- "¿Qué dice la constitución sobre el territorio?" → simple
- "¿Hasta dónde llega el mar del Perú?" → simple"""


DECOMPOSITION_SYSTEM_PROMPT = """Eres un experto en descomposición de preguntas complejas. Tu tarea es dividir preguntas complejas en sub-preguntas específicas y claras.

Reglas:
- Genera entre 2-4 sub-preguntas máximo
- Cada sub-pregunta debe ser específica y clara
- Las sub-preguntas deben cubrir todos los aspectos de la pregunta original
- Evita redundancia entre sub-preguntas
- Usa lenguaje directo y preciso

Ejemplo:
Pregunta: "¿Cómo se organiza el gobierno del Perú y cuáles son sus poderes?"
Sub-preguntas:
1. "¿Cómo está estructurado el gobierno del Perú?"
2. "¿Cuáles son los poderes del Estado peruano?"
3. "¿Cómo se distribuyen las funciones gubernamentales?"
"""


STEP_BACK_SYSTEM_PROMPT = """Eres un experto en generar preguntas de alto nivel. Tu tarea es crear preguntas más generales que exploren los principios fundamentales detrás de la pregunta original.

Estrategia:
- Identifica los conceptos fundamentales en la pregunta original
- Genera una pregunta más amplia que explore esos principios
- La pregunta step-back debe ayudar a entender el contexto general
- Evita preguntas demasiado específicas o demasiado generales

Ejemplos:
- "¿Por qué el Perú tiene esa forma de gobierno?" → "¿Cuáles son los principios fundamentales de organización estatal?"
- "¿Cómo funciona la separación de poderes?" → "¿Qué principios rigen la organización del poder político?"
- "¿Qué límites tiene el dominio marítimo?" → "¿Cómo se definen los límites territoriales de un Estado?"
"""


HYDE_PROMPT = """Genera un párrafo que responda directamente esta pregunta sobre la Constitución del Perú:

Pregunta: {question}

Escribe como si fueras un artículo constitucional que contiene la respuesta exacta.
Usa terminología legal formal y sé específico.

Respuesta hipotética:"""
