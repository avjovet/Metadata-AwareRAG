"""Constantes y configuración para el sistema de self-query."""

FILTER_PRIORITIES = {
    "primary": ["document_type", "source"],
    "secondary": ["article_number", "year", "title"],
    "redundant": ["topic"]
}

VALID_VALUES = {
    "document_type": ["constitucion", "decreto", "faq"],
    "source": ["Constitución Política del Perú", "Compendio Derecho Laboral", "Preguntas Frecuentes"],
    "title": {
        "constitucion": [f"Artículo {i}" for i in range(1, 207)],
        "decreto": ["Decreto Legislativo N.° 728", "Decreto Legislativo N.° 713", "Decreto Legislativo N.° 650"],
        "faq": [f"Pregunta Frecuente {i}" for i in range(1, 99)]
    }
}

GENERIC_VALUES = {
    "title": ["Constitución", "Constitución Política", "Decreto", "Decreto Legislativo", "FAQ", "Pregunta"],
    "source": ["Constitución", "Ley", "Decreto", "FAQ"]
}

