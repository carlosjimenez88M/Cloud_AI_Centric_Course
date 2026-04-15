"""
role_base — Módulo de Role-Based Prompting con OpenAI

Los 4 Fantásticos + Ricardo Arjona analizan 4 tareas temáticas usando
diferentes personas definidas. Cada persona tiene su estilo, vocabulario
y marco analítico únicos.

Uso:
    from role_base.engine import PersonaEngine
    from role_base.personas import PERSONAS, PERSONA_LABELS
    from role_base.tasks import TASKS
"""

from personas import PERSONAS, PERSONA_LABELS
from tasks import TASKS
from engine import PersonaEngine

__all__ = ["PERSONAS", "PERSONA_LABELS", "TASKS", "PersonaEngine"]
