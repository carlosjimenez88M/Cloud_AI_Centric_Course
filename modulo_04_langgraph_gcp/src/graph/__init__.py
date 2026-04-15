"""Graph module — define el estado y el workflow del sistema multi-agente."""
from .state import MilYUnaState

# workflow se importa de forma lazy para evitar errores cuando las
# dependencias opcionales (langchain-google-vertexai) no están instaladas.
# Usa: from graph.workflow import create_workflow, run_demo_queries

__all__ = ["MilYUnaState"]
