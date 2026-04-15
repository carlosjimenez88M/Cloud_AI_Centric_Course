"""
orchestration — Orquestación profunda con LangGraph + LangChain + OpenAI

Flujo del pipeline:
  Pregunta
    └─► [Router]           → clasifica: story | character | philosophy | creative
         └─► [Retriever]   → recupera fragmentos del corpus
              └─► [Chain]  → análisis específico según ruta
                   └─► [Synthesizer] → respuesta final enriquecida

Módulos:
  orchestration.prompts  — Plantillas de prompts por cadena
  orchestration.chains   — Cadenas LangChain por tipo de análisis
  orchestration.router   — Router LLM que decide la ruta
  orchestration.graph    — Grafo LangGraph que orquesta todo
"""

from graph import OrchestrationGraph, OrchestrationState

__all__ = ["OrchestrationGraph", "OrchestrationState"]
