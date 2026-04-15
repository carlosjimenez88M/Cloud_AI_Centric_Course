"""
Analyst Agent — realiza análisis literario profundo sobre Las Mil y Una Noches.

Responsabilidades:
  1. Recibir la query y el contexto recuperado por el Retriever
  2. Aplicar el ANALYST_PROMPT para generar un análisis académico riguroso
  3. Cubrir dimensiones: narratológica, histórica, filosófica y comparativa

Se activa para queries de tipo "analytical" o "hybrid".
Entrega su output al Supervisor, que luego decidirá si también activa al Creative.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph.state import MilYUnaState
from prompts.templates import ANALYST_PROMPT
from shared.logger import get_logger

log = get_logger("agents.analyst")


def analyst_node(
    state: MilYUnaState,
    llm: BaseChatModel,
) -> dict:
    """
    Nodo Analyst del grafo LangGraph.

    Genera un análisis literario profundo basado en la query del usuario
    y los fragmentos recuperados del ChromaDB.

    Args:
        state: Estado actual del grafo (contiene query y retrieved_context).
        llm:   Modelo de lenguaje (inyectado via functools.partial).

    Returns:
        Diccionario con analysis actualizado y "analyst" añadido a steps_completed.
    """
    query = state["query"]
    context = state.get("retrieved_context", "")

    log.info("🔍 [agent]Analyst[/agent] realizando análisis literario...")

    # Construye el prompt con la query y el contexto recuperado
    prompt = ANALYST_PROMPT.format(
        query=query,
        retrieved_context=context if context else "No hay contexto disponible del texto.",
    )

    # Invoca el LLM para el análisis
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extrae el texto de la respuesta
    analysis = response.content
    if isinstance(analysis, list):
        analysis = " ".join(str(part) for part in analysis)
    analysis = analysis.strip()

    log.info(
        f"  Análisis generado: [dim]{len(analysis)} chars[/dim]"
    )
    log.info("  [success]Analyst completado[/success]")

    # Actualiza la lista de pasos completados
    steps = list(state.get("steps_completed", []))
    if "analyst" not in steps:
        steps.append("analyst")

    return {
        "analysis": analysis,
        "steps_completed": steps,
    }
