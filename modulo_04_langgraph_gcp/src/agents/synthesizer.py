"""
Synthesizer Agent — nodo final que combina todos los outputs en la respuesta definitiva.

Responsabilidades:
  1. Integrar el análisis literario (Analyst) y el contenido creativo (Creative)
  2. Estructurar una respuesta coherente, fluida y bien formateada
  3. Asegurar que la respuesta final responde directamente la pregunta del usuario

El Synthesizer es el último nodo antes de END en el grafo.
Es análogo al narrador Scheherazade: teje todos los hilos en un relato unificado.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph.state import MilYUnaState
from prompts.templates import SYNTHESIZER_PROMPT
from shared.logger import get_logger

log = get_logger("agents.synthesizer")


def synthesizer_node(
    state: MilYUnaState,
    llm: BaseChatModel,
) -> dict:
    """
    Nodo Synthesizer del grafo LangGraph.

    Combina el análisis literario, el contenido creativo y el contexto recuperado
    en una respuesta final cohesiva que responde la pregunta del usuario.

    Args:
        state: Estado actual del grafo (contiene todos los outputs previos).
        llm:   Modelo de lenguaje (inyectado via functools.partial).

    Returns:
        Diccionario con final_answer actualizado.
    """
    query = state["query"]
    query_type = state.get("query_type", "hybrid")
    analysis = state.get("analysis", "")
    creative_content = state.get("creative_content", "")
    retrieved_context = state.get("retrieved_context", "")

    log.info("✍️  [agent]Synthesizer[/agent] construyendo respuesta final...")
    log.info(
        f"  Inputs disponibles: "
        f"análisis={'sí' if analysis else 'no'}, "
        f"creativo={'sí' if creative_content else 'no'}, "
        f"contexto={'sí' if retrieved_context else 'no'}"
    )

    # Construye el prompt de síntesis con todos los outputs previos
    prompt = SYNTHESIZER_PROMPT.format(
        query=query,
        query_type=query_type,
        analysis=analysis if analysis else "(no se realizó análisis literario)",
        creative_content=creative_content if creative_content else "(no se generó contenido creativo)",
        retrieved_context=retrieved_context if retrieved_context else "(no hay contexto recuperado del libro)",
    )

    # Invoca el LLM para la síntesis final
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extrae el texto de la respuesta
    final_answer = response.content
    if isinstance(final_answer, list):
        final_answer = " ".join(str(part) for part in final_answer)
    final_answer = final_answer.strip()

    log.info(
        f"  Respuesta final generada: [dim]{len(final_answer)} chars[/dim]"
    )
    log.info("  [success]Synthesizer completado — Pipeline finalizado[/success]")

    # Agrega "synthesizer" a los pasos completados
    steps = list(state.get("steps_completed", []))
    if "synthesizer" not in steps:
        steps.append("synthesizer")

    return {
        "final_answer": final_answer,
        "steps_completed": steps,
    }
