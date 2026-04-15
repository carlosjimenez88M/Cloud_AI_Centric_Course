"""
Creative Agent — genera contenido creativo inspirado en Las Mil y Una Noches.

Responsabilidades:
  1. Generar poemas, canciones (estilo Arjona), cuentos breves y reinterpretaciones
  2. Usar el contexto recuperado por el Retriever como inspiración
  3. Fusionar el imaginario árabe/oriental con referencias culturales latinoamericanas

Se activa para queries de tipo "creative" o "hybrid".
Ricardo Arjona es la referencia musical/poética para la audiencia latinoamericana:
metáforas inesperadas, imágenes cotidianas elevadas, melancolía filosófica.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph.state import MilYUnaState
from prompts.templates import CREATIVE_PROMPT
from shared.logger import get_logger

log = get_logger("agents.creative")


def creative_node(
    state: MilYUnaState,
    llm: BaseChatModel,
) -> dict:
    """
    Nodo Creative del grafo LangGraph.

    Genera contenido creativo (poemas, canciones, cuentos breves) inspirado
    en Las Mil y Una Noches, con el estilo lírico de Ricardo Arjona.

    Args:
        state: Estado actual del grafo (contiene query y retrieved_context).
        llm:   Modelo de lenguaje (inyectado via functools.partial).

    Returns:
        Diccionario con creative_content actualizado y "creative" añadido a steps_completed.
    """
    query = state["query"]
    context = state.get("retrieved_context", "")

    log.info("🎨 [agent]Creative[/agent] generando contenido creativo...")

    # Construye el prompt con la solicitud creativa y el contexto del libro
    prompt = CREATIVE_PROMPT.format(
        query=query,
        retrieved_context=context if context else "Las Mil y Una Noches — imaginario árabe clásico.",
    )

    # Invoca el LLM para la creación
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extrae el texto de la respuesta
    creative_content = response.content
    if isinstance(creative_content, list):
        creative_content = " ".join(str(part) for part in creative_content)
    creative_content = creative_content.strip()

    log.info(
        f"  Contenido creativo generado: [dim]{len(creative_content)} chars[/dim]"
    )
    log.info("  [success]Creative completado[/success]")

    # Actualiza la lista de pasos completados
    steps = list(state.get("steps_completed", []))
    if "creative" not in steps:
        steps.append("creative")

    return {
        "creative_content": creative_content,
        "steps_completed": steps,
    }
