"""
Retriever Agent — busca en ChromaDB y sintetiza los fragmentos recuperados.

Responsabilidades:
  1. Ejecutar búsqueda semántica en el ChromaDB de Las Mil y Una Noches
  2. Formatear los fragmentos recuperados con metadata (número de página)
  3. Opcionalmente resumir el contexto si es muy extenso

El Retriever es siempre el primer agente especializado que se ejecuta,
porque todos los demás agentes necesitan el contexto del libro para trabajar.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph.state import MilYUnaState
from prompts.templates import RETRIEVER_SUMMARY_PROMPT
from shared.logger import get_logger

log = get_logger("agents.retriever")

# Umbral de caracteres a partir del cual se resume el contexto con el LLM.
# Si el contexto crudo supera este límite, se pide al LLM que lo sintetice.
_CONTEXT_SUMMARIZE_THRESHOLD = 3000


def _format_documents(docs: list[Any]) -> str:
    """
    Formatea la lista de documentos recuperados en un string estructurado.

    Cada fragmento incluye:
      - Número secuencial
      - Número de página (del metadata si está disponible)
      - Texto del fragmento (máx 500 caracteres por legibilidad)

    Args:
        docs: Lista de documentos de LangChain (Document objects).

    Returns:
        String con todos los fragmentos formateados.
    """
    if not docs:
        return "No se encontraron fragmentos relevantes en el libro."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        # Truncamos a 500 chars para evitar contextos demasiado largos
        text = doc.page_content[:500].strip()
        parts.append(f"[Fragmento {i} — Pág. {page}]\n{text}")

    return "\n\n---\n\n".join(parts)


def retriever_node(
    state: MilYUnaState,
    retriever: Any,
    llm: BaseChatModel,
) -> dict:
    """
    Nodo Retriever del grafo LangGraph.

    Busca en ChromaDB los fragmentos más relevantes para la query del usuario
    y los formatea como contexto para los agentes siguientes.

    Args:
        state:     Estado actual del grafo.
        retriever: LangChain retriever conectado al ChromaDB (inyectado via partial).
        llm:       Modelo de lenguaje para resumir si el contexto es muy largo.

    Returns:
        Diccionario con retrieved_context actualizado y "retriever" añadido a steps_completed.
    """
    query = state["query"]
    log.info(f"📚 [agent]Retriever[/agent] buscando en ChromaDB...")
    log.info(f"  Query: [dim]{query[:80]}...[/dim]" if len(query) > 80 else f"  Query: [dim]{query}[/dim]")

    # Ejecuta la búsqueda semántica en ChromaDB
    docs = retriever.invoke(query)
    log.info(f"  {len(docs)} fragmento(s) recuperado(s):")
    for i, doc in enumerate(docs[:3], 1):   # muestra preview de los 3 primeros
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace("\n", " ").strip()
        log.info(f"    [{i}] Pág.{page}: [dim]{preview}...[/dim]")

    # Formatea los documentos con metadata
    raw_context = _format_documents(docs)

    # Si el contexto es demasiado largo, lo resumimos con el LLM
    # Esto evita que los tokens de contexto consuman demasiado del presupuesto
    if len(raw_context) > _CONTEXT_SUMMARIZE_THRESHOLD:
        log.info(
            f"  Contexto extenso ({len(raw_context)} chars) → resumiendo con LLM..."
        )
        summary_prompt = RETRIEVER_SUMMARY_PROMPT.format(
            query=query,
            raw_fragments=raw_context,
        )
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        context = response.content
        if isinstance(context, list):
            context = " ".join(str(part) for part in context)
        log.info(f"  Resumen generado: {len(context)} chars")
    else:
        # El contexto es manejable, lo usamos directamente
        context = raw_context

    # Añade "retriever" a la lista de pasos completados
    steps = list(state.get("steps_completed", []))
    if "retriever" not in steps:
        steps.append("retriever")

    log.info("  [success]Retriever completado[/success]")

    return {
        "retrieved_context": context,
        "steps_completed": steps,
    }
