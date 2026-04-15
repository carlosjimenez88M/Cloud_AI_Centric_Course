"""
Supervisor Agent — orquestador central del sistema multi-agente.

Responsabilidades:
  1. Clasificar la intención de la query (analytical | creative | hybrid)
  2. Decidir qué agente ejecutar a continuación según el estado actual
  3. Implementar el mecanismo anti-loop (max_iterations)

El Supervisor es el "cerebro" del sistema: todos los demás agentes reportan
de vuelta a él antes de que decida el próximo paso. Esto permite un flujo
dinámico en lugar de una secuencia rígida.

Flujo típico:
  Supervisor → Retriever → Supervisor → Analyst/Creative → Supervisor → Synthesizer
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graph.state import MilYUnaState
from prompts.templates import CLASSIFIER_PROMPT
from shared.logger import get_logger

log = get_logger("agents.supervisor")


def _classify_query(query: str, llm: BaseChatModel) -> str:
    """
    Usa el LLM para clasificar la intención de la query.

    Args:
        query: Pregunta del usuario.
        llm:   Modelo de lenguaje configurado.

    Returns:
        Una de: "analytical", "creative", "hybrid"
    """
    prompt = CLASSIFIER_PROMPT.format(query=query)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extrae el texto de la respuesta y limpia espacios/saltos
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(str(part) for part in raw)
    classification = raw.strip().lower()

    # Valida que la respuesta sea una de las categorías esperadas
    valid = {"analytical", "creative", "hybrid"}
    if classification not in valid:
        # Si el modelo devuelve texto extra, busca la palabra clave en la respuesta
        for cat in valid:
            if cat in classification:
                return cat
        # Fallback seguro: hybrid cubre ambas dimensiones
        log.warning(
            f"Clasificación inesperada: '{classification}' → usando 'hybrid' como fallback"
        )
        return "hybrid"

    return classification


def supervisor_node(
    state: MilYUnaState,
    llm: BaseChatModel,
    max_iterations: int = 6,
) -> dict:
    """
    Nodo Supervisor del grafo LangGraph.

    Primer agente en ejecutarse. Hace dos cosas:
      1. Si query_type está vacío → clasifica la query con el LLM
      2. Decide qué agente llamar a continuación según el estado actual

    Args:
        state:          Estado actual del grafo.
        llm:            Modelo de lenguaje (inyectado via functools.partial).
        max_iterations: Límite de ciclos para prevenir bucles infinitos.

    Returns:
        Diccionario con los campos del estado a actualizar.
    """
    log.info("🎯 [agent]Supervisor[/agent] evaluando estado del grafo...")

    # ── Anti-loop: si superamos el límite de iteraciones, forzamos síntesis ──
    current_iterations = state.get("iteration_count", 0)
    if current_iterations >= max_iterations:
        log.warning(
            f"  Límite de iteraciones alcanzado ({max_iterations}) → "
            "forzando síntesis final"
        )
        return {
            "next_agent": "synthesizer",
            "iteration_count": current_iterations + 1,
        }

    # ── PASO 1: Clasificar la query si aún no tiene tipo ─────────────────────
    if not state.get("query_type"):
        log.info("  Clasificando intención de la query con LLM...")
        query_type = _classify_query(state["query"], llm)
        log.info(f"  Tipo detectado: [highlight]{query_type}[/highlight]")
        return {
            "query_type": query_type,
            "next_agent": "retriever",   # Siempre arrancamos con el retriever
            "steps_completed": [],       # Reinicia los pasos completados
            "iteration_count": 0,        # Reinicia el contador
        }

    # ── PASO 2: Routing dinámico basado en el estado actual ───────────────────
    steps = state.get("steps_completed", [])
    query_type = state["query_type"]

    log.info(
        f"  Query type: [highlight]{query_type}[/highlight]  |  "
        f"Pasos completados: {steps}"
    )

    # Lógica de routing:
    # 1. Si el retriever no ha corrido → retriever (siempre primero)
    # 2. Si es analytical/hybrid y no ha corrido el analyst → analyst
    # 3. Si es creative/hybrid y no ha corrido el creative → creative
    # 4. Si ya corrieron todos los agentes necesarios → synthesizer

    if "retriever" not in steps:
        next_agent = "retriever"

    elif query_type in ("analytical", "hybrid") and "analyst" not in steps:
        next_agent = "analyst"

    elif query_type in ("creative", "hybrid") and "creative" not in steps:
        next_agent = "creative"

    elif query_type == "analytical" and "analyst" not in steps:
        # Fallback para analytical sin analyst
        next_agent = "analyst"

    elif query_type == "creative" and "creative" not in steps:
        # Fallback para creative sin creative
        next_agent = "creative"

    elif "analyst" not in steps and "creative" not in steps:
        # Fallback genérico: si no hay análisis ni creativo, hacer análisis
        next_agent = "analyst"

    else:
        # Todos los agentes necesarios ya ejecutaron → sintetizar
        next_agent = "synthesizer"

    log.info(f"  → Próximo agente: [node]{next_agent}[/node]")

    return {
        "next_agent": next_agent,
        "iteration_count": current_iterations + 1,
    }
