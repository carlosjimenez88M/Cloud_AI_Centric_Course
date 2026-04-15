"""
Workflow — construye y ejecuta el grafo LangGraph multi-agente.

Topología del grafo:
  START
    │
    ▼
  [supervisor]  ← Clasifica la query y decide el siguiente agente
    │
    ├──► [retriever]  → vuelve a supervisor
    ├──► [analyst]    → vuelve a supervisor
    ├──► [creative]   → vuelve a supervisor
    └──► [synthesizer] → END

El Supervisor actúa como hub central: después de cada agente especializado,
el flujo regresa al Supervisor para decidir el siguiente paso.
Esto permite un routing dinámico en lugar de una secuencia fija.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import warnings as _warnings

from langchain_chroma import Chroma
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph

from agents.supervisor import supervisor_node
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.creative import creative_node
from agents.synthesizer import synthesizer_node
from graph.state import MilYUnaState
from shared.logger import get_logger, console

log = get_logger("graph.workflow")


# ── Queries de demostración ───────────────────────────────────────────────────
# Tres queries que ejercitan los tres tipos del sistema:
#   1. analytical  — análisis de personaje/tema
#   2. creative    — generación poética
#   3. hybrid      — análisis + creación en una sola pregunta

DEMO_QUERIES: list[str] = [
    # Tipo: analytical — activa Retriever + Analyst + Synthesizer
    "¿Quién es Scheherazade y cuál es su importancia en la literatura mundial?",

    # Tipo: creative — activa Retriever + Creative + Synthesizer
    "Escribe un haiku al estilo Arjona inspirado en la historia de Aladino",

    # Tipo: hybrid — activa Retriever + Analyst + Creative + Synthesizer
    "Analiza el personaje del rey Shahryar y escribe un poema sobre su redención",
]


# ── Función para crear el estado inicial ─────────────────────────────────────

def _make_initial_state(query: str) -> MilYUnaState:
    """
    Crea el estado inicial del grafo para una nueva query.

    Todos los campos comienzan vacíos/en cero; el Supervisor los irá
    llenando a medida que delega a los agentes especializados.
    """
    return MilYUnaState(
        query=query,
        query_type="",
        retrieved_context="",
        analysis="",
        creative_content="",
        final_answer="",
        messages=[],
        steps_completed=[],
        iteration_count=0,
        next_agent="",
    )


# ── Función principal: construye el grafo ────────────────────────────────────

def create_workflow(
    cfg: dict[str, Any],
    project_id: str,
    location: str,
    vector_store: Chroma,
) -> Any:
    """
    Construye y compila el grafo LangGraph del sistema multi-agente.

    Pasos:
      1. Inicializa el LLM (ChatVertexAI con Gemini Flash Lite)
      2. Configura el retriever con el ChromaDB existente
      3. Crea funciones parciales para inyectar dependencias en los nodos
      4. Define el StateGraph con sus nodos y edges
      5. Compila y retorna el grafo listo para invocar

    Args:
        cfg:          Configuración cargada desde config.yaml.
        project_id:   ID del proyecto GCP.
        location:     Región de Vertex AI.
        vector_store: ChromaDB cargado desde modulo_03_gcp.

    Returns:
        Grafo compilado (CompiledStateGraph), invocable con .invoke(state).
    """
    model_cfg = cfg["model"]
    log.info("Construyendo sistema multi-agente...")

    # ── 1. Inicializa el LLM (Gemini Flash Lite via Vertex AI) ───────────────
    log.info(f"  Inicializando LLM: [cyan]{model_cfg['name']}[/cyan]")
    # Suprimir warning de clase deprecated — ChatVertexAI usa Vertex AI (ADC),
    # no la Gemini API directa. Mientras el proyecto no habilite
    # generativelanguage.googleapis.com, esta es la clase correcta.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        llm = ChatVertexAI(
            model_name=model_cfg["name"],
            project=project_id,
            location=location,
            temperature=model_cfg.get("temperature", 0.3),
            max_output_tokens=model_cfg.get("max_output_tokens", 3000),
        )

    # ── 2. Configura el retriever sobre el ChromaDB ───────────────────────────
    top_k = cfg["rag"]["top_k"]
    log.info(f"  Configurando retriever (top_k={top_k})")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # ── 3. Límite de iteraciones anti-loop ────────────────────────────────────
    max_iter = cfg["multi_agent"]["max_iterations"]
    log.info(f"  Max iteraciones: {max_iter}")

    # ── 4. Inyección de dependencias con functools.partial ────────────────────
    # LangGraph espera que los nodos sean funciones que toman solo `state`.
    # Usamos partial para "pre-aplicar" el llm, retriever, etc., dejando
    # solo `state` como argumento libre.
    supervisor = partial(supervisor_node, llm=llm, max_iterations=max_iter)
    retriever_fn = partial(retriever_node, retriever=retriever, llm=llm)
    analyst = partial(analyst_node, llm=llm)
    creative = partial(creative_node, llm=llm)
    synthesizer = partial(synthesizer_node, llm=llm)

    # ── 5. Define el StateGraph ───────────────────────────────────────────────
    builder = StateGraph(MilYUnaState)

    # Registra los nodos (cada uno es una función state → dict)
    builder.add_node("supervisor", supervisor)
    builder.add_node("retriever", retriever_fn)
    builder.add_node("analyst", analyst)
    builder.add_node("creative", creative)
    builder.add_node("synthesizer", synthesizer)

    # El punto de entrada siempre es el Supervisor
    builder.set_entry_point("supervisor")

    # Edges condicionales desde el Supervisor:
    # El Supervisor pone en state["next_agent"] el nombre del siguiente nodo.
    # add_conditional_edges lee ese valor y enruta al nodo correcto.
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s["next_agent"],  # Función que extrae la decisión del estado
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "creative": "creative",
            "synthesizer": "synthesizer",
        },
    )

    # Todos los agentes especializados regresan al Supervisor
    # (excepto el Synthesizer que va directo a END)
    builder.add_edge("retriever", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("creative", "supervisor")
    builder.add_edge("synthesizer", END)

    # Compila el grafo — a partir de aquí es invocable
    app = builder.compile()
    log.info("[success]Grafo multi-agente compilado exitosamente[/success]")
    return app


# ── Función de demostración ───────────────────────────────────────────────────

def query_single(app: Any, question: str) -> dict[str, Any]:
    """
    Procesa una única pregunta personalizada y retorna resultado estructurado.

    Args:
        app:      Grafo compilado de create_workflow().
        question: Pregunta del usuario.

    Returns:
        Dict con: answer, query_type, steps, latency_s.
    """
    initial = _make_initial_state(question)
    t0 = time.perf_counter()
    final = app.invoke(initial)
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "answer":      final.get("final_answer", "(sin respuesta)"),
        "query_type":  final.get("query_type", "?"),
        "steps":       final.get("steps_completed", []),
        "latency_s":   elapsed,
    }


def run_demo_queries(app: Any) -> None:
    """
    Ejecuta las 3 queries demo del sistema multi-agente y muestra los resultados
    formateados con Rich en la consola.

    Cada query ejercita un flujo diferente:
      - Query 1 (analytical): Supervisor → Retriever → Analyst → Synthesizer
      - Query 2 (creative):   Supervisor → Retriever → Creative → Synthesizer
      - Query 3 (hybrid):     Supervisor → Retriever → Analyst → Creative → Synthesizer

    Args:
        app: Grafo compilado retornado por create_workflow().
    """
    console.rule("[bold magenta]═══  DEMO: Las Mil y Una Noches — Multi-Agent  ═══[/bold magenta]")
    console.print(
        "  Ejecutando [bold]3 queries de demostración[/bold] que ejercitan "
        "diferentes flujos del grafo multi-agente.\n"
    )

    for i, query in enumerate(DEMO_QUERIES, 1):
        # Separador visual con número de query
        console.print(f"\n[bold cyan]{'═'*65}[/bold cyan]")
        console.print(f"[bold cyan]  Query {i} de {len(DEMO_QUERIES)}[/bold cyan]")
        console.print(f"[bold cyan]{'═'*65}[/bold cyan]")
        console.print(f"\n  [bold white]Pregunta:[/bold white] {query}\n")

        # Crea el estado inicial y ejecuta el grafo
        initial_state = _make_initial_state(query)
        t0 = time.perf_counter()

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            log.error(f"Error ejecutando query {i}: {e}")
            console.print(f"  [bold red]Error:[/bold red] {e}")
            continue

        elapsed = round(time.perf_counter() - t0, 2)

        # Muestra metadata del flujo
        query_type = final_state.get("query_type", "?")
        steps = final_state.get("steps_completed", [])
        console.print(
            f"  [dim]Tipo:[/dim] [magenta]{query_type}[/magenta]  "
            f"[dim]|  Agentes:[/dim] [cyan]{' → '.join(steps)}[/cyan]  "
            f"[dim]|  Tiempo:[/dim] [yellow]{elapsed}s[/yellow]\n"
        )

        # Muestra la respuesta final
        final_answer = final_state.get("final_answer", "(sin respuesta)")
        console.print("[bold green]  Respuesta Final:[/bold green]")
        console.print("  " + "─" * 60)

        # Indenta cada línea de la respuesta para que se vea limpia en consola
        for line in final_answer.strip().splitlines():
            console.print(f"  {line}")

        console.print()

    # Cierre
    console.rule("[bold green]Demo completado exitosamente[/bold green]")
    console.print(
        "\n  El sistema multi-agente procesó exitosamente las 3 queries.\n"
        "  Cada una siguió un flujo diferente según su tipo de intención.\n"
    )
