"""
Grafo LangGraph de Orquestación Profunda — Las Mil y Una Noches
════════════════════════════════════════════════════════════════════

Estado del grafo (OrchestrationState):
  question      — pregunta del usuario
  route         — tipo de análisis (story|character|philosophy|creative)
  context       — fragmentos recuperados del corpus (RAG)
  analysis      — resultado del chain específico
  final_response — respuesta integrada por el Synthesizer

Topología del grafo:
  START
    │
    ▼
  [route_node]        ← LLMRouter clasifica la pregunta
    │
    ▼
  [retrieve_node]     ← RAG recupera fragmentos relevantes
    │
    ▼
  [analyze_node]      ← Chain específico según la ruta
    │
    ▼
  [synthesize_node]   ← Sintetiza análisis en respuesta final
    │
    ▼
  END
"""

from __future__ import annotations

import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from shared.logger import get_logger, console

from chains import build_chains, get_chain
from router import LLMRouter

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]

log = get_logger("orchestration.graph")


# ── Estado ────────────────────────────────────────────────────────────────────

class OrchestrationState(TypedDict):
    question: str
    route: str
    context: str
    analysis: str
    final_response: str


# ── Preguntas de demostración ─────────────────────────────────────────────────

DEMO_QUESTIONS: list[str] = [
    # story
    "¿Qué sucede en el primer cuento que narra Scheherazade al rey?",
    # character
    "Analiza en profundidad el personaje de Scheherazade: motivaciones, estrategia y legado.",
    # philosophy
    "¿Qué enseña Las Mil y Una Noches sobre el poder del relato para transformar corazones?",
    # creative
    "Escribe un breve poema al estilo árabe clásico sobre Scheherazade y la noche.",
]


# ── Nodos del grafo ───────────────────────────────────────────────────────────

def _make_route_node(router: LLMRouter):
    def route_node(state: OrchestrationState) -> OrchestrationState:
        log.info(f"  [route] Clasificando pregunta…")
        route = router.route(state["question"])
        return {**state, "route": route}
    return route_node


def _make_retrieve_node(retriever: Any, top_k: int):
    def retrieve_node(state: OrchestrationState) -> OrchestrationState:
        log.info(f"  [retrieve] Buscando fragmentos (top_k={top_k})…")
        docs = retriever.invoke(state["question"])
        parts: list[str] = []
        for i, doc in enumerate(docs[:top_k], 1):
            page = doc.metadata.get("page", "?")
            text = doc.page_content[:400].strip()
            parts.append(f"[Fragmento {i} — Pág. {page}]\n{text}")
        context = "\n\n---\n\n".join(parts) if parts else "Sin fragmentos relevantes."
        log.info(f"  [retrieve] {len(parts)} fragmento(s) recuperado(s)")
        return {**state, "context": context}
    return retrieve_node


def _make_analyze_node(chains: dict[str, Runnable]):
    def analyze_node(state: OrchestrationState) -> OrchestrationState:
        route = state["route"]
        log.info(f"  [analyze] Ejecutando cadena [{route}]…")
        chain = get_chain(chains, route)
        analysis = chain.invoke({
            "question": state["question"],
            "context": state["context"],
        })
        return {**state, "analysis": analysis}
    return analyze_node


def _make_synthesize_node(chains: dict[str, Runnable]):
    def synthesize_node(state: OrchestrationState) -> OrchestrationState:
        log.info("  [synthesize] Integrando análisis en respuesta final…")
        synthesis_chain = chains["synthesis"]
        final = synthesis_chain.invoke({
            "question": state["question"],
            "analysis": state["analysis"],
        })
        return {**state, "final_response": final}
    return synthesize_node


# ── OrchestrationGraph ────────────────────────────────────────────────────────

class OrchestrationGraph:
    """
    Orquestador LangGraph con routing dinámico, chaining y síntesis.

    Args:
        vector_store: ChromaDB ya indexado.
        project_id:   ID del proyecto GCP.
        location:     Región de Vertex AI.
        model_name:   Modelo LLM principal.
        top_k:        Fragmentos a recuperar por búsqueda.
    """

    def __init__(
        self,
        vector_store: Chroma,
        project_id: str,
        location: str,
        model_name: str,
        top_k: int = 5,
    ) -> None:
        log.info("Construyendo OrchestrationGraph…")
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        log.info("  Inicializando cadenas LangChain…")
        chains = build_chains(
            project_id=project_id,
            location=location,
            model_name=model_name,
        )
        router = LLMRouter(
            project_id=project_id,
            location=location,
            model_name=model_name,
        )

        builder = StateGraph(OrchestrationState)
        builder.add_node("route", _make_route_node(router))
        builder.add_node("retrieve", _make_retrieve_node(retriever, top_k))
        builder.add_node("analyze", _make_analyze_node(chains))
        builder.add_node("synthesize", _make_synthesize_node(chains))

        builder.add_edge(START, "route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "analyze")
        builder.add_edge("analyze", "synthesize")
        builder.add_edge("synthesize", END)

        self._graph = builder.compile()
        log.info("[success]OrchestrationGraph compilado[/success]")

    def query(self, question: str) -> dict[str, str]:
        """
        Ejecuta el pipeline completo para una pregunta.

        Returns:
            Diccionario con: route, context_preview, analysis_preview, final_response.
        """
        initial: OrchestrationState = {
            "question": question,
            "route": "",
            "context": "",
            "analysis": "",
            "final_response": "",
        }
        t0 = time.perf_counter()
        final_state = self._graph.invoke(initial)
        elapsed = round(time.perf_counter() - t0, 2)

        return {
            "question": question,
            "route": final_state["route"],
            "context_preview": final_state["context"][:200] + "…",
            "analysis_preview": final_state["analysis"][:200] + "…",
            "final_response": final_state["final_response"],
            "latency_s": elapsed,
        }

    def run(self, questions: list[str] | None = None) -> None:
        """Ejecuta preguntas demo y muestra resultados formateados."""
        qs = questions or DEMO_QUESTIONS
        console.rule(
            "[bold magenta]Orquestación Profunda — Las Mil y Una Noches[/bold magenta]"
        )
        console.print(
            f"  Pipeline: [cyan]Pregunta → Router → RAG → Chain → Síntesis[/cyan]\n"
        )

        for i, q in enumerate(qs, 1):
            console.print(f"[bold cyan]{'─'*60}[/bold cyan]")
            console.print(f"[bold cyan]Pregunta {i}:[/bold cyan] {q}")
            result = self.query(q)

            console.print(
                f"  [bold]Ruta detectada:[/bold] [magenta]{result['route']}[/magenta]  "
                f"[dim]latencia: {result['latency_s']}s[/dim]"
            )
            console.print(f"\n[bold green]Respuesta final:[/bold green]")
            for line in result["final_response"].strip().splitlines():
                console.print(f"  {line}")
            console.print()

        console.rule("[bold green]Orquestación completada[/bold green]")
