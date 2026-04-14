"""
STEP 4 — Agentic RAG (LangGraph ReAct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsabilidad única: construir y ejecutar un agente ReAct con LangGraph
que responde preguntas usando recuperación de documentos como herramienta.

Arquitectura del grafo:
    Usuario
      └─► [agent]  ← LLM decide si usar herramienta
               ├─ usa herramienta ─► [tools] ─► [agent]  (loop)
               └─ respuesta final ─► END

Uso independiente:
    from rag.step4_agent import AgenticRAG
    rag = AgenticRAG(vector_store, project_id="...", location="...", model_name="...")
    rag.run()         # ejecuta preguntas demo
    answer = rag.ask("¿Quién es Scheherazade?")
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Sequence, TypedDict

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from shared.logger import get_logger, console

log = get_logger("rag.step4")

# ── Estado del grafo ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Historial de mensajes"]


# ── Preguntas de demostración ──────────────────────────────────────────────────

DEMO_QUESTIONS: list[str] = [
    "¿Quién es Scheherazade y cuál es su rol en Las Mil y Una Noches?",
    "Describe brevemente la historia del marinero Simbad.",
    "¿Qué papel juega el rey Shahryar al inicio del relato?",
    "Menciona tres objetos o elementos mágicos que aparecen en los cuentos.",
]

# ── Agente ─────────────────────────────────────────────────────────────────────

def _build_retrieval_tool(retriever: Any) -> Any:
    """Crea la herramienta de recuperación RAG ligada al retriever."""

    @tool
    def buscar_en_libro(pregunta: str) -> str:
        """
        Busca fragmentos relevantes en Las Mil y Una Noches para responder
        preguntas sobre personajes, cuentos, escenas o temas del libro.
        Úsala siempre que necesites información del texto.
        """
        docs = retriever.invoke(pregunta)
        if not docs:
            return "No se encontraron fragmentos relevantes para esta pregunta."

        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            text = doc.page_content[:500].strip()
            parts.append(f"[Fragmento {i} — Página {page}]\n{text}")

        return "\n\n---\n\n".join(parts)

    return buscar_en_libro


def build_graph(
    vector_store: Chroma,
    project_id: str,
    location: str,
    model_name: str,
    top_k: int = 5,
) -> Any:
    """
    Compila el grafo LangGraph ReAct.

    Returns:
        Grafo compilado (callable, acepta AgentState).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieval_tool = _build_retrieval_tool(retriever)
    tools = [retrieval_tool]

    llm = ChatVertexAI(
        model_name=model_name,
        project=project_id,
        location=location,
        temperature=0.3,
        max_output_tokens=2000,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # ── Nodos ────────────────────────────────────────────────────────────────
    def call_agent(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(list(state["messages"]))
        return {"messages": list(state["messages"]) + [response]}

    def call_tools(state: AgentState) -> AgentState:
        result = tool_node.invoke(state)
        return {"messages": list(state["messages"]) + result["messages"]}

    # ── Condición de salida ───────────────────────────────────────────────────
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    # ── Grafo ────────────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_agent)
    builder.add_node("tools", call_tools)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


# ── AgenticRAG ────────────────────────────────────────────────────────────────

class AgenticRAG:
    """
    Orquesta el agente LangGraph para consultas sobre el corpus de documentos.

    Args:
        vector_store: ChromaDB ya indexado (output del STEP 3).
        project_id:   ID del proyecto GCP.
        location:     Región de Vertex AI.
        model_name:   Modelo LLM a usar para el agente.
        top_k:        Número de fragmentos a recuperar por búsqueda.
    """

    _SYSTEM_PROMPT = (
        "Eres un experto en literatura árabe clásica, especialmente en "
        "Las Mil y Una Noches. Usa la herramienta de búsqueda para respaldar "
        "tus respuestas con fragmentos reales del libro siempre que sea posible."
    )

    def __init__(
        self,
        vector_store: Chroma,
        project_id: str,
        location: str,
        model_name: str,
        top_k: int = 5,
    ) -> None:
        self._graph = build_graph(
            vector_store=vector_store,
            project_id=project_id,
            location=location,
            model_name=model_name,
            top_k=top_k,
        )
        log.info(
            f"AgenticRAG listo — modelo: [cyan]{model_name}[/cyan]  top_k={top_k}"
        )

    def ask(self, question: str) -> str:
        """
        Envía una pregunta al agente y devuelve la respuesta final como string.

        Args:
            question: Pregunta en lenguaje natural.

        Returns:
            Texto de la respuesta del agente.
        """
        initial_state: AgentState = {
            "messages": [
                HumanMessage(content=f"{self._SYSTEM_PROMPT}\n\nPregunta: {question}")
            ]
        }
        t0 = time.perf_counter()
        final_state = self._graph.invoke(initial_state)
        elapsed = round(time.perf_counter() - t0, 2)
        log.info(f"  Latencia: {elapsed}s")

        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                return content if isinstance(content, str) else str(content)

        return "(sin respuesta)"

    def run(self, questions: list[str] | None = None) -> None:
        """
        Ejecuta una lista de preguntas y muestra los resultados.

        Args:
            questions: Lista de preguntas. Si es None, usa DEMO_QUESTIONS.
        """
        qs = questions or DEMO_QUESTIONS
        log.info(
            f"[step]STEP 4 — Agentic RAG:[/step] "
            f"{len(qs)} pregunta(s) de demostración"
        )
        console.rule("[bold blue]Las Mil y Una Noches — Agentic RAG[/bold blue]")

        for i, q in enumerate(qs, 1):
            console.print(f"\n[bold cyan]Pregunta {i}:[/bold cyan] {q}")
            log.info("  Consultando agente…")
            answer = self.ask(q)
            console.print("[bold green]Respuesta:[/bold green]")
            for line in answer.strip().splitlines():
                console.print(f"  {line}")

        log.info("[success]STEP 4 OK — demostración completada[/success]")
