"""
Estado del grafo LangGraph para el sistema multi-agente.

MilYUnaState es un TypedDict que representa el estado compartido entre
todos los nodos (agentes) del grafo. Cada agente lee del estado y
devuelve un diccionario con los campos que quiere actualizar.

El campo `messages` usa el operador `add_messages` de LangGraph, que
acumula mensajes en lugar de sobreescribirlos (comportamiento de append).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MilYUnaState(TypedDict):
    """
    Estado compartido entre todos los agentes del grafo.

    Campos:
        query:            Pregunta original del usuario (inmutable durante la ejecución).
        query_type:       Clasificación de la query: analytical | creative | hybrid.
                          Vacío al inicio, lo llena el Supervisor en el primer ciclo.
        retrieved_context: Fragmentos del libro recuperados por el Retriever Agent.
        analysis:         Análisis literario generado por el Analyst Agent.
        creative_content: Poema / canción / cuento generado por el Creative Agent.
        final_answer:     Respuesta final sintetizada por el Synthesizer Agent.
        messages:         Historial de mensajes (acumulativo, usa add_messages).
        steps_completed:  Lista de agentes que ya ejecutaron (evita repeticiones).
        iteration_count:  Contador de iteraciones del Supervisor (anti-loop).
        next_agent:       Próximo agente a ejecutar (decisión del Supervisor).
    """

    # Pregunta del usuario — no cambia durante la ejecución
    query: str

    # Tipo de query clasificado por el Supervisor
    # Posibles valores: "analytical", "creative", "hybrid", "" (vacío al inicio)
    query_type: str

    # Contexto recuperado por el Retriever desde ChromaDB
    retrieved_context: str

    # Output del Analyst Agent (análisis literario académico)
    analysis: str

    # Output del Creative Agent (poema, canción, cuento)
    creative_content: str

    # Respuesta final del Synthesizer (lo que ve el usuario)
    final_answer: str

    # Historial de mensajes — Annotated con add_messages para comportamiento acumulativo
    # add_messages es el operador de LangGraph que hace append en lugar de reemplazar
    messages: Annotated[list[BaseMessage], add_messages]

    # Lista de agentes que ya ejecutaron en esta sesión
    # Ejemplo: ["retriever", "analyst"] significa que esos dos ya corrieron
    steps_completed: list[str]

    # Contador de iteraciones del Supervisor (para el mecanismo anti-loop)
    # Si supera max_iterations en config.yaml, el Supervisor fuerza la síntesis
    iteration_count: int

    # El próximo agente que el Supervisor decide ejecutar
    # Posibles valores: "retriever", "analyst", "creative", "synthesizer"
    next_agent: str
