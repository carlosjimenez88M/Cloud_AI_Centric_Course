"""
Router LLM — Determina qué cadena de análisis usar.

El router usa el propio LLM (gemini-2.5-flash-lite) para clasificar
la pregunta en una de las 4 rutas: story | character | philosophy | creative.

Esto demuestra el patrón de routing dinámico en orquestaciones LLM.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from shared.logger import get_logger

from prompts import ROUTER_PROMPT

import warnings as _warnings

log = get_logger("orchestration.router")

VALID_ROUTES = frozenset({"story", "character", "philosophy", "creative"})
DEFAULT_ROUTE = "story"


class LLMRouter:
    """
    Usa el LLM para clasificar la intención de la pregunta y determinar
    qué cadena de análisis aplicar.

    Args:
        project_id:   ID del proyecto GCP.
        location:     Región de Vertex AI.
        model_name:   Modelo LLM para la clasificación (puede ser el mismo que
                      el de análisis o uno más pequeño/rápido).
    """

    def __init__(self, project_id: str, location: str, model_name: str) -> None:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            llm = ChatVertexAI(
                model_name=model_name,
                project=project_id,
                location=location,
                temperature=0.0,       # Temperatura 0 para clasificación determinista
                max_output_tokens=10,  # Solo necesitamos una palabra
            )
        self._chain = ROUTER_PROMPT | llm | StrOutputParser()
        log.info(f"LLMRouter listo — modelo: [cyan]{model_name}[/cyan]")

    def route(self, question: str) -> str:
        """
        Clasifica la pregunta y retorna la ruta apropiada.

        Returns:
            Una de: "story" | "character" | "philosophy" | "creative"
        """
        raw = self._chain.invoke({"question": question}).strip().lower()
        # Limpiar salida del LLM (puede incluir puntos, espacios, etc.)
        route = raw.split()[0] if raw.split() else DEFAULT_ROUTE
        route = route.strip(".,;:\"'")

        if route not in VALID_ROUTES:
            log.warning(
                f"Router retornó ruta inválida '{raw}' — usando fallback '{DEFAULT_ROUTE}'"
            )
            return DEFAULT_ROUTE

        log.info(f"  Router: '{question[:60]}…' → [bold cyan]{route}[/bold cyan]")
        return route
