"""
Cadenas LangChain para cada tipo de análisis.

Cada cadena sigue el patrón:
  prompt | llm | parser

El módulo expone get_chain(route) para obtener la cadena adecuada.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI
from shared.logger import get_logger

from prompts import CHAIN_PROMPTS, SYNTHESIS_PROMPT

import warnings as _warnings

log = get_logger("orchestration.chains")


def build_chains(
    project_id: str,
    location: str,
    model_name: str,
    temperature: float = 0.5,
    max_output_tokens: int = 2000,
) -> dict[str, Runnable]:
    """
    Construye todas las cadenas de análisis y la cadena de síntesis.

    Returns:
        Diccionario con claves: "story", "character", "philosophy", "creative",
        "synthesis".
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        llm = ChatVertexAI(
            model_name=model_name,
            project=project_id,
            location=location,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    parser = StrOutputParser()

    chains: dict[str, Runnable] = {}
    for route, prompt in CHAIN_PROMPTS.items():
        chains[route] = prompt | llm | parser
        log.info(f"  Cadena [{route}] lista")

    chains["synthesis"] = SYNTHESIS_PROMPT | llm | parser
    log.info("  Cadena [synthesis] lista")

    return chains


def get_chain(chains: dict[str, Runnable], route: str) -> Runnable:
    """
    Retorna la cadena correspondiente a la ruta.
    Si la ruta no existe, usa 'story' como fallback.
    """
    if route not in chains:
        log.warning(f"Ruta desconocida '{route}' — usando fallback 'story'")
        return chains["story"]
    return chains[route]
