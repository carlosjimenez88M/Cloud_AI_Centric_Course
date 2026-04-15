"""
Cadenas LangChain para cada tipo de análisis.

Cada cadena sigue el patrón:
  prompt | llm | parser

El módulo expone get_chain(route) para obtener la cadena adecuada.
"""

from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from shared.logger import get_logger

from prompts import CHAIN_PROMPTS, SYNTHESIS_PROMPT

log = get_logger("orchestration.chains")


def build_chains(
    model_name: str,
    temperature: float = 0.5,
    max_tokens: int = 2000,
) -> dict[str, Runnable]:
    """
    Construye todas las cadenas de análisis y la cadena de síntesis.

    Returns:
        Diccionario con claves: "story", "character", "philosophy", "creative",
        "synthesis".
    """
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
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
