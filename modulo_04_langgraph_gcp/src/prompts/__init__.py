"""Prompts module — plantillas de instrucciones para cada agente del sistema."""
from .templates import (
    CLASSIFIER_PROMPT,
    RETRIEVER_SUMMARY_PROMPT,
    ANALYST_PROMPT,
    CREATIVE_PROMPT,
    SYNTHESIZER_PROMPT,
)

__all__ = [
    "CLASSIFIER_PROMPT",
    "RETRIEVER_SUMMARY_PROMPT",
    "ANALYST_PROMPT",
    "CREATIVE_PROMPT",
    "SYNTHESIZER_PROMPT",
]
