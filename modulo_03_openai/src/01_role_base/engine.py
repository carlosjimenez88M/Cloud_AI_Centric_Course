"""
PersonaEngine — Motor de generación con personas usando OpenAI.

Responsabilidades:
  - Construir el prompt completo (persona + tarea)
  - Llamar a OpenAI Chat Completions con la configuración del config.yaml
  - Calcular métricas de calidad de la respuesta
  - Reportar resultados con logging coloreado
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI
from shared.logger import get_logger, console

from personas import PERSONAS, PERSONA_LABELS

log = get_logger("role_base.engine")


class PersonaEngine:
    """
    Orquesta llamadas a OpenAI Chat Completions con distintas personas.

    Args:
        cfg: Diccionario cargado desde config.yaml.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        model_cfg = cfg["model"]
        self.model = model_cfg["name"]
        self.temperature = float(model_cfg.get("temperature", 0.7))
        self.max_tokens = int(model_cfg.get("max_output_tokens", 2000))

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        log.info(f"PersonaEngine — modelo: [bold magenta]{self.model}[/bold magenta]")

    def run_task(self, persona_key: str, task: dict[str, str]) -> dict[str, Any]:
        """
        Ejecuta una tarea con la persona indicada.

        Returns:
            Diccionario con: persona, label, task_id, response, latency_s,
            word_count, quality_score, metrics, (o 'error').
        """
        if persona_key not in PERSONAS:
            return {"error": f"Persona desconocida: {persona_key}"}

        label = PERSONA_LABELS[persona_key]
        system_prompt = PERSONAS[persona_key]
        user_prompt = f"--- TAREA: {task['title']} ---\n{task['prompt']}"

        try:
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency = round(time.perf_counter() - t0, 2)
            text = response.choices[0].message.content or ""
            metrics = _compute_metrics(text, persona_key, task["id"])

            return {
                "persona": persona_key,
                "label": label,
                "task_id": task["id"],
                "task_title": task["title"],
                "response": text,
                "latency_s": latency,
                "word_count": metrics["word_count"],
                "quality_score": metrics["quality_score"],
                "metrics": metrics,
            }

        except Exception as exc:
            log.error(f"[{label} / {task['title']}]: {exc}")
            return {"error": str(exc), "persona": persona_key, "task_id": task["id"]}


# ── Métricas de calidad ───────────────────────────────────────────────────────

_PERSONA_KEYWORDS: dict[str, list[str]] = {
    "mr_fantastic": ["electromagnético", "dimensional", "fascinante", "cuántico",
                     "energía", "análisis", "probabilidad", "sistema"],
    "the_thing":    ["roca", "batalla", "golpe", "fuerza", "dureza",
                     "hora de machacar", "táctico", "experiencia"],
    "human_torch":  ["fuego", "velocidad", "llamas", "calor", "innovar",
                     "cool", "acción", "primero"],
    "invisible_woman": ["invisible", "barrera", "oculto", "proteger",
                        "estrategia", "percibir", "equipo", "diplomacia"],
    "arjona":       ["corazón", "amor", "alma", "metáfora", "vida",
                     "historia", "sueño", "tiempo", "viento", "noche"],
    "mix":          ["precisión", "táctica", "innovación", "diplomacia",
                     "poética", "síntesis", "equilibrio", "combinación"],
}

_TASK_KEYWORDS: dict[str, list[str]] = {
    "TODO 1": ["fortaleza", "debilidad", "rol", "equipo", "compensar"],
    "TODO 2": ["villano", "rival", "batalla", "victoria", "derrota", "probabilidad"],
    "TODO 3": ["estrofa", "coro", "canción", "verso", "ritmo", "alma"],
    "TODO 4": ["tropical", "escenario", "selva", "calor", "playa", "vegetación"],
}


def _compute_metrics(text: str, persona_key: str, task_id: str) -> dict[str, Any]:
    lower = text.lower()
    words = len(text.split())

    pk = _PERSONA_KEYWORDS.get(persona_key, [])
    tk = _TASK_KEYWORDS.get(task_id, [])

    pk_cov = sum(1 for kw in pk if kw in lower) / len(pk) if pk else 0.0
    tk_cov = sum(1 for kw in tk if kw in lower) / len(tk) if tk else 0.0
    length_bonus = min(words / 200, 1.0)

    score = round(pk_cov * 0.35 + tk_cov * 0.35 + length_bonus * 0.30, 3)

    return {
        "word_count": words,
        "persona_keyword_coverage": round(pk_cov, 2),
        "task_keyword_coverage": round(tk_cov, 2),
        "length_bonus": round(length_bonus, 2),
        "quality_score": score,
        "alignment": "Alto" if score > 0.65 else "Medio" if score > 0.40 else "Bajo",
    }


# ── Helpers de presentación ───────────────────────────────────────────────────

def print_result(result: dict[str, Any]) -> None:
    if "error" in result:
        console.print(f"  [red]ERROR:[/red] {result['error']}")
        return
    m = result["metrics"]
    console.print(
        f"  [cyan]{result['label']}[/cyan]  "
        f"score=[bold]{m['quality_score']:.3f}[/bold]  "
        f"align=[bold]{m['alignment']}[/bold]  "
        f"palabras={m['word_count']}  latencia={result['latency_s']}s"
    )


def print_best_response(result: dict[str, Any], max_chars: int = 600) -> None:
    if "error" in result:
        return
    console.print(f"\n[bold green]--- Respuesta de {result['label']} ---[/bold green]")
    preview = result["response"][:max_chars].strip()
    for line in preview.splitlines():
        console.print(f"  {line}")
    if len(result["response"]) > max_chars:
        console.print("  [dim]...(respuesta continúa)[/dim]")
