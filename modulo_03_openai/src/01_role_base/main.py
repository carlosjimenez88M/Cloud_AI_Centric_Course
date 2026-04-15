"""
Lección 1: Role-Based Prompting — Los 4 Fantásticos + Ricardo Arjona
════════════════════════════════════════════════════════════════════════

Ejecución:
    cd modulo_03_openai
    uv run 01_role_base.py
"""

from __future__ import annotations

from typing import Any

from shared.logger import get_logger, console
from shared.config_loader import load_config, load_env

from engine import PersonaEngine, print_result, print_best_response
from personas import PERSONA_LABELS
from tasks import TASKS

log = get_logger("role_base.main")


def main() -> None:
    load_env()
    cfg = load_config()

    console.rule("[bold magenta]Los 4 Fantásticos + Arjona — Role-Based Prompting[/bold magenta]")
    console.print(f"  Modelo   : [cyan]{cfg['model']['name']}[/cyan]")
    console.print()

    engine = PersonaEngine(cfg=cfg)
    all_results: dict[str, dict[str, Any]] = {}

    for task_key, task in TASKS.items():
        console.rule(f"[bold blue]{task['id']}: {task['title']}[/bold blue]")
        task_results: dict[str, Any] = {}

        for persona_key in PERSONA_LABELS:
            log.info(
                f"Ejecutando {task['id']} con [bold]{PERSONA_LABELS[persona_key]}[/bold]..."
            )
            result = engine.run_task(persona_key, task)
            task_results[persona_key] = result
            print_result(result)

        valid = {k: v for k, v in task_results.items() if "error" not in v}
        if valid:
            best_key = max(valid, key=lambda k: valid[k]["quality_score"])
            console.print(
                f"\n  [bold green]Mejor para '{task['id']}':[/bold green] "
                f"[bold]{PERSONA_LABELS[best_key]}[/bold]"
            )
            print_best_response(valid[best_key])

        all_results[task_key] = task_results
        console.print()

    # ── Ranking final ─────────────────────────────────────────────────────────
    console.rule("[bold magenta]Ranking General[/bold magenta]")
    scores: dict[str, list[float]] = {pk: [] for pk in PERSONA_LABELS}

    for task_results_v in all_results.values():
        for pk, res in task_results_v.items():
            if "error" not in res:
                scores[pk].append(res["quality_score"])

    avg = {pk: sum(v) / len(v) for pk, v in scores.items() if v}
    for rank, (pk, s) in enumerate(sorted(avg.items(), key=lambda x: x[1], reverse=True), 1):
        bar = "█" * int(s * 20)
        console.print(
            f"  {rank}. [cyan]{PERSONA_LABELS[pk]:<38}[/cyan]  {bar:<20} {s:.3f}"
        )

    console.print()
    console.print("[bold green]Demostración completada.[/bold green]")


if __name__ == "__main__":
    main()
