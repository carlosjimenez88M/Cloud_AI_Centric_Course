"""
Multi-Agent System: El Guardián de Las Mil y Una Noches
Módulo 04 — LangGraph + Vertex AI (Google Cloud)
Cloud AI-Centric Course — Henry (LatAm)

════════════════════════════════════════════════════════════════════════
Arquitectura del Sistema Multi-Agente:

  Usuario
    │
    ▼
  Supervisor ──► clasifica query (analytical | creative | hybrid)
    │
    ├──► Retriever  ──► busca en ChromaDB de Las Mil y Una Noches
    │       └──► vuelve al Supervisor
    │
    ├──► Analyst    ──► análisis literario profundo
    │       └──► vuelve al Supervisor
    │
    ├──► Creative   ──► poemas y canciones estilo Arjona
    │       └──► vuelve al Supervisor
    │
    └──► Synthesizer ──► respuesta final integrada
            └──► END

════════════════════════════════════════════════════════════════════════

Ejecución:
    cd modulo_04_langgraph_gcp
    uv run run_demo.py

Prerequisito: El ChromaDB de modulo_03_gcp debe existir.
    cd modulo_03_gcp && uv run 02_rag_pipeline.py
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# ── Silenciar warnings antes de importar LangChain / gRPC ────────────────────
# "ignore::all" es la única forma de suprimir LangChainDeprecationWarning, que
# extiende PendingDeprecationWarning y es emitido desde módulos de terceros.
# Para ver los warnings durante desarrollo, comenta esta línea.
warnings.filterwarnings("ignore")
# Suprimir el ruido de gRPC "FD from fork parent still in poll list"
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

# ── sys.path setup ────────────────────────────────────────────────────────────
# Añadimos src/ al path para que los imports funcionen sin instalar el paquete.
# Este patrón es intencional para material educativo: transparente y explícito.
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Imports del módulo ────────────────────────────────────────────────────────
from shared.logger import get_logger, console
from shared.config import load_config, get_project_id, get_location, validate_config
from rag.store import load_vector_store
from graph.workflow import create_workflow, run_demo_queries, query_single

log = get_logger("run_demo")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="El Guardián de Las Mil y Una Noches — Sistema Multi-Agente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  uv run run_demo.py                          # 3 queries de demostración\n"
            "  uv run run_demo.py -q '¿Quién es Aladino?'  # pregunta personalizada\n"
            "  uv run run_demo.py --verify                 # solo verificar entorno\n"
        ),
    )
    parser.add_argument(
        "--query", "-q",
        metavar="PREGUNTA",
        help="Pregunta personalizada a procesar con el sistema multi-agente",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verificar el entorno (equivale a uv run verify_setup.py)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada principal del sistema multi-agente.

    Modos:
      --verify            Verifica el entorno sin correr el demo
      --query "pregunta"  Procesa una pregunta personalizada
      (sin flags)         Ejecuta las 3 queries de demostración
    """
    args = _parse_args()
    t_start = time.perf_counter()

    # ── Modo: verificar entorno ───────────────────────────────────────────────
    if args.verify:
        verify_path = Path(__file__).parent / "verify_setup.py"
        os.execv(sys.executable, [sys.executable, str(verify_path)])
        return  # no llega aquí

    # ── Carga y valida configuración ──────────────────────────────────────────
    cfg = load_config()
    errors = validate_config(cfg)
    if errors:
        console.print("[bold red]Errores en config.yaml:[/bold red]")
        for err in errors:
            console.print(f"  [red]• {err}[/red]")
        sys.exit(1)

    project_id = get_project_id()
    location   = get_location()

    # ── Banner de bienvenida ──────────────────────────────────────────────────
    console.rule("[bold magenta]El Guardián de Las Mil y Una Noches[/bold magenta]")
    console.print()
    console.print("  [bold]Sistema Multi-Agente[/bold] con LangGraph + Vertex AI")
    console.print(f"  Proyecto GCP : [cyan]{project_id}[/cyan]")
    console.print(f"  Modelo       : [cyan]{cfg['model']['name']}[/cyan]")
    console.print(f"  Región       : [cyan]{location}[/cyan]")
    console.print(
        f"  ChromaDB     : [cyan]{cfg['rag']['chroma_path']}[/cyan]  "
        f"(colección: [cyan]{cfg['rag']['collection']}[/cyan])"
    )
    if args.query:
        console.print(f"  Modo         : [yellow]query personalizada[/yellow]")
    console.print()

    # ── Carga el ChromaDB ─────────────────────────────────────────────────────
    log.info("Cargando vector store (ChromaDB)...")
    try:
        vector_store = load_vector_store(cfg, project_id, location)
    except FileNotFoundError as e:
        console.print(f"[bold red]ERROR:[/bold red] {e}")
        console.print(
            "\n[yellow]Solución:[/yellow] Ejecuta primero el pipeline RAG:\n"
            "  cd ../modulo_03_gcp\n"
            "  uv run 02_rag_pipeline.py\n"
        )
        sys.exit(1)

    # ── Construye el grafo multi-agente ───────────────────────────────────────
    log.info("Construyendo grafo multi-agente...")
    app = create_workflow(cfg, project_id, location, vector_store)

    # ── Ejecuta queries ───────────────────────────────────────────────────────
    if args.query:
        # Modo query única personalizada
        console.rule("[bold cyan]Query Personalizada[/bold cyan]")
        console.print(f"\n  [bold]Pregunta:[/bold] {args.query}\n")
        result = query_single(app, args.query)
        console.print(
            f"  [dim]Tipo:[/dim] [magenta]{result['query_type']}[/magenta]  "
            f"[dim]|  Agentes:[/dim] [cyan]{' → '.join(result['steps'])}[/cyan]  "
            f"[dim]|  Tiempo:[/dim] [yellow]{result['latency_s']}s[/yellow]\n"
        )
        console.print("[bold green]Respuesta:[/bold green]")
        console.print("  " + "─" * 60)
        for line in result["answer"].strip().splitlines():
            console.print(f"  {line}")
    else:
        # Modo demo: 3 queries predefinidas
        run_demo_queries(app)

    # ── Resumen final ─────────────────────────────────────────────────────────
    elapsed_total = round(time.perf_counter() - t_start, 2)
    log.info(f"[success]Módulo 04 completado[/success] — Tiempo total: {elapsed_total}s")


if __name__ == "__main__":
    main()
