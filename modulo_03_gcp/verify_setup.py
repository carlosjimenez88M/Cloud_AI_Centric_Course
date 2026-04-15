"""
verify_setup.py — Verificación del entorno antes de correr el módulo 03.

Comprueba cada componente del sistema y muestra ✅ / ❌ con instrucciones
de solución. Ejecutar esto PRIMERO si hay cualquier problema.

Uso:
    cd modulo_03_gcp
    uv run verify_setup.py
"""

import os
import sys
import subprocess
import warnings
import time

warnings.filterwarnings("ignore")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Rich para output coloreado ────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    class _FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("=" * 60)
    console = _FallbackConsole()

PASS = "[bold green]  ✅  PASS[/bold green]"
FAIL = "[bold red]  ❌  FAIL[/bold red]"
WARN = "[bold yellow]  ⚠️   WARN[/bold yellow]"

results: list[tuple[str, str, str]] = []  # (check, status, detail)

_ROOT = Path(__file__).resolve().parent


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    status = PASS if ok else FAIL
    results.append((name, "✅ PASS" if ok else "❌ FAIL", detail))
    console.print(f"{status}  [bold]{name}[/bold]")
    if detail:
        console.print(f"         [dim]{detail}[/dim]")
    if not ok and fix:
        console.print(f"         [yellow]→ Solución:[/yellow] {fix}")
    return ok


def warn(name: str, detail: str = "", fix: str = "") -> None:
    results.append((name, "⚠️  WARN", detail))
    console.print(f"{WARN}  [bold]{name}[/bold]")
    if detail:
        console.print(f"         [dim]{detail}[/dim]")
    if fix:
        console.print(f"         [yellow]→ Sugerencia:[/yellow] {fix}")


# ══════════════════════════════════════════════════════════════════
console.rule("[bold cyan]Verificación del Entorno — Módulo 03[/bold cyan]")
console.print()

# ── 1. Python version ─────────────────────────────────────────────
py_version = sys.version_info
ok = py_version >= (3, 11)
check(
    "Python >= 3.11",
    ok,
    detail=f"Versión actual: {py_version.major}.{py_version.minor}.{py_version.micro}",
    fix="Descarga Python 3.11+ desde https://www.python.org/downloads/",
)

# ── 2. Variables de entorno (.env) ────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = _ROOT / "src" / ".env"
    load_dotenv(env_path)
    env_exists = env_path.exists()
    project_id = os.getenv("PROJECT_ID", "")
    check(
        "Archivo src/.env existe",
        env_exists,
        detail=f"Ruta: {env_path}",
        fix=f"Crea el archivo con: echo 'PROJECT_ID=tu-proyecto' > {env_path}",
    )
    check(
        "PROJECT_ID configurado",
        bool(project_id),
        detail=f"PROJECT_ID = '{project_id}'" if project_id else "PROJECT_ID no está definido",
        fix="Agrega PROJECT_ID=tu-proyecto-gcp al archivo src/.env",
    )
    location = os.getenv("LOCATION", "us-central1")
    check(
        "LOCATION configurado",
        True,
        detail=f"LOCATION = '{location}'",
    )
except ImportError:
    check("python-dotenv instalado", False, fix="Ejecuta: uv sync")

# ── 3. gcloud CLI ────────────────────────────────────────────────
try:
    result = subprocess.run(
        ["gcloud", "--version"],
        capture_output=True, text=True, timeout=10
    )
    gcloud_ok = result.returncode == 0
    version_line = result.stdout.splitlines()[0] if result.stdout else "?"
    check(
        "Google Cloud CLI (gcloud) instalado",
        gcloud_ok,
        detail=version_line,
        fix="Instala desde: https://cloud.google.com/sdk/docs/install",
    )
except FileNotFoundError:
    check(
        "Google Cloud CLI (gcloud) instalado",
        False,
        fix="Instala desde: https://cloud.google.com/sdk/docs/install\n"
            "         Windows: descarga GoogleCloudSDKInstaller.exe y reinicia la terminal.",
    )
except subprocess.TimeoutExpired:
    check("Google Cloud CLI (gcloud) instalado", False, fix="Timeout ejecutando gcloud --version")

# ── 4. gcloud auth ────────────────────────────────────────────────
try:
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True, text=True, timeout=15
    )
    active_account = result.stdout.strip()
    check(
        "Cuenta GCP autenticada",
        bool(active_account),
        detail=f"Cuenta activa: {active_account}" if active_account else "Sin cuenta activa",
        fix="Ejecuta: gcloud auth login",
    )
    # ADC check
    adc_result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True, timeout=15
    )
    adc_ok = adc_result.returncode == 0 and len(adc_result.stdout.strip()) > 10
    check(
        "Application Default Credentials (ADC)",
        adc_ok,
        detail="Credenciales ADC válidas" if adc_ok else "ADC no configurado",
        fix="Ejecuta: gcloud auth application-default login",
    )
except FileNotFoundError:
    check("Cuenta GCP autenticada", False, fix="Instala gcloud primero")
    check("Application Default Credentials (ADC)", False, fix="Instala gcloud primero")

# ── 5. PDF del corpus ────────────────────────────────────────────
pdf_path = _ROOT / "src" / "database" / "Anónimo Las Mil y Una Noches.pdf"
pdf_exists = pdf_path.exists()
check(
    "PDF 'Anónimo Las Mil y Una Noches.pdf' existe",
    pdf_exists,
    detail=f"Ruta: {pdf_path}",
    fix="Coloca el PDF en src/database/. Es el corpus del RAG pipeline.",
)

# ── 6. GCS bucket accesible ──────────────────────────────────────
bucket_name = "mlops-practices-wb-cap2-end_to_end"
try:
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"gs://{bucket_name}/"],
        capture_output=True, text=True, timeout=15
    )
    bucket_ok = result.returncode == 0
    check(
        f"GCS bucket '{bucket_name}' accesible",
        bucket_ok,
        detail="Bucket accesible" if bucket_ok else result.stderr.strip()[:100],
        fix=f"Verifica que el bucket existe y tienes permisos:\n"
            f"         gcloud storage ls gs://{bucket_name}/",
    )
except FileNotFoundError:
    check(f"GCS bucket '{bucket_name}' accesible", False, fix="Instala gcloud primero")
except subprocess.TimeoutExpired:
    check(f"GCS bucket '{bucket_name}' accesible", False, fix="Timeout accediendo al bucket")

# ── 7. Vertex AI API ─────────────────────────────────────────────
try:
    import google.auth
    credentials, project = google.auth.default()
    auth_ok = credentials is not None
    check(
        "Google Auth (ADC) funciona en Python",
        auth_ok,
        detail=f"Proyecto detectado: {project or 'no detectado'}",
        fix="Ejecuta: gcloud auth application-default login",
    )
except Exception as e:
    check(
        "Google Auth (ADC) funciona en Python",
        False,
        detail=str(e)[:100],
        fix="Ejecuta: gcloud auth application-default login",
    )

# Test real de llamada a Vertex AI
console.print()
console.print("  [dim]Probando conexión con Vertex AI (llamada real al LLM)...[/dim]")
try:
    from shared.config_loader import get_project_id, get_location, load_config
    from google import genai
    from google.genai.types import GenerateContentConfig

    pid = get_project_id()
    loc = get_location()
    cfg = load_config()
    model_name = cfg["model"]["name"] if cfg else "gemini-2.5-flash-lite"

    client = genai.Client(vertexai=True, project=pid, location=loc)
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model_name,
        contents="Responde solo con: OK",
        config=GenerateContentConfig(max_output_tokens=10, temperature=0),
    )
    latency = round(time.perf_counter() - t0, 2)
    api_ok = bool(response.text and len(response.text.strip()) > 0)
    check(
        f"Vertex AI API funciona ({model_name})",
        api_ok,
        detail=f"Respuesta: '{response.text.strip()}'  |  Latencia: {latency}s",
        fix="Verifica que Vertex AI API está habilitada en tu proyecto GCP",
    )
except Exception as e:
    check(
        "Vertex AI API funciona",
        False,
        detail=str(e)[:150],
        fix="Habilita Vertex AI API en: https://console.cloud.google.com → Vertex AI → Enable",
    )

# ── Resumen final ─────────────────────────────────────────────────
console.print()
console.rule("[bold cyan]Resumen[/bold cyan]")

passes = sum(1 for _, s, _ in results if "PASS" in s)
fails  = sum(1 for _, s, _ in results if "FAIL" in s)
warns  = sum(1 for _, s, _ in results if "WARN" in s)

console.print(f"\n  Total: [green]{passes} ✅  PASS[/green]  |  [red]{fails} ❌  FAIL[/red]  |  [yellow]{warns} ⚠️  WARN[/yellow]\n")

if fails == 0:
    console.print(Panel(
        "[bold green]Todo listo.[/bold green] Puedes ejecutar las lecciones:\n\n"
        "  [cyan]uv run 01_role_base.py[/cyan]         # Lección 1 — Role Prompting\n"
        "  [cyan]uv run 02_rag_pipeline.py[/cyan]      # Lección 2 — RAG Pipeline\n"
        "  [cyan]uv run 03_orchestration.py[/cyan]     # Lección 3 — Orquestación",
        title="✅ Entorno verificado",
        border_style="green",
    ))
else:
    console.print(Panel(
        f"[bold red]Hay {fails} problema(s) que resolver antes de continuar.[/bold red]\n\n"
        "Sigue las instrucciones de solución mostradas arriba.\n"
        "Si el problema persiste, revisa la sección Solución de problemas del README.md",
        title="❌ Se requieren correcciones",
        border_style="red",
    ))

sys.exit(0 if fails == 0 else 1)
