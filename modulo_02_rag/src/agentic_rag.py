"""
    Agentic RAG: Patrón de Recuperación Amplificada con Razonamiento de Agentes
    Demostración 05 - Analizando 'Hamlet' de William Shakespeare
"""

import os
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env en la raíz del proyecto
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path)

# ====================================================================
# 1. Preparar la Base de Datos Vectorial (Chroma) con Hamlet en PDF
# ====================================================================

print("\n1. Inicializando carga del PDF de Hamlet para RAG...")
pdf_path = os.path.join(os.path.dirname(__file__), "William Shakespeare Hamlet.pdf")

if not os.path.exists(pdf_path):
    print(f"Error: No se encontró el archivo PDF en {pdf_path}")
    exit(1)

# Cargamos el documento PDF
loader = PyPDFLoader(pdf_path)
documentos_extraidos = loader.load()
print(f"   -> Páginas extraídas del PDF: {len(documentos_extraidos)}")

# Dividimos el texto en trozos (chunks) para que el retriever pueda sacar fragmentos semánticos viables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
documentos_particionados = text_splitter.split_documents(documentos_extraidos)
print(documentos_particionados)
print(f"   -> Chunks generados para la BD Vectorial: {len(documentos_particionados)}")

# Inicializamos el generador de Embeddings de OpenAI (modelo veloz ideal para RAG)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# print("2. Procesando los Embeddings y almacenando en Chroma en memoria (Esto puede tardar un poco...)")
# Creamos la base de datos Chroma a partir de los chunks particionados
vector_store = Chroma.from_documents(
    documents=documentos_particionados,
    embedding=embeddings_model,
)

# Configuramos el recuperador buscando los 3 fragmentos más relevantes por cada iteración

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print(retriever)


# ====================================================================
# 2. Crear las Herramientas (Tools) de Búsqueda para el Agentic RAG
# ====================================================================

@tool
def search_shakespeare_tool(query: str) -> str:
    """
    Herramienta vital para buscar información en el libreto original de Hamlet.
    Recibe un término de búsqueda (query) sumamente preciso y recupera escenas, extractos o menciones.
    
    INSTRUCCIÓN: 
    Usa esta herramienta cuando necesites encontrar actos, monólogos, eventos o comportamiento de personajes.
    SI la información retornada no responde los detalles ('what', 'why', 'who'),
    REFLEXIONA sobre qué está faltando, reformula un nuevo término 'query' más específico (por ejemplo cambiando nombres o usando abstractos) y úsala de nuevo de forma secuencial.
    """
    print(f"\n[Acción del Agente] Ejecutando recuperación vectorial con la query: '{query}'")
    docs = retriever.invoke(query)
    
    if not docs:
        return "No se encontraron resultados literarios consistentes para esa búsqueda."
    
    # Retornamos el contenido consolidado referenciando en la respuesta de cuál página sale la cita
    resultados = []
    for doc in docs:
        pagina = doc.metadata.get("page", "?")
        resultados.append(f"- [Pág {pagina}]: {doc.page_content}")
    
    return "\n\n".join(resultados)


# ====================================================================
# 3. Construir el Agente Inteligente Reflexivo
# ====================================================================

# El LLM actuará como cerebro del Agente (razonamiento y orquestación)
llm = ChatOpenAI(model="gpt-5.4-nano", 
                 temperature=0.2)

# Un System Prompt diseñado para forzar un "Agentic RAG" genuino con reflexión literaria y reintentos (Looping)
system_prompt = """
Eres un profesor literario y un analista experto operando sobre la arquitectura "Agentic RAG" (Retrieval-Augmented Generation con razonamiento reflexivo iterativo).

Tu flujo de trabajo OBLIGATORIO para esta tarea es el siguiente:
1. PLANIFICACIÓN: Piensa los conceptos nucleares de la pregunta del usuario. Diseña una query concisa para buscar los argumentos iniciales.
2. EJECUCIÓN (Retrieve): Llama a search_shakespeare_tool con tu query inicial.
3. REFLEXIÓN (Reason): Lee ávidamente los fragmentos del libro obtenidos y asimila la información. Analiza:
   - ¿Qué arcos de personaje faltan explicar?
   - ¿Encontraste un suceso pero te falta el antecedente o el desenlace?
4. REINTENTO / REFORMULACIÓN (Retry): Si el trasfondo es insuficiente (y casi siempre lo es en la primera búsqueda para preguntas complejas), REFORMULA el concepto, abstrae ideas y vuelve a llamar múltiples veces a search_shakespeare_tool buscando los detalles que te hacen falta para una respuesta contundente.
5. RESPUESTA FINAL: Sintetiza toda la evidencia recopilada en un gran desarrollo crítico sobre el tema. Evita ser superficial. Usa citas de apoyo cuando sirvan, justificándote.
"""

from langchain_core.messages import SystemMessage

# Creamos el Agente con LangGraph
agent = create_react_agent(
    model=llm, 
    tools=[search_shakespeare_tool]
)


# ====================================================================
# 4. Ejecución del Script y Demostración
# ====================================================================

if __name__ == "__main__":
    
    # Pregunta sobre Hamlet especialmente compleja que obligará al Agente a buscar múltiples veces (Agentic RAG)
    pregunta_usuario = "Analiza minuciosamente el estado psicológico y la aparente locura de Hamlet: ¿Cuáles pasajes demuestran que es una estrategia fingida (\"locura metódica\") frente a Polonio, y cómo contrasta esto con la locura real y trágica que acaba sufriendo Ofelia por la muerte de su padre? Aporta profundo detalle."
    
    print("\n" + "="*80)
    print("INICIANDO FLUJO: RETRIEVE -> REASON -> RETRY (AGENTIC RAG)".center(80))
    print("="*80)
    print(f"PREGUNTA DEL USUARIO:\n> {pregunta_usuario}\n")

    # Invocamos el Agente pasándole el SystemPrompt y la consulta juntos como historial de mensajes
    response = agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            ("user", pregunta_usuario)
        ]
    })

    print("\n" + "="*80)
    print("RESPUESTA FINAL SINTETIZADA DEL AGENTE RAG:")
    print("="*80)
    print(response["messages"][-1].content)
    
    print("\n" + "="*80)
    print("ESTRUCTURA DEL GRAFO AGENTIC RAG".center(80))
    print("="*80)
    try:
        agent.get_graph().print_ascii()
    except Exception as e:
        print("No se pudo dibujar el grafo:", e)
    print("="*80 + "\n")
