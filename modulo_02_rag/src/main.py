"""
    Introducción a las bases de datos vectoriales desde una aplicación
    Henry: Modulo 2
"""

# Tipos de bases de datos 
    # Bases de datos relacionales : SQL , entre otras, y funcionan para información tabular
    # Bases de datos NoSQL : MongoDB, entre otras, y funcionan para información no tabular
    # Bases de datos vectoriales : Chroma (por ejemplo), almacenan información semantica de manera vectorial

# Bases de conocimiento RAG 
# Rag es un patron que intenta recuperar contexto y usarlo para generar mejores outputs


#################################################
# --------- Creando una base de datos --------- #
#################################################

import os
from typing import List, Dict, Any
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text, create_engine
import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env en la raíz del proyecto
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path)

#############################################
# ---- 01. importando la base de datos ---- #
#############################################
# Creamos la conexión a nuestra base de datos SQLite 'sales.db' usando ruta absoluta
db_path = os.path.join(os.path.dirname(__file__), 'sales.db')
db_engine = create_engine(f"sqlite:///{db_path}")
inspector = sqlalchemy.inspect(db_engine)

print("Tablas disponibles:", inspector.get_table_names())
table_name = "sales"


##################################################
# ---- 02. extrayendo el schema de la tabla ---- #
##################################################
schema = inspector.get_columns(table_name)
column_names = [column["name"] for column in schema]
print(f"Columnas en la tabla '{table_name}':", column_names)


##################################################
# ---- 03. extrayendo las primeras 10 filas ---- #
##################################################
sql = f"SELECT * FROM {table_name} LIMIT 10"

with db_engine.begin() as connection:
    answer = connection.execute(text(sql)).fetchall()

print("\nPrimeras 10 filas (como tuplas):")
print(answer)


#################################################
# ---- 04. creando un dataframe con pandas ---- #
#################################################
df = pd.DataFrame(answer, columns=column_names)
print("\nPrimeras 10 filas (como DataFrame):")
print(df)


###########################################################
# ---- 05. creando tools para el agente inteligente ----  #
###########################################################

@tool
def list_tables_tool(config: RunnableConfig) -> List[str]:
    """
    Lista TODAS las tablas disponibles en la base de datos de ventas.
    Usa esta herramienta primero para conocer qué tablas existen en la base de datos.
    """
    db_engine_local: Engine = config.get("configurable", {}).get("db_engine")
    inspector_local = sqlalchemy.inspect(db_engine_local)
    return inspector_local.get_table_names()

@tool
def get_table_schema_tool(table: str, config: RunnableConfig) -> List[Dict[str, Any]]:
    """
    Retorna el esquema (columnas y tipos de datos) de una tabla específica.
    Usa esta herramienta después de listar las tablas para entender la estructura de los datos antes de hacer consultas SQL.
    """
    db_engine_local: Engine = config.get("configurable", {}).get("db_engine")
    inspector_local = sqlalchemy.inspect(db_engine_local)
    return inspector_local.get_columns(table)

@tool
def execute_sql_tool(query: str, config: RunnableConfig) -> Any:
    """
    Ejecuta una consulta SQL proporcionada y devuelve los resultados.
    Asegúrate de que la consulta SQL sea sintácticamente correcta para SQLite interactuando con las tablas conocidas.
    Usa siempre esta herramienta para responder preguntas numéricas y operacionales sobre los datos.
    """
    db_engine_local: Engine = config.get("configurable", {}).get("db_engine")
    try:
        with db_engine_local.begin() as connection:
            result = connection.execute(text(query)).fetchall()
            # Convertimos a un formato de lista de diccionarios para que sea fácilmente interpretado por el LLM
            keys = connection.execute(text(query)).keys()
            return [dict(zip(keys, row)) for row in result]
    except Exception as e:
        return f"Error ejecutando la consulta: {e}"


##########################################################
# ---- 06. probando los tools creados manualmente ----   #
##########################################################

runnable_config = {'configurable': {'db_engine': db_engine}}
print("\nProbando tool list_tables_tool:")
print(list_tables_tool.invoke({}, runnable_config))

print("\nProbando tool get_table_schema_tool:")
print(get_table_schema_tool.invoke({"table": "sales"}, runnable_config))

print("\nProbando tool execute_sql_tool:")
sample_query = "SELECT count(*) as total_ventas FROM sales"
print(execute_sql_tool.invoke({"query": sample_query}, runnable_config))


#######################################################
# ---- 07. configurando el agente con langgraph ----  #
#######################################################
# Inicializamos el Language Model de OpenAI a usar. El API KEY es provisto automáticamente vía variables de entorno.
llm = ChatOpenAI(model="gpt-5.1", temperature=0)

# Unimos las herramientas en un array para inyectárselas a nuestro agente React
tools = [list_tables_tool, get_table_schema_tool, execute_sql_tool]

# Creamos el Agente Inteligente usando React Agent de la librería LangGraph
agent = create_react_agent(llm, tools)

# Mostramos el grafo directamente en la terminal
print("\n" + "="*80)
print("ESTRUCTURA DEL GRAFO (TEXT2SQL)".center(80))
print("="*80)
try:
    agent.get_graph().print_ascii()
except Exception as e:
    print(f"No se pudo imprimir el grafo en ASCII: {e}")
print("="*80 + "\n")

# Mostramos el grafo al estilo Jupyter Notebook (si aplica) 
# y adicionalmente lo guardamos como PNG por si se está corriendo desde la terminal
try:
    from IPython.display import Image, display
    display(
        Image(
            agent.get_graph().draw_mermaid_png()
        )
    )
except Exception:
    pass

try:
    # Siempre tratamos de guardarlo en caso de estar en una terminal
    graph_path = os.path.join(os.path.dirname(__file__), 'agent_graph.png')
    with open(graph_path, "wb") as f:
        f.write(agent.get_graph().draw_mermaid_png())
    print(f"\n[Info] Grafo guardado exitosamente en: {graph_path}")
except Exception as e:
    print(f"No se pudo guardar el grafo en PNG: {e}")


################################################
# ---- 08. ejecutando el agente text2sql ----  #
################################################
print("\n" + "="*80)
print("INICIANDO AGENTE DE TEXTO A SQL (TEXT2SQL)".center(80))
print("="*80)

# Una pregunta de ejemplo en español que el agente debe resolver transformando a SQL automáticamente
pregunta_usuario = "¿Cuántas ventas hay registradas en total y cuál es el promedio del precio y cantidad de los modelos en las transacciones?"

print(f"\nPregunta del usuario:\n> {pregunta_usuario}\n")

# Invocamos el agente con el mensaje inicial y su configuración respectiva
response = agent.invoke(
    {"messages": [("user", pregunta_usuario)]},
    config=runnable_config
)

# Imprimimos la última respuesta (el resultado final) generada por el agente
print("Respuesta generada por el Agente Text-to-SQL:")
print(response["messages"][-1].content)
print("="*80 + "\n")