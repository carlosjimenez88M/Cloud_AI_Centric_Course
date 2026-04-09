'''
    Introducción a las bases de datos vectoriales
    Henry: Modulo 2
'''

# Tipos de bases de datos 
    # Bases de datos relacionales : SQL , entre otras, y funcionan para información tabular
    # Bases de datos NoSQL : MongoDB, entre otras, y funcionan para información no tabular
    # Bases de datos vectoriales : Chroma (por ejemplo), almacenan información semantica de manera vectorial

# Bases de conocimiento RAG 
# Rag es un patron que intenta recuperar contexto y usarlo para generar mejores outputs


#################################################
# --------- Creando una base de datos --------- #
#################################################

from typing import List
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text, create_engine
import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig











