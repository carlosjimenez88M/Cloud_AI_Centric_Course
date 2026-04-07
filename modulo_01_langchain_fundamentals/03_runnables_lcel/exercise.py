"""
Módulo 01 - Lección 03: Ejercicio — AI Business Advisor

Workflow multi-paso con LCEL:
  1. Genera una idea de negocio para una industria
  2. Analiza fortalezas y debilidades
  3. Genera un reporte estructurado (Pydantic)
  4. Orquestación end-to-end
"""

#######################
# ---- Libraries ---- #
#######################

import logging
import yaml
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent

###########################
# ---- Logger Design ---- #
###########################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


##########################
# ---- Load Config ---- #
##########################

with open(SCRIPT_DIR / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

logger.info(f"Config cargado: {config}")


######################
# ---- Call API ---- #
######################

if not load_dotenv(find_dotenv()):
    logger.warning("No es posible cargar las variables")
    raise FileNotFoundError("Falta el archivo o las Api Key estan malas")
else:
    logger.info("Se cargo la variable")


########################
# ---- LLM Design ---- #
########################

logger.info("Inicializando LLM")
llm_kwargs = {"model": config['model']}
if config.get('temperature') is not None:
    llm_kwargs["temperature"] = config['temperature']
llm = ChatOpenAI(**llm_kwargs)

logger.info(f"LLM Cargado con {config['model']}")

##################
# ---- Logs ---- #
##################

logger.info("Logs activados")
logs = []
parser = StrOutputParser()

parser_and_log_output_chain = RunnableParallel(
    output=parser,
    log=RunnableLambda(lambda x: logs.append(x))
)

idea_prompt = PromptTemplate(
    template=(
        "Eres un advisor creativo y con profundo conocimiento en AI Powered software Engineering. "
        "Genera una idea innovadora de negocio para el sector {industry}. "
        "Tu respuesta debe ser breve, clara y concisa. "
        "No incluyas ningun tipo de formato, solo texto plano. "
        "No incluyas ningun tipo de explicacion adicional, saludo ni despedida."
    ),
    input_variables=["industry"]
)

idea_chain = idea_prompt | llm | parser_and_log_output_chain

idea_result = idea_chain.invoke({"industry": "Inmoviliario"})
print(idea_result['output'])

print('==' * 32)
print(logs)
print('==' * 32)

###########################
# ---- Idea Analysis ---- #
###########################

analysis_prompt = PromptTemplate(
    template=(
        "Analiza las siguientes ideas: "
        "Idea: {idea}. "
        "Identify 3 key strengths and 3 potential weaknesses of the idea."
    ),
    input_variables=["idea"]
)

analysis_chain = analysis_prompt | llm | parser_and_log_output_chain

analysis_result = analysis_chain.invoke({"idea": idea_result["output"]})
print(analysis_result["output"])
print(logs)

#=============================#
# ---- Report Generation ---- #
#=============================#

report_prompt = PromptTemplate(
    template=(
        "Here is a business analysis: "
        "Strengths & Weaknesses: {output} "
        "Generate a structured business report. "
        "In spanish"
    ),
    input_variables=["output"]
)


class AnalysisReport(BaseModel):
    """Strengths and Weaknesses about a business idea"""
    strengths: list = Field(default=[], description="Idea's strength list")
    weaknesses: list = Field(default=[], description="Idea's weakness list")


report_chain = (
    report_prompt | llm.with_structured_output(
        schema=AnalysisReport, method="function_calling"
    )
)

report_result = report_chain.invoke({"output": analysis_result["output"]})
print(report_result)

#======================#
# ---- End to End ---- #
#======================#

e2e_chain = (
    RunnablePassthrough()
    | idea_chain
    | RunnableParallel(idea=RunnablePassthrough())
    | analysis_chain
    | report_chain
)

final_result = e2e_chain.invoke({"industry": "Inmoviliario"})
print(final_result)
print(final_result.strengths)
print(final_result.weaknesses)
print(e2e_chain.get_graph().print_ascii())
