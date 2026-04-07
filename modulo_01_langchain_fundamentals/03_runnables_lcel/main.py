"""
Módulo 01 - Lección 03: Runnables y LCEL (LangChain Expression Language)

Temas cubiertos:
  - Interfaz Runnable: invoke, batch, stream, inspect
  - Composición con LCEL (operador pipe |)
  - RunnableSequence, RunnableParallel, RunnableLambda
  - Configuración de ejecución (run_name, tags, metadata)
  - Visualización del grafo de ejecución
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
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.tracers.context import collect_runs
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

###########################
# ---- Output Parser ---- #
###########################

prompt = PromptTemplate(
    template="Dime un chiste sobre {tema}",
    input_variables=["tema"]
)

parser = StrOutputParser()

chain = prompt | llm | parser
print(chain.invoke({"tema": "Batman"}))


#######################
# ---- Runnables ---- #
#######################

runnables = [prompt, llm, parser]
print('==' * 32)
for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINVOKE: {repr(runnable.invoke)}")
    print(f"\tBATCH: {repr(runnable.batch)}")
    print(f"\tSTREAM: {repr(runnable.stream)}\n")
print('==' * 32)

#####################
# ---- INSPECT ---- #
#####################

print('INSPECT')
for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINPUT: {repr(runnable.get_input_schema())}")
    print(f"\tOUTPUT: {repr(runnable.get_output_schema())}")
    print(f"\tCONFIG: {repr(runnable.config_schema())}\n")

####################
# ---- CONFIG ---- #
####################

print('CONFIG')
with collect_runs() as run_collection:
    result = llm.invoke(
        "Hello",
        config={
            'run_name': 'demo_run',
            'tags': ['demo', 'lcel'],
            'metadata': {'lesson': 2}
        }
    )
print(run_collection.traced_runs)

###############################
# ---- Compose Runnables ---- #
###############################

chain = RunnableSequence(prompt, llm, parser)
print(chain.invoke({"tema": "DC Comics"}))

for chunk in chain.stream({"tema": "DC Comics"}):
    print(chunk, end="", flush=True)

print('==' * 32)
print('BATCH')
print('==' * 32)
chain.batch([
    {"tema": "DC Comics"},
    {"tema": "Linterna Verde"},
    {"tema": "La serie FROM"},
])

print(chain.get_graph().print_ascii())

#========================================#
# ---- Turn any function into a Runnable ---- #
#========================================#

def double(x: int) -> int:
    return 2 * x

runnable = RunnableLambda(double)
print(runnable.invoke(2))

################################
# ---- Parallel Runnables ---- #
################################

parallel_chain = RunnableParallel(
    double=RunnableLambda(lambda x: x * 2),
    triple=RunnableLambda(lambda x: x * 3),
)
print(parallel_chain.invoke(3))
print(parallel_chain.get_graph().print_ascii())

##################
# ---- LCEL ---- #
##################

print('LCEL')
chain = prompt | llm | parser
print(chain.invoke({"tema": "Acuaman"}))
