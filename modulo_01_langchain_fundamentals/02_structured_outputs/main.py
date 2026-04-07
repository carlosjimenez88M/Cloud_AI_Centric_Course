"""
Módulo 01 - Lección 02: Structured Outputs

Temas cubiertos:
  - Output Parsers: StrOutputParser, PydanticOutputParser
  - Structured Output con Pydantic y TypedDict
  - with_structured_output() para function calling
  - Manejo de errores y JSON malformado
  - 3 approaches para fix parsing: structured_output, LCEL fix chain, with_fallbacks
"""

#######################
# ---- Libraries ---- #
#######################

import logging
import yaml
from datetime import datetime
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from pathlib import Path

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

if not load_dotenv():
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

chain = llm | StrOutputParser()
result = chain.invoke("Cual es la capital de Colombia?")
print(result)

##############################
# ---- Parser Timestamp ---- #
##############################

class DateResult(BaseModel):
    date: datetime = Field(description="A random datetime value")

llm_datetime = llm.with_structured_output(DateResult)
result_time = llm_datetime.invoke("Generate a random datetime value")
print(result_time.date)

##############################
# ---- Boolean Parser ---- #
##############################

class BoolResult(BaseModel):
    result: bool = Field(description="True or False answer")

llm_bool = llm.with_structured_output(BoolResult)
result_bool = llm_bool.invoke("Are you an AI?")
print(result_bool.result)

#########################
# ---- Dict Schema ---- #
#########################

class UserInfo(TypedDict):
    name: Annotated[str, "Name of the user"]
    age: Annotated[int, "Age of the user"]
    email: Annotated[str, "Email of the user"]

llm_with_structure = llm.with_structured_output(UserInfo)
print(llm_with_structure.invoke(
    "My name is Daniel, and i have 38 years old and I love Jazz Music "
    "and this is my email danieljimenez@gmail.com"
))

#============================#
# ---- Pydantic Schemas ---- #
#============================#

class PydanticUserInfo(BaseModel):
    name: Annotated[str, Field(description="Name of the user")]
    age: Annotated[int, Field(description="Age of the user")]
    email: Annotated[str, Field(description="Email of the user")]

llm_pydantic = llm.with_structured_output(PydanticUserInfo)
logger.info("Pydantic Schema")
print(llm_pydantic.invoke(
    "My name is Daniel, and i have 38 years old and I love Jazz Music "
    "and this is my email danieljimenez@gmail.com"
))

#################################
# ---- Dealing with Errors ---- #
#################################

class Performer(BaseModel):
    name: Annotated[str, Field(description="Name of the performer")]
    film_names: Annotated[list[str], Field(description="List of film names")]

llm_with_structure = llm.with_structured_output(Performer)
response = llm_with_structure.invoke(
    "What is a the best movies to Tom Hanks?, Top 5 Only please"
)
print(response)

#======================#
# ---- Fix Parser ---- #
#======================#
print('==' * 32)

parser = PydanticOutputParser(pydantic_object=Performer)
print(parser.parse(response.model_dump_json()))

misformated_results = (
    "{'name':'Tom Hanks', 'film_names':['Forrest Gump (1994)', "
    "'Philadelphia (1993)', 'Saving Private Ryan (1998)', "
    "'Cast Away (2000)', 'Apollo 13 (1995)']}"
)

try:
    parser.parse(misformated_results)
except OutputParserException as e:
    print(f"[ERROR ESPERADO] Parser fallo: {e}")

# ----------------------------------------------------------------
# APPROACH 1: with_structured_output con include_raw=True
# ----------------------------------------------------------------

print('\n--- Approach 1: with_structured_output (previene el problema) ---')
llm_structured = llm.with_structured_output(Performer, include_raw=True)
raw_result = llm_structured.invoke(
    "What is a the best movies to Tom Hanks?, Top 5 Only please"
)
print(f"Parsed OK: {raw_result['parsed']}")
print(f"Parsing error: {raw_result['parsing_error']}")

# ----------------------------------------------------------------
# APPROACH 2: LCEL fix chain (reemplaza OutputFixingParser)
# ----------------------------------------------------------------

print('\n--- Approach 2: LCEL fix chain (reemplaza OutputFixingParser) ---')

FIX_PROMPT = PromptTemplate.from_template(
    "Instructions:\n"
    "{instructions}\n\n"
    "Completion:\n"
    "{completion}\n\n"
    "Above, the Completion did not satisfy the constraints given in the Instructions.\n"
    "Error:\n"
    "{error}\n\n"
    "Please try again. Only respond with an answer that satisfies the constraints "
    "laid out in the Instructions:"
)


def build_fix_chain(base_parser, fix_llm):
    """
    Construye un chain LCEL que corrige JSON malformado usando el LLM.
    Replica el comportamiento de OutputFixingParser.from_llm() usando
    solo langchain_core primitives.
    """
    fix_chain = FIX_PROMPT | fix_llm | StrOutputParser() | base_parser

    def attempt_parse_or_fix(malformed_text: str):
        try:
            return base_parser.parse(malformed_text)
        except (OutputParserException, Exception) as exc:
            return fix_chain.invoke({
                "instructions": base_parser.get_format_instructions(),
                "completion": malformed_text,
                "error": str(exc),
            })

    return RunnableLambda(attempt_parse_or_fix)


fix_chain = build_fix_chain(parser, llm)

print(f"JSON malformado:\n  {misformated_results}\n")
corrected = fix_chain.invoke(misformated_results)
print(f"Resultado corregido: {corrected}")

# ----------------------------------------------------------------
# APPROACH 3: with_fallbacks (para integrar en chains LCEL)
# ----------------------------------------------------------------

print('\n--- Approach 3: with_fallbacks (para integrar en chains LCEL) ---')


def make_llm_fixer(base_parser, fix_llm):
    fix_chain_inner = FIX_PROMPT | fix_llm | StrOutputParser() | base_parser

    def fix_from_exception_dict(inputs: dict):
        return fix_chain_inner.invoke({
            "instructions": base_parser.get_format_instructions(),
            "completion": inputs["input"],
            "error": str(inputs["exception"]),
        })

    return RunnableLambda(fix_from_exception_dict)


robust_parser = (
    RunnableLambda(lambda x: parser.parse(x["input"]))
    .with_fallbacks(
        [make_llm_fixer(parser, llm)],
        exceptions_to_handle=(Exception,),
        exception_key="exception",
    )
)

corrected_v2 = robust_parser.invoke({"input": misformated_results})
print(f"Resultado corregido (with_fallbacks): {corrected_v2}")
print('==' * 32)
