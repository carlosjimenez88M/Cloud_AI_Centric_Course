"""
Módulo 01 - Lección 01: Introducción a LangChain

Temas cubiertos:
  - Inicialización del LLM (ChatOpenAI)
  - Estructura de mensajes: SystemMessage, HumanMessage, AIMessage
  - PromptTemplate básico
  - FewShotPromptTemplate con ejemplos de razonamiento
"""

#######################
# ---- Libraries ---- #
#######################

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

###########################
# ---- Logger Design ---- #
###########################

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


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

llm = ChatOpenAI(model="gpt-5-mini", temperature=None)

logger.info("Testing ApiCall")
# print(
#     llm.invoke(
#         "Que es tan efimero en el pensamiento de Vargas LLosa en su libro "
#         "llamada a la tribu, que realmente vale la pena hacer una critica"
#     )
# )

# ==========================#
# ---- Chat Structure ---- #
# ==========================#

messages = [
    SystemMessage("Eres un especialista en culturas latinoamericanas vista desde un optica antropologica"),
    HumanMessage(
        "De qué trata la cancion de La Cuca: Señorita Caradepizza?"
    ),
    AIMessage(
        "Vas sa responder desde la antropologica latinoamericana, de manera muy pero muy breve , pero no superficial"
    ),
]

print(llm.invoke(messages))

# # ==========================#

# # ---- Template cases ---- #
# # ==========================#

print("==" * 32)
logger.info("Inicializando template para el caso de Donde estan las rubias")

topic = "Donde estan las rubias?"
prompt = f"Dime 10 datos curiosos sobre la pelicula {topic}"
print(llm.invoke(prompt))
logger.info("Finalizando el ejemplo de template basico")


# # ============================#
# #  ---- Prompt Template ---- #
# # ============================#

print('=='*64)
prompt_template = PromptTemplate(
    template="dime datos curiosos sobre {topic}")

logger.info("Iniciando ejemplo con Freddy Mercury")
#print(prompt_template.format(topic="Dominc Miller"))
print(llm.invoke(prompt_template.invoke({"topic": "Freddy Mercury"})))
logger.info("Finalizando ejemplo de Freddy Mercury")
print('=='*64)

# ===========================#
# ---- Few Shot Prompt ---- #
# ===========================#

example_prompt = PromptTemplate(
    template="Question: {input}\nThought: {thought}\nResponse: {output}"
)

examples = [
    {
        "input": "A train leaves city A for city B at 60 km/h, and another train leaves city B for city A at 40 km/h. If the distance between them is 200 km, how long until they meet?",
        "thought": "The trains are moving towards each other, so their relative speed is 60 + 40 = 100 km/h. The time to meet is distance divided by relative speed: 200 / 100 = 2 hours.",
        "output": "2 hours",
    },
    {
        "input": "If a store applies a 20% discount to a $50 item, what is the final price?",
        "thought": "A 20% discount means multiplying by 0.8. So, $50 × 0.8 = $40.",
        "output": "$40",
    },
    {
        "input": "A farmer has chickens and cows. If there are 10 heads and 32 legs, how many of each animal are there?",
        "thought": "Let x be chickens and y be cows. We have two equations: x + y = 10 (heads) and 2x + 4y = 32 (legs). Solving: x + y = 10 → x = 10 - y. Substituting: 2(10 - y) + 4y = 32 → 20 - 2y + 4y = 32 → 2y = 12 → y = 6, so x = 4.",
        "output": "4 chickens, 6 cows",
    },
    {
        "input": "If a car travels 90 km in 1.5 hours, what is its average speed?",
        "thought": "Speed is distance divided by time: 90 km / 1.5 hours = 60 km/h.",
        "output": "60 km/h",
    },
    {
        "input": "John is twice as old as Alice. In 5 years, their combined age will be 35. How old is Alice now?",
        "thought": "Let Alice's age be x. Then John's age is 2x. In 5 years, their ages will be x+5 and 2x+5. Their sum is 35: x+5 + 2x+5 = 35 → 3x + 10 = 35 → 3x = 25 → x = 8.33.",
        "output": "8.33 years old",
    },
]
print(example_prompt.invoke(examples[0]).to_string())
logger.info("Iniciando ejemplo con Few Shot Prompt")
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

response = llm.invoke(
    prompt_template.invoke(
        {"input": "If today is Wednesday, what day will it be in 10 days?"}
    )
)
print(response.content)
logger.info("Finalizando ejemplo con Few Shot Prompt")