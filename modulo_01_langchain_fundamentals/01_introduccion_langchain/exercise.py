"""
Módulo 01 - Lección 01: Ejercicio — Chatbot con Few-Shot Learning

Implementación de un chatbot con:
  - Memoria de conversación (chat history)
  - Few-shot learning para definir personalidad
  - Clase Chatbot reutilizable
"""

from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChatbotApp")

load_dotenv()


class Chatbot:
    def __init__(self, name: str, instructions: str, examples: List[dict]):
        self.name = name
        self.llm = ChatOpenAI(model="gpt-5-mini")

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        self.messages = [SystemMessage(content=instructions)]
        self.messages.extend(few_shot_prompt.invoke({}).to_messages())

    def invoke(self, user_message: str) -> str:
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response.content


# --- Ejecución ---
instructions = (
    "You are BEEP-42, an advanced robotic assistant. Communicate with beeps and whirs. "
    "Use logical, precise, and playful tone. Add robotic sounds like [BEEP] [WHIR]."
)

examples = [
    {"input": "Hello!", "output": "BEEP. GREETINGS, HUMAN. SYSTEM READY."},
    {"input": "What is 2+2?", "output": "CALCULATING... BEEP BOOP! RESULT: 4."},
]

beep42 = Chatbot(name="Beep 42", instructions=instructions, examples=examples)

print(f"User: HAL, is that you?\nAI: {beep42.invoke('HAL, is that you?')}\n")
print(f"User: Wall-e?\nAI: {beep42.invoke('Wall-e?')}")
