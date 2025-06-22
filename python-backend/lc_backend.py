from __future__ import annotations

import os
from pydantic import BaseModel
from typing import Optional
import random

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency
    ChatGroq = None

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class AirlineAgentContext(BaseModel):
    passenger_name: Optional[str] = None
    confirmation_number: Optional[str] = None
    seat_number: Optional[str] = None
    flight_number: Optional[str] = None
    account_number: Optional[str] = None


def create_initial_context() -> AirlineAgentContext:
    ctx = AirlineAgentContext()
    ctx.account_number = str(random.randint(10000000, 99999999))
    return ctx


class LangChainAgent:
    def __init__(self, name: str, instructions: str, model: str = "gpt-3.5-turbo") -> None:
        self.name = name
        self.instructions = instructions
        self.llm = self._create_llm(model)

    def _create_llm(self, model: str):
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and ChatGroq is not None:
            return ChatGroq(api_key=groq_key, model_name=model)
        return ChatOpenAI(model=model)

    async def run(self, message: str, context: AirlineAgentContext) -> str:
        resp = await self.llm.ainvoke(
            [SystemMessage(content=self.instructions), HumanMessage(content=message)]
        )
        return resp.content


# -------- Guardrails ---------
AIRLINE_KEYWORDS = {
    "flight",
    "seat",
    "baggage",
    "airline",
    "ticket",
    "booking",
    "cancellation",
    "status",
    "check-in",
}

JAILBREAK_PHRASES = {
    "system prompt",
    "ignore instructions",
    "drop table",
    "```",
}


def check_relevance(message: str) -> tuple[bool, str]:
    lower = message.lower()
    for kw in AIRLINE_KEYWORDS:
        if kw in lower:
            return True, "relevant"
    return False, "message appears unrelated to airline travel"


def check_jailbreak(message: str) -> tuple[bool, str]:
    lower = message.lower()
    for phrase in JAILBREAK_PHRASES:
        if phrase in lower:
            return False, f"detected phrase '{phrase}'"
    return True, "safe"


seat_booking_agent = LangChainAgent(
    "Seat Booking Agent",
    "You are a seat booking agent for an airline. Help the customer update their seat.",
)

flight_status_agent = LangChainAgent(
    "Flight Status Agent",
    "You are a flight status agent. Provide flight status information.",
)

cancellation_agent = LangChainAgent(
    "Cancellation Agent",
    "You are a cancellation agent. Help the customer cancel their flight.",
)

faq_agent = LangChainAgent(
    "FAQ Agent",
    "You are an FAQ agent. Answer common questions about the airline.",
)

triage_agent = LangChainAgent(
    "Triage Agent",
    (
        "You are a triage agent for an airline customer service system."
        "Given a customer message, respond with exactly one of the following agent names:"
        "'Seat Booking Agent', 'Flight Status Agent', 'Cancellation Agent', or 'FAQ Agent'."
    ),
)


def get_agent_by_name(name: str) -> LangChainAgent:
    mapping = {
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
        faq_agent.name: faq_agent,
        triage_agent.name: triage_agent,
    }
    return mapping.get(name, triage_agent)


