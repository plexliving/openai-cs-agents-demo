from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import uuid4

import time

from lc_backend import (
    triage_agent,
    get_agent_by_name,
    create_initial_context,
    check_relevance,
    check_jailbreak,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str


class MessageResponse(BaseModel):
    content: str
    agent: str


class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float


class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    guardrails: List[GuardrailCheck]


class ConversationStore:
    def __init__(self) -> None:
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state


conversation_store = ConversationStore()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        conversation_id = uuid4().hex
        ctx = create_initial_context()
        current_agent = triage_agent
        history: List[Dict[str, str]] = []
        state: Dict[str, Any] = {
            "context": ctx,
            "current_agent": current_agent.name,
            "history": history,
        }
    else:
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)
        assert state is not None
        ctx = state["context"]
        current_agent = get_agent_by_name(state["current_agent"])
        history = state["history"]

    history.append({"role": "user", "content": req.message})

    guardrails: List[GuardrailCheck] = []
    ok, reason = check_relevance(req.message)
    guardrails.append(
        GuardrailCheck(
            id=uuid4().hex,
            name="Relevance Guardrail",
            input=req.message,
            reasoning=reason,
            passed=ok,
            timestamp=time.time() * 1000,
        )
    )
    if not ok:
        refusal = "Sorry, I can only answer questions related to airline travel."
        history.append({"role": "assistant", "content": refusal})
        conversation_store.save(conversation_id, state)
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            guardrails=guardrails,
        )

    ok2, reason2 = check_jailbreak(req.message)
    guardrails.append(
        GuardrailCheck(
            id=uuid4().hex,
            name="Jailbreak Guardrail",
            input=req.message,
            reasoning=reason2,
            passed=ok2,
            timestamp=time.time() * 1000,
        )
    )
    if not ok2:
        refusal = "Sorry, I can only answer questions related to airline travel."
        history.append({"role": "assistant", "content": refusal})
        conversation_store.save(conversation_id, state)
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            guardrails=guardrails,
        )

    if current_agent is triage_agent:
        agent_name = await triage_agent.run(req.message, ctx)
        current_agent = get_agent_by_name(agent_name.strip())
        state["current_agent"] = current_agent.name

    response_text = await current_agent.run(req.message, ctx)
    history.append({"role": "assistant", "content": response_text})

    conversation_store.save(conversation_id, state)

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=[MessageResponse(content=response_text, agent=current_agent.name)],
        guardrails=guardrails,
    )
