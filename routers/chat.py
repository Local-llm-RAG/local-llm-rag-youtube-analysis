from typing import List, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field
from core.llm import run_completion
from core.prompt import build_simple_chat_prompt
from core.state import model_path

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repeat_penalty: float = 1.1


class ChatResponse(BaseModel):
    answer: str
    model_path: str
    history_size: int


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    current_history = req.history
    prompt = build_simple_chat_prompt(current_history, req.message)

    out = run_completion(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repeat_penalty=req.repeat_penalty,
        stop=["</s>", "### User:", "### System:"],
    )

    answer = out["choices"][0]["text"].strip()
    current_history.append({"user": req.message, "assistant": answer})

    return ChatResponse(answer=answer, model_path=model_path or "", history_size=len(current_history))
