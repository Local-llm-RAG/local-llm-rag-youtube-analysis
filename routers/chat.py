from typing import List, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field
from core.llm import run_completion, pick_model_path
from core.prompt import build_simple_chat_prompt
from core.llm import llm_settings

router = APIRouter()


# TODO: default values to config
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
        stop=llm_settings().default_stop,
    )

    answer = out["choices"][0]["text"].strip()
    current_history.append({"user": req.message, "assistant": answer})

    return ChatResponse(answer=answer, model_path=pick_model_path(llm_settings().model_dir, llm_settings().model_pattern) or "", history_size=len(current_history))
