from fastapi import APIRouter
from pydantic import BaseModel, Field
from core.llm import run_completion
from core.prompt import build_simple_chat_prompt, SYSTEM
from core.state import history, model_path

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    clear_history: bool = False


class ChatResponse(BaseModel):
    answer: str
    model_path: str
    history_size: int


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.clear_history:
        history.clear()

    prompt = build_simple_chat_prompt(history, req.message)

    out = run_completion(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repeat_penalty=req.repeat_penalty,
        stop=["</s>", "### User:", "### System:"],
    )

    answer = out["choices"][0]["text"].strip()
    history.append({"user": req.message, "assistant": answer})

    if len(history) > 20:
        del history[:-20]

    return ChatResponse(answer=answer, model_path=model_path or "", history_size=len(history))


@router.post("/chat/clear")
def clear_chat():
    history.clear()
    return {"status": "cleared"}
