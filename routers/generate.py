from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from core.llm import run_completion
from core.state import model_path

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repeat_penalty: float = Field(1.1, ge=0.0, le=2.0)
    stop: Optional[list[str]] = None


class GenerateResponse(BaseModel):
    text: str
    usage: Optional[Dict[str, Any]]
    model_path: str


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    out = run_completion(**req.dict())
    return GenerateResponse(
        text=out["choices"][0]["text"].strip(),
        usage=out.get("usage"),
        model_path=model_path or "",
    )
