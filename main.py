from __future__ import annotations

import os
import time
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama


MODEL_DIR = os.getenv("MODEL_DIR", r"C:\models\bggpt-2.6b")
PATTERN = os.getenv("MODEL_PATTERN", r"*Q4_K_M*.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "2"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

DEFAULT_REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))
DEFAULT_STOP = ["</s>", "### Instruction:", "### Response:"]

SYSTEM = os.getenv(
    "SYSTEM_PROMPT",
    "Ти си полезен асистент. Отговаряй на български. Бъди кратък и точен.",
)


app = FastAPI(title="Local LLM HTTP (llama_cpp)")

llm: Optional[Llama] = None
MODEL_PATH: Optional[str] = None

# in-memory chat history (per-process)
# in production, move this to redis/db keyed by conversation_id/user_id
_history: List[Dict[str, str]] = []


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repeat_penalty: float = Field(DEFAULT_REPEAT_PENALTY, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    text: str
    usage: Optional[Dict[str, Any]] = None
    model_path: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repeat_penalty: float = Field(DEFAULT_REPEAT_PENALTY, ge=0.0, le=2.0)
    clear_history: bool = False


class ChatResponse(BaseModel):
    answer: str
    model_path: str
    history_size: int


def pick_model_path(model_dir: str, pattern: str) -> str:
    matches = glob.glob(str(Path(model_dir) / pattern))
    if not matches:
        raise FileNotFoundError(f"No GGUF found in {model_dir} matching {pattern}")
    return matches[0]


def ensure_llm() -> Llama:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return llm


def run_completion(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    stop: Optional[List[str]],
) -> Dict[str, Any]:
    model = ensure_llm()
    out = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=stop or DEFAULT_STOP,
    )
    return out


def build_simple_chat_prompt(system: str, history: List[Dict[str, str]], user_msg: str) -> str:
    lines = [f"### System:\n{system}\n"]
    for h in history:
        lines.append(f"### User:\n{h['user']}\n")
        lines.append(f"### Assistant:\n{h['assistant']}\n")
    lines.append(f"### User:\n{user_msg}\n")
    lines.append("### Assistant:\n")
    return "".join(lines)



@app.on_event("startup")
def load_model() -> None:
    global llm, MODEL_PATH

    MODEL_PATH = pick_model_path(MODEL_DIR, PATTERN)

    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print(f"[llama_cpp] Loaded model: {MODEL_PATH}")
    print(f"[llama_cpp] Startup load time: {time.time() - t0:.2f}s")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    out = run_completion(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repeat_penalty=req.repeat_penalty,
        stop=req.stop,
    )
    text = out["choices"][0]["text"].strip()
    usage = out.get("usage")
    return GenerateResponse(text=text, usage=usage, model_path=MODEL_PATH or "")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _history

    if req.clear_history:
        _history = []

    prompt = build_simple_chat_prompt(SYSTEM, _history, req.message)

    out = run_completion(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repeat_penalty=req.repeat_penalty,
        stop=["</s>", "### User:", "### System:"],  # stops for this template
    )
    answer = out["choices"][0]["text"].strip()

    _history.append({"user": req.message, "assistant": answer})
    # keep history bounded
    if len(_history) > 20:
        _history = _history[-20:]

    return ChatResponse(answer=answer, model_path=MODEL_PATH or "", history_size=len(_history))


@app.post("/chat/clear")
def chat_clear():
    global _history
    _history = []
    return {"status": "cleared"}
