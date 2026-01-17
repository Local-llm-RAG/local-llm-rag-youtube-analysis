import os
import time
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from fastapi import HTTPException
from llama_cpp import Llama, CreateCompletionResponse
from core.state import llm, model_path

MODEL_DIR = os.getenv("MODEL_DIR", r"C:\models\bggpt-2.6b")
PATTERN = os.getenv("MODEL_PATTERN", r"*Q4_K_M*.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "2"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

DEFAULT_REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))
DEFAULT_STOP = ["</s>", "### Instruction:", "### Response:"]


def pick_model_path(model_dir: str, pattern: str) -> str:
    matches = glob.glob(str(Path(model_dir) / pattern))
    if not matches:
        raise FileNotFoundError(f"No GGUF found in {model_dir} matching {pattern}")
    return matches[0]


def load_model() -> None:
    global llm, model_path

    model_path = pick_model_path(MODEL_DIR, PATTERN)

    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    print(f"[llama_cpp] Loaded model: {model_path}")
    print(f"[llama_cpp] Startup load time: {time.time() - t0:.2f}s")


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
) -> CreateCompletionResponse | Iterator[CreateCompletionResponse]:
    model = ensure_llm()
    return model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=stop or DEFAULT_STOP,
    )
