import glob
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Iterator

from fastapi import HTTPException
from llama_cpp import Llama, CreateCompletionResponse

from util.app_settings import LLMSettings
from util.loader import load_config

@lru_cache(maxsize=1)
def llm_settings() -> LLMSettings:
    return load_config(LLMSettings, section="llm")


def load_model() -> None:
    global llm, model_path

    model_path = pick_model_path(llm_settings().model_dir, llm_settings().model_pattern)

    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=llm_settings().n_ctx,
        n_threads=llm_settings().n_threads,
        n_gpu_layers=llm_settings().n_gpu_layers,
        verbose=False,
    )

    print(f"[llama_cpp] Loaded model: {model_path}")


def pick_model_path(model_dir: str, pattern: str) -> str:
    matches = glob.glob(str(Path(model_dir) / pattern))
    if not matches:
        raise FileNotFoundError(
            f"No GGUF found in {model_dir} matching {pattern}"
        )
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
    repeat_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
) -> CreateCompletionResponse | Iterator[CreateCompletionResponse]:
    model = ensure_llm()
    cfg = llm_settings().llm

    return model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty or cfg.default_repeat_penalty,
        stop=stop or cfg.default_stop,
    )