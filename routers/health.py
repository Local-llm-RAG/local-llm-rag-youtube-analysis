from functools import lru_cache

from fastapi import APIRouter
from util.loader import load_config
from util.app_settings import LLMSettings

router = APIRouter()

@lru_cache(maxsize=1)
def llm_settings() -> LLMSettings:
    return load_config(LLMSettings, section="llm")

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": llm_settings().model_path,
        "n_ctx": llm_settings().N_CTX,
        "n_threads": llm_settings().N_THREADS,
        "n_gpu_layers": llm_settings().N_GPU_LAYERS,
    }
