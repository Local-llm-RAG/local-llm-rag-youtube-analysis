from fastapi import APIRouter
from core.state import model_path
from core.llm import N_CTX, N_THREADS, N_GPU_LAYERS

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": model_path,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
    }
