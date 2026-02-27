from fastapi import FastAPI

from core.embedding.embed import load_embedding_model
from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.transcripts import router as transcript_router
from routers.embedding import router as embedding_router
from core.llm import load_model
import torch

print(torch.cuda.is_available())
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Torch build:", torch.__version__)
app = FastAPI(title="Local LLM HTTP (llama_cpp)")

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(transcript_router, prefix="/youtube")
app.include_router(embedding_router)

@app.on_event("startup")
def startup():
    load_model()
    load_embedding_model()
