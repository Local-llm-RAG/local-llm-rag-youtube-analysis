from fastapi import FastAPI

from routers.health import router as health_router
from routers.generate import router as generate_router
from routers.chat import router as chat_router
from routers.transcripts import router as transcript_router
from core.llm import load_model

app = FastAPI(title="Local LLM HTTP (llama_cpp)")

# Controllers
app.include_router(health_router)
app.include_router(generate_router)
app.include_router(chat_router)
app.include_router(transcript_router, prefix="/youtube")

@app.on_event("startup")
def startup():
    load_model()
