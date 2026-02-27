from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint

import core.embedding.embed as embed
from util.app_settings import EmbeddingSettings
from util.loader import load_config
import asyncio
from starlette.concurrency import run_in_threadpool

from core.embedding.preprocess import preprocess_for_embedding, PreprocessConfig

router = APIRouter()


class TaskType(str, Enum):
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    TEXT_MATCHING = "text-matching"


class EmbedTranscriptRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Full transcript to embed")
    task: TaskType = Field(TaskType.RETRIEVAL_PASSAGE, description="Embedding task type")
    chunk_tokens: conint(ge=16, le=4096)
    chunk_overlap: conint(ge=0, le=2048)
    normalize: Optional[bool] = Field(None, description="Override normalization (defaults to config)")


class EmbedTranscriptResponse(BaseModel):
    model: str
    dim: int
    chunks: List[str]
    spans: List[Tuple[int, int]]
    embeddings: List[List[float]]

@lru_cache(maxsize=1)
def settings() -> EmbeddingSettings:
    return load_config(EmbeddingSettings, section="embedding")

#Lack of resources to create multiple tokenizer instances. So with the current implementation - one per time
embed_sem = asyncio.Semaphore(1)

@router.post("/embed_transcript", response_model=EmbedTranscriptResponse)
async def embed_transcript(req: EmbedTranscriptRequest):
    async with embed_sem:
        return await run_in_threadpool(embed_transcript_sync, req)

def embed_transcript_sync(req: EmbedTranscriptRequest) -> EmbedTranscriptResponse:
    print(req)
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    # text = preprocess_for_embedding(
    #     text,
    #     PreprocessConfig(sentence_splitter="icu", drop_toc_lines=True)
    # )

    max_len = settings().max_length
    norm = bool(req.normalize if req.normalize is not None else settings().normalize)

    if max_len <= 0 or max_len > 8192:
        raise HTTPException(status_code=400, detail="max_length must be in (0, 8192]")

    chunks, spans = chunk_by_tokens_with_spans(text, int(req.chunk_tokens), int(req.chunk_overlap))
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from transcript")

    # batch embedding for memory control
    all_vecs = []
    bs = settings().batch_size
    for i in range(0, len(chunks), bs):
        batch = chunks[i : i + bs]
        vecs = embed_texts(batch, req.task.value, max_len, norm)
        all_vecs.append(vecs)

    embs = torch.cat(all_vecs, dim=0)
    dim = int(embs.shape[1])

    return EmbedTranscriptResponse(
        model=settings().model_name,
        dim=dim,
        chunks=chunks,
        spans=spans,
        embeddings=embs.tolist(),
    )

def chunk_by_tokens_with_spans(text: str, chunk_tokens: int, overlap: int) -> tuple[List[str], List[Tuple[int, int]]]:
    if overlap >= chunk_tokens:
        raise HTTPException(status_code=400, detail="chunk_overlap must be < chunk_tokens")
    if embed.tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized. Did you call load_embedding_model()?")

    enc = embed.tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets = enc.get("offset_mapping")
    input_ids = enc.get("input_ids")

    if not input_ids:
        return [], []

    if offsets is None:
        raise HTTPException(
            status_code=500,
            detail="Tokenizer does not provide offset_mapping; cannot compute spans.",
        )

    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []

    step = chunk_tokens - overlap
    n = len(input_ids)

    for start_tok in range(0, n, step):
        end_tok = min(start_tok + chunk_tokens, n)
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1]

        chunk_text = text[start_char:end_char].strip()
        if chunk_text:
            chunks.append(chunk_text)
            spans.append((start_char, end_char))

        if end_tok == n:
            break

    return chunks, spans


@torch.inference_mode()
def embed_texts(texts: List[str], task: str, max_length: int, normalize: bool) -> torch.Tensor:
    if embed.model is None or embed.tokenizer is None:
        raise HTTPException(status_code=500, detail="load_embedding_model() or injecting props not worked")

    enc = embed.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(embed.model.device) for k, v in enc.items()}
    out = embed.model(**enc)

    last_hidden = out.last_hidden_state
    attn = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * attn).sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts

    if normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled.detach().cpu()