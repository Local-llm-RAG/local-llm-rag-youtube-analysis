from __future__ import annotations

import random
import threading
import time
from functools import lru_cache
from queue import Empty, Queue
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api.proxies import GenericProxyConfig

from util.loader import load_config
from util.app_settings import AppSettings

router = APIRouter()
thread_local = threading.local()


class Snippet(BaseModel):
    text: str
    start: float
    duration: float


class TranscriptResponse(BaseModel):
    video_id: str = Field(..., description="YouTube video id")
    language: str | None = Field(None, description="Language code returned (if available)")
    text: str = Field(..., description="All snippets joined into a single text")


@lru_cache(maxsize=1)
def settings() -> AppSettings:
    return load_config(AppSettings)


@lru_cache(maxsize=1)
def get_pool() -> "ProxyLeasePool":
    return ProxyLeasePool(settings())

@router.get("/transcript", response_model=TranscriptResponse)
def get_transcript(
    video_id: str = Query(..., min_length=5, max_length=32),
    languages: list[str] = Query(default=settings().youtube.default_languages),
):
    try:
        fetched = fetch_transcript(video_id, languages)

        full_text = " ".join(s.text.strip() for s in fetched.snippets if s.text and s.text.strip())
        language = getattr(fetched, "language_code", None) or getattr(fetched, "language", None)

        return TranscriptResponse(video_id=video_id, language=language, text=full_text)

    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable.")
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail=f"No transcript found for requested languages: {languages}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}: {e}")


def fetch_transcript(video_id: str, languages: list[str]):
    cfg = settings()
    pool = get_pool()

    leased: Optional[str] = None
    last_exc: Optional[Exception] = None

    try:
        leased = pool.get(cfg.pool.lease_timeout_seconds)

        for switch_idx in range(1, cfg.pool.max_profile_switches_per_request + 1):
            try:
                youtube_client = get_client_for_profile(leased)
                return youtube_client.fetch(video_id, languages=languages)

            except Exception as e:
                last_exc = e

                if not retriable_exceptions(e):
                    raise

                if switch_idx == cfg.pool.max_profile_switches_per_request:
                    raise

                pool.release(leased)
                leased = None

                _sleep_backoff(
                    base=cfg.pool.base_sleep_seconds,
                    attempt=switch_idx,
                    jitter=cfg.pool.jitter_seconds,
                )

                leased = pool.get(cfg.pool.lease_timeout_seconds)

        raise last_exc or RuntimeError("Unknown transcript fetch failure")

    finally:
        if leased is not None:
            pool.release(leased)


class ProxyLeasePool:
    def __init__(self, cfg: AppSettings) -> None:
        self._cfg = cfg
        self._q: Queue[str] = Queue()

        for i in range(cfg.webshare.first_profile, cfg.webshare.max_profile + 1):
            self._q.put(_profile_username(i))

    def get(self, timeout_s: float) -> str:
        try:
            return self._q.get(timeout=timeout_s)
        except Empty:
            raise HTTPException(
                status_code=503,
                detail=f"All proxy profiles are busy. Try again later (waited {timeout_s}s).",
            )

    def release(self, username: str) -> None:
        self._q.put(username)


def get_client_for_profile(username: str) -> YouTubeTranscriptApi:
    cache: Dict[str, YouTubeTranscriptApi] = getattr(thread_local, "client_cache", None)
    if cache is None:
        cache = {}
        thread_local.client_cache = cache

    existing = cache.get(username)
    if existing is not None:
        return existing

    proxy = _webshare_url(username)
    proxy_config = GenericProxyConfig(http_url=proxy, https_url=proxy)

    client = YouTubeTranscriptApi(
        proxy_config=proxy_config,
        http_client=_get_thread_session(),
    )
    cache[username] = client
    return client


def _get_thread_session() -> Session:
    s = getattr(thread_local, "session", None)
    if s is None:
        s = Session()
        s.trust_env = False
        thread_local.session = s
    return s


def retriable_exceptions(err: Exception) -> bool:
    msg = (str(err) or "").lower()
    return any(
        x in msg
        for x in ["too many requests", "429", "rate limit", "quota", "blocked", "captcha", "requestblocked", "ipblocked"]
    )


def _sleep_backoff(base: float, attempt: int, jitter: float) -> None:
    time.sleep((base * (2 ** (attempt - 1))) + random.uniform(0.0, jitter))


def _profile_username(i: int) -> str:
    cfg = settings()
    return f"{cfg.webshare.base_user}-{i}"


def _webshare_url(username: str) -> str:
    cfg = settings()
    return f"http://{username}:{cfg.webshare.password}@{cfg.webshare.host}:{cfg.webshare.port}"