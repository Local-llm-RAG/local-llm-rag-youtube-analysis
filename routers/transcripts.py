from __future__ import annotations

import random
import time
import threading
from queue import Queue, Empty
from typing import Dict, Optional

import requests
from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from requests import Session

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api.proxies import GenericProxyConfig

router = APIRouter()

# -------------------------
# Webshare settings (hardcoded)
# -------------------------
BASE_USER = "USERNAME"
PASSWORD = "PASSWORD"
HOST = "p.webshare.io"
PORT = 80

FIRST_PROFILE = 1
MAX_PROFILE = 25

# For each request, how many times we allow switching profiles when blocked
MAX_PROFILE_SWITCHES_PER_REQUEST = 6

# If all profiles are currently leased, how long to wait to get one (seconds)
LEASE_WAIT_SECONDS = 5.0

# -------------------------
# Models
# -------------------------
class Snippet(BaseModel):
    text: str
    start: float
    duration: float


class TranscriptResponse(BaseModel):
    video_id: str = Field(..., description="YouTube video id")
    language: str | None = Field(None, description="Language code returned (if available)")
    snippets: list[Snippet]
    text: str = Field(..., description="All snippets joined into a single text")


# -------------------------
# Helpers
# -------------------------
def _looks_like_block(err: Exception) -> bool:
    msg = (str(err) or "").lower()
    return any(
        x in msg
        for x in [
            "too many requests",
            "429",
            "rate limit",
            "quota",
            "blocked",
            "captcha",
            "requestblocked",
            "ipblocked",
        ]
    )


def _username(i: int) -> str:
    return f"{BASE_USER}-{i}"


def _proxy_url(username: str) -> str:
    return f"http://{username}:{PASSWORD}@{HOST}:{PORT}"


# -------------------------
# Thread-local Session (safe for parallel calls)
# -------------------------
_thread_local = threading.local()


def _get_thread_session() -> Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = Session()
        s.trust_env = False  # CRITICAL: ignore Windows/corp proxy env vars
        _thread_local.session = s
    return s


# -------------------------
# Profile leasing pool
# -------------------------
class ProxyLeasePool:
    """
    A pool of usernames. Each request "leases" one profile so parallel calls spread out.
    """
    def __init__(self) -> None:
        self._q: Queue[str] = Queue()
        for i in range(FIRST_PROFILE, MAX_PROFILE + 1):
            self._q.put(_username(i))

    def lease(self, timeout_s: float) -> str:
        try:
            return self._q.get(timeout=timeout_s)
        except Empty:
            raise HTTPException(
                status_code=503,
                detail=f"All proxy profiles are busy. Try again later (waited {timeout_s}s).",
            )

    def release(self, username: str) -> None:
        # Return profile back to the pool
        self._q.put(username)


pool = ProxyLeasePool()

# Cache one YouTubeTranscriptApi per (username, thread) via thread-local dict
# (so we don't share the same client across threads)
def _get_ytt(username: str) -> YouTubeTranscriptApi:
    cache: Dict[str, YouTubeTranscriptApi] = getattr(_thread_local, "ytt_cache", None)
    if cache is None:
        cache = {}
        _thread_local.ytt_cache = cache

    existing = cache.get(username)
    if existing is not None:
        return existing

    proxy = _proxy_url(username)
    proxy_config = GenericProxyConfig(http_url=proxy, https_url=proxy)

    client = YouTubeTranscriptApi(
        proxy_config=proxy_config,
        http_client=_get_thread_session(),
    )
    cache[username] = client
    return client


def _fetch_with_leased_profile(video_id: str, languages: list[str]):
    """
    Lease a profile (so parallel calls don't pile onto the same username).
    If blocked: release it, lease a new one, and retry.
    """
    leased: Optional[str] = None
    last_exc: Exception | None = None
    base_sleep = 0.6

    try:
        leased = pool.lease(LEASE_WAIT_SECONDS)

        for switch_idx in range(1, MAX_PROFILE_SWITCHES_PER_REQUEST + 1):
            try:
                ytt = _get_ytt(leased)
                return ytt.fetch(video_id, languages=languages)

            except Exception as e:
                last_exc = e

                # Not a block -> don't rotate; fail immediately
                if not _looks_like_block(e):
                    raise

                # If we've exhausted switches -> raise
                if switch_idx == MAX_PROFILE_SWITCHES_PER_REQUEST:
                    raise

                # Blocked -> rotate profile:
                # 1) release current lease
                pool.release(leased)
                leased = None

                # 2) small backoff
                time.sleep((base_sleep * (2 ** (switch_idx - 1))) + random.uniform(0.0, 0.35))

                # 3) lease a new one
                leased = pool.lease(LEASE_WAIT_SECONDS)

        raise last_exc or RuntimeError("Unknown transcript fetch failure")

    finally:
        # Always release whatever we currently hold
        if leased is not None:
            pool.release(leased)


# -------------------------
# Debug endpoint
# -------------------------
@router.get("/debug/proxy")
def debug_proxy():
    leased = pool.lease(LEASE_WAIT_SECONDS)
    proxy = _proxy_url(leased)

    try:
        proxies = {"http": proxy, "https": proxy}
        r = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=20)
        return {
            "status": "ok",
            "leased_username": leased,
            "proxy": proxy,
            "exit_ip": r.json(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "fail",
                "leased_username": leased,
                "proxy": proxy,
                "error": f"{type(e).__name__}: {e}",
            },
        )
    finally:
        pool.release(leased)


# -------------------------
# Transcript endpoint
# -------------------------
@router.get("/transcript", response_model=TranscriptResponse)
def get_transcript(
    video_id: str = Query(..., min_length=5, max_length=32),
    languages: list[str] = Query(default=["bgn", "bg"]),
):
    try:
        fetched = _fetch_with_leased_profile(video_id, languages)

        snippets = [
            Snippet(text=s.text, start=float(s.start), duration=float(s.duration))
            for s in fetched.snippets
        ]

        full_text = " ".join(s.text.strip() for s in fetched.snippets if s.text and s.text.strip())
        language = getattr(fetched, "language_code", None) or getattr(fetched, "language", None)

        return TranscriptResponse(
            video_id=video_id,
            language=language,
            snippets=snippets,
            text=full_text,
        )

    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable.")
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail=f"No transcript found for requested languages: {languages}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}: {e}")
