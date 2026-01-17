from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

router = APIRouter()


class Snippet(BaseModel):
    text: str
    start: float
    duration: float


class TranscriptResponse(BaseModel):
    video_id: str = Field(..., description="YouTube video id")
    language: str | None = Field(None, description="Language code returned (if available)")
    snippets: list[Snippet]
    text: str = Field(..., description="All snippets joined into a single text")


def _looks_like_rate_limit(err: Exception) -> bool:
    msg = (str(err) or "").lower()
    # be tolerant across versions / messages
    return any(
        token in msg
        for token in [
            "too many requests",
            "429",
            "rate limit",
            "quota",
            "temporarily blocked",
        ]
    )


@router.get("/transcript", response_model=TranscriptResponse)
def get_transcript(
    video_id: str = Query(..., min_length=5, max_length=32),
    languages: list[str] = Query(default=["bgn", "bg", "en"]),
):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=languages)

        snippets = [
            Snippet(text=s.text, start=float(s.start), duration=float(s.duration))
            for s in fetched.snippets
        ]

        full_text = " ".join(
            s.text.strip()
            for s in fetched.snippets
            if s.text and s.text.strip()
        )

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
        raise HTTPException(
            status_code=404,
            detail=f"No transcript found for requested languages: {languages}",
        )
    except Exception as e:
        if _looks_like_rate_limit(e):
            raise HTTPException(status_code=429, detail="Too many requests to YouTube. Try again later.")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}: {e}")
