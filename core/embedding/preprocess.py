from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional


# ----------------------------
# Optional dependencies
# ----------------------------
def _try_import_ftfy():
    try:
        import ftfy  # type: ignore
        return ftfy
    except Exception:
        return None


def _try_import_pyicu():
    try:
        import icu  # type: ignore
        return icu
    except Exception:
        return None


def _try_import_blingfire():
    try:
        from blingfire import text_to_sentences  # type: ignore
        return text_to_sentences
    except Exception:
        return None


# ----------------------------
# Heuristics / regex
# ----------------------------
_WEIRD_SPACES = re.compile(r"[\u00A0\u2007\u202F\u2009\u200A\u200B\u2060]+")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_WS_LINES = re.compile(r"[ \t]+\n")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_LINEBREAK = re.compile(r"(?<!\n)\n(?!\n)")  # single newline inside paragraph
_FORMFEED = re.compile(r"\f")

# TOC-ish patterns: "1. Introduction 2. Notation 2.1. ..." etc
_TOC_NUMBERED = re.compile(r"(?:^|\s)(\d+(\.\d+)*)(?:\s*[\)\.]|\s+)")
# Reference markers and bracket noise often appear mid-text; we don't remove them aggressively,
# but we can normalize surrounding spacing a bit.
_SPACES_AROUND_PUNCT = re.compile(r"\s+([,.;:!?])")
_PUNCT_AROUND_SPACES = re.compile(r"([(\[{])\s+|\s+([)\]}])")


@dataclass(frozen=True)
class PreprocessConfig:
    # Multilingual sentence splitting:
    # "icu" = best coverage when PyICU installed
    # "blingfire" = fast fallback if installed
    # "none" = don't sentence split, only paragraph cleanup
    sentence_splitter: str = "icu"

    # Light removal of TOC-ish lines:
    drop_toc_lines: bool = True

    # Remove exact adjacent duplicate paragraphs:
    dedupe_adjacent_paragraphs: bool = True

    # Hard limit to avoid pathological huge strings
    max_chars: int = 2_000_000


def preprocess_for_embedding(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Clean up extracted academic section text to produce better embedding chunks.
    This is NOT semantic rewriting; it's just normalization, de-noising, and robust sentence/paragraph boundaries.
    """
    cfg = cfg or PreprocessConfig()
    if not text:
        return ""

    # Safety: cap gigantic sections (rare but can happen with bad extraction)
    if len(text) > cfg.max_chars:
        text = text[: cfg.max_chars]

    text = _basic_unicode_cleanup(text)
    text = _normalize_whitespace(text)
    text = _fix_pdf_linewrap(text)
    text = _normalize_punctuation_spacing(text)

    # Split into paragraphs early so TOC lines can be dropped cleanly
    paragraphs = _split_paragraphs(text)

    if cfg.drop_toc_lines:
        paragraphs = _drop_toc_like_paragraphs(paragraphs)

    if cfg.dedupe_adjacent_paragraphs:
        paragraphs = _dedupe_adjacent(paragraphs)

    # Sentence split inside each paragraph (helps chunking a lot)
    paragraphs = [_sentenceize(p, splitter=cfg.sentence_splitter) for p in paragraphs]

    # Join back. We keep paragraph boundaries with blank line; inside paragraph, sentences newline-separated.
    out = "\n\n".join(p for p in paragraphs if p.strip())
    out = out.strip()

    # Final tidy
    out = _MANY_NEWLINES.sub("\n\n", out)
    return out


# ----------------------------
# Core cleanup steps
# ----------------------------
def _basic_unicode_cleanup(text: str) -> str:
    # Optional: fix mojibake / “smart quotes gone wrong” / weirdness
    ftfy = _try_import_ftfy()
    if ftfy is not None:
        try:
            text = ftfy.fix_text(text)
        except Exception:
            pass

    # Normalize unicode for stable downstream behavior
    text = unicodedata.normalize("NFKC", text)
    # Remove formfeed chars often used for page breaks
    text = _FORMFEED.sub("\n", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WEIRD_SPACES.sub(" ", text)
    text = _TRAILING_WS_LINES.sub("\n", text)
    # Keep newlines for now; normalize repeated spaces
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _fix_pdf_linewrap(text: str) -> str:
    """
    Fix typical PDF extraction artifacts:
    - hyphenated line breaks: "exam-\nple" -> "example"
    - single newlines inside paragraphs -> spaces
    - keep paragraph breaks (double newline)
    """
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Convert “soft wraps” (single newlines) into spaces, preserving real paragraph breaks
    # But first ensure paragraph breaks are consistent
    text = _MANY_NEWLINES.sub("\n\n", text)
    text = _SOFT_LINEBREAK.sub(" ", text)

    # Re-normalize paragraph breaks
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _normalize_punctuation_spacing(text: str) -> str:
    # Common extraction spacing weirdness
    text = _SPACES_AROUND_PUNCT.sub(r"\1", text)
    text = _PUNCT_AROUND_SPACES.sub(lambda m: (m.group(1) or "") + (m.group(2) or ""), text)
    return text


def _split_paragraphs(text: str) -> List[str]:
    # Split by blank lines
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _dedupe_adjacent(paragraphs: List[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for p in paragraphs:
        if prev is None or p != prev:
            out.append(p)
        prev = p
    return out


# ----------------------------
# TOC-like filtering (very light, for embedding signal quality)
# ----------------------------
def _drop_toc_like_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Drops paragraphs that look like a Table of Contents line dump.
    We keep this conservative: only remove if clearly TOC-like.
    """
    out: List[str] = []
    for p in paragraphs:
        if _looks_like_toc(p):
            continue
        out.append(p)
    return out


def _looks_like_toc(p: str) -> bool:
    s = p.strip()
    if len(s) < 40:
        return False

    # Count "numbered headings" markers like 1., 2.1, 3.2.4
    matches = list(_TOC_NUMBERED.finditer(s))
    if len(matches) < 3:
        return False

    # If it's mostly short heading-like fragments, it's likely TOC
    # Heuristic: lots of digits/dots, low lowercase ratio, not many sentence-ending punctuation marks
    digit_dot_ratio = sum(ch.isdigit() or ch == "." for ch in s) / max(1, len(s))
    lowercase_ratio = sum("a" <= ch <= "z" for ch in s) / max(1, len(s))
    sentence_punct = s.count(".") + s.count("!") + s.count("?")

    # TOC lines often have many dots/digits and fewer real sentences
    if digit_dot_ratio > 0.12 and lowercase_ratio < 0.55 and sentence_punct < 6:
        return True

    # Also: if it’s one long line with many numbered markers and very few verbs-like punctuation
    if len(matches) >= 6 and sentence_punct < 4:
        return True

    return False


# ----------------------------
# Sentence splitting
# ----------------------------
def _sentenceize(paragraph: str, splitter: str) -> str:
    paragraph = paragraph.strip()
    if not paragraph:
        return ""

    if splitter == "none":
        return paragraph

    if splitter == "icu":
        sentences = _split_sentences_icu(paragraph)
        if sentences:
            return "\n".join(sentences)

    if splitter in ("blingfire", "icu"):
        sentences = _split_sentences_blingfire(paragraph)
        if sentences:
            return "\n".join(sentences)

    # Fallback: simple regex-based
    sentences = _split_sentences_regex(paragraph)
    return "\n".join(sentences)


def _split_sentences_icu(text: str) -> List[str]:
    icu = _try_import_pyicu()
    if icu is None:
        return []

    try:
        bi = icu.BreakIterator.createSentenceInstance(icu.Locale.getDefault())
        bi.setText(text)
        out: List[str] = []
        start = bi.first()
        while True:
            end = bi.nextBoundary()
            if end == icu.BreakIterator.DONE:
                break
            sent = text[start:end].strip()
            if sent:
                out.append(sent)
            start = end
        return out
    except Exception:
        return []


def _split_sentences_blingfire(text: str) -> List[str]:
    fn = _try_import_blingfire()
    if fn is None:
        return []
    try:
        # blingfire returns sentences separated by \n
        s = fn(text)
        parts = [x.strip() for x in s.split("\n")]
        return [x for x in parts if x]
    except Exception:
        return []


def _split_sentences_regex(text: str) -> List[str]:
    # Very conservative fallback: split on [.?!] followed by space + uppercase/number/quote/bracket
    # Works “okay” for many languages but not perfect.
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[\{]*[\p{Lu}\p{N}])", text)
    out = [p.strip() for p in pieces if p.strip()]
    return out if out else [text.strip()]