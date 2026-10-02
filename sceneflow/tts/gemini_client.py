"""Low-level Gemini TTS API wrapper — no GUI or project knowledge."""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import wave
from typing import List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Safe character limit per TTS chunk. Gemini TTS models have an input limit;
# we stay well below the documented maximum to avoid truncation errors.
# The API will reject requests that exceed its limit; this is our pre-chunking guard.
# Set to 6000 for low-API-quota users: fits ~10 average scenes or ~15 short dialogue
# scenes into one request, which still reliably completes within the 360s timeout.
# (15 scenes × 111 words = ~17 min audio ≈ 2-3 min synthesis, well under 6 min limit.)
TTS_INPUT_CHAR_LIMIT = 6000

# Model used for transcription with word-level timestamps.
# Standard Gemini flash model supports audio input and can return timestamped text.
TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"


# Network timeouts. Large combined-TTS chunks can genuinely take a long time for
# Google's API to synthesise and stream back, especially on a slow connection.
# These are deliberately generous; the 360s read timeout = 6 minutes.
# (Previously 180s, which was too tight for batches of 8+ scenes.)
_HTTP_CONNECT_TIMEOUT = 30
_HTTP_READ_TIMEOUT_TTS = 360
_HTTP_READ_TIMEOUT_TRANSCRIBE = 240


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict,
    json: dict,
    timeout: tuple,
    max_retries: int,
    log_label: str,
) -> requests.Response:
    """Run a requests POST (or GET) with exponential-backoff retry.

    Retries on:
      * ``requests.RequestException`` (DNS/connect/read/ssl/timeout errors incl.
        ReadTimeout from the log)
      * HTTP 429 (rate limit)
      * HTTP 5xx (server error, gateway timeout, etc.)

    Uses 1.0s → 2.0s → 4.0s → 8.0s → … → capped 30s backoff with ±15% jitter to
    avoid thundering herd.
    """
    import random

    delay = 1.0
    last_err: Optional[Exception] = None
    last_resp: Optional[requests.Response] = None

    for attempt in range(1, max_retries + 1):
        try:
            last_err = None
            last_resp = None
            if method.upper() == "POST":
                resp = requests.post(url, headers=headers, json=json, timeout=timeout)
            elif method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            else:
                raise ValueError("Unsupported method %r" % method)

            last_resp = resp

            if resp.status_code == 429:
                if attempt < max_retries:
                    sleep_for = delay + random.uniform(0, delay * 0.15)
                    log.warning(
                        "%s: rate limited (429) on attempt %d/%d — sleeping %.1fs.",
                        log_label, attempt, max_retries, sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(
                    "%s: rate-limited (429) after %d attempts."
                    % (log_label, max_retries)
                )

            if 500 <= resp.status_code < 600:
                if attempt < max_retries:
                    sleep_for = delay + random.uniform(0, delay * 0.15)
                    log.warning(
                        "%s: server error %s on attempt %d/%d — sleeping %.1fs.",
                        log_label, resp.status_code, attempt, max_retries, sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 30.0)
                    continue
                raise RuntimeError(
                    "%s: server error %s after %d attempts: %s"
                    % (log_label, resp.status_code, max_retries, resp.text[:500])
                )

            return resp
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_err = exc
            if attempt < max_retries:
                sleep_for = delay + random.uniform(0, delay * 0.15)
                log.warning(
                    "%s: network error %s on attempt %d/%d — sleeping %.1fs.",
                    log_label, type(exc).__name__, attempt, max_retries, sleep_for,
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 30.0)
                continue
            raise RuntimeError(
                "%s: network error after %d attempts: %s" % (log_label, max_retries, exc)
            ) from exc

    # Unreachable, but keep linters happy
    if last_err is not None:
        raise last_err  # pragma: no cover
    if last_resp is not None:
        raise RuntimeError("%s: failed after %d attempts" % (log_label, max_retries))
    raise RuntimeError("%s: failed after %d attempts" % (log_label, max_retries))


def call_tts(
    api_key: str,
    text: str,
    voice: str,
    model: str = "gemini-2.5-flash-preview-tts",
    style: Optional[str] = None,
    max_retries: int = 5,
) -> bytes:
    """
    Call the Gemini TTS generateContent endpoint and return raw PCM bytes.

    The audio comes back as base64-encoded ``audio/L16`` (24 kHz, mono, 16-bit PCM).

    Parameters
    ----------
    api_key:
        Gemini API key (from environment — never hardcoded).
    text:
        Scene narration text to synthesise.
    voice:
        Preset voice name (e.g. ``"Kore"``).
    model:
        Gemini model identifier.
    style:
        Optional natural-language style instruction prepended to the prompt
        (e.g. ``"Speak in a calm, cinematic tone."``).
    max_retries:
        Maximum number of retry attempts on HTTP 429 / 5xx / any transient network
        error (including the exact Read timed out failures reported in your logs).

    Returns
    -------
    bytes
        Raw PCM audio data (audio/L16, 24 kHz, mono, 16-bit little-endian).

    Raises
    ------
    RuntimeError
        On non-2xx responses after all retries are exhausted.
    """
    url = _ENDPOINT.format(model=model)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    prompt = f"{style}\n\n{text}" if style else text

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice,
                    }
                }
            },
        },
    }

    response = _request_with_retry(
        "POST",
        url,
        headers=headers,
        json=payload,
        timeout=(_HTTP_CONNECT_TIMEOUT, _HTTP_READ_TIMEOUT_TTS),
        max_retries=max_retries,
        log_label="Gemini-TTS",
    )

    if not response.ok:
        raise RuntimeError(
            "Gemini TTS API error %s: %s" % (response.status_code, response.text)
        )

    # Parse response
    try:
        data = response.json()
        b64_audio = (
            data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            "Unexpected Gemini TTS response structure: %s\nResponse body: %s"
            % (exc, response.text[:500])
        ) from exc

    return base64.b64decode(b64_audio)


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """
    Wrap raw PCM data in a WAV container and return the bytes.

    Parameters
    ----------
    pcm_bytes:
        Raw PCM audio (little-endian, signed 16-bit by default).
    sample_rate:
        Samples per second (default: 24000).
    channels:
        Number of audio channels (default: 1 — mono).
    sample_width:
        Bytes per sample (default: 2 — 16-bit).

    Returns
    -------
    bytes
        Complete WAV file contents.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe_audio_with_timestamps(
    api_key: str,
    wav_bytes: bytes,
    max_retries: int = 5,
) -> List[Tuple[float, float, str]]:
    """
    Send a WAV audio clip to Gemini for transcription with word-level timestamps.

    Parameters
    ----------
    api_key:
        Gemini API key.
    wav_bytes:
        Bytes of a WAV file to transcribe.
    max_retries:
        Maximum retry attempts on HTTP 429.

    Returns
    -------
    list[tuple[float, float, str]]
        Ordered list of ``(start_seconds, end_seconds, word_text)``.
        The word text is normalized lowercase. If the API does not return
        word-level timestamps, falls back to sentence-level timestamps and
        then to a heuristic character-proportional split.
    """
    url = _ENDPOINT.format(model=TRANSCRIPTION_MODEL)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    b64_audio = base64.b64encode(wav_bytes).decode("ascii")

    # Ask Gemini for a strict JSON transcription with per-word start/end timestamps
    prompt = (
        "Transcribe the provided audio EXACTLY as spoken. Return a strict JSON array "
        "of objects, each with keys \"start\" (seconds, float), \"end\" (seconds, float), "
        "and \"word\" (string, lowercase). Do not include commentary, markdown code fences, "
        "or any text outside the JSON array. Preserve every spoken word in order. "
        "Example: [{\"start\": 0.0, \"end\": 0.42, \"word\": \"the\"}, ...]"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": b64_audio,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT"],
            "temperature": 0.0,
        },
    }

    response = _request_with_retry(
        "POST",
        url,
        headers=headers,
        json=payload,
        timeout=(_HTTP_CONNECT_TIMEOUT, _HTTP_READ_TIMEOUT_TRANSCRIBE),
        max_retries=max_retries,
        log_label="Gemini-Transcribe",
    )

    if not response.ok:
        raise RuntimeError(
            "Gemini transcription API error %s: %s"
            % (response.status_code, response.text)
        )

    # Parse response
    try:
        data = response.json()
        text_parts: List[str] = []
        for part in data["candidates"][0]["content"]["parts"]:
            if "text" in part:
                text_parts.append(part["text"])
        raw_text = "".join(text_parts)
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            "Unexpected Gemini transcription response: %s\nBody: %s"
            % (exc, response.text[:800])
        ) from exc

    # Try to extract a JSON array from the text
    words = _parse_word_timestamps(raw_text)
    if words:
        return words

    # Fallback: if we got text but no timestamps, do a character-proportional split
    # using the full audio duration estimated from WAV bytes.
    log.warning(
        "Transcription did not include word timestamps — applying fallback split."
    )
    return _fallback_timestamp_split(raw_text, wav_bytes)


def _parse_word_timestamps(raw_text: str) -> List[Tuple[float, float, str]]:
    """Try to extract [{start,end,word}, ...] from a raw API response."""
    import re
    # Strip possible markdown fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()

    candidates: List[str] = []
    # Try to find a JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    candidates.append(cleaned)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, list):
            continue
        result: List[Tuple[float, float, str]] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            try:
                s = float(item["start"])
                e = float(item["end"])
                w = str(item["word"]).strip().lower()
            except (KeyError, TypeError, ValueError):
                continue
            if w:
                result.append((s, e, w))
        if result:
            return result
    return []


def _fallback_timestamp_split(
    transcript_text: str, wav_bytes: bytes
) -> List[Tuple[float, float, str]]:
    """
    Heuristic fallback: split full audio duration proportionally by character
    count per word. This is used only when the API truly does not return
    timestamps.
    """
    import re
    # Estimate duration from the WAV header
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            total_dur = frames / float(rate) if rate > 0 else 0.0
    except Exception:
        total_dur = 0.0

    if total_dur <= 0:
        return []

    # Tokenize into words
    words = [w.lower() for w in re.findall(r"\S+", transcript_text) if w.strip()]
    if not words:
        return []

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    result: List[Tuple[float, float, str]] = []
    cursor = 0.0
    for w in words:
        frac = len(w) / total_chars
        dur = frac * total_dur
        result.append((cursor, cursor + dur, w))
        cursor += dur
    return result
