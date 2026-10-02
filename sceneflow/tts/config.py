"""TTS configuration — reads credentials from environment variables only."""
from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"

# Suggested voice names (soft list — not validated; let the API surface unknown names)
SUGGESTED_VOICES = ["Kore", "Puck", "Zephyr", "Charon", "Fenrir", "Leda", "Orus", "Aoede"]


class MissingApiKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not set in the environment."""


def get_api_key() -> str:
    """
    Return the Gemini API key from the ``GEMINI_API_KEY`` environment variable.
    Used for TTS / voiceover generation.

    Raises :class:`MissingApiKeyError` if the variable is absent or empty.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingApiKeyError(
            "Set the GEMINI_API_KEY environment variable before generating voiceover."
        )
    return key


def get_image_api_key() -> str:
    """
    Return the Gemini API key for image generation from ``GEMINI_IMAGE_API_KEY``.
    Falls back to ``GEMINI_API_KEY`` if ``GEMINI_IMAGE_API_KEY`` is not set.

    Raises :class:`MissingApiKeyError` if neither variable is set.
    """
    key = os.environ.get("GEMINI_IMAGE_API_KEY", "").strip()
    if key:
        return key
    # Fall back to the main key
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingApiKeyError(
            "Set the GEMINI_IMAGE_API_KEY environment variable before generating images."
        )
    return key
