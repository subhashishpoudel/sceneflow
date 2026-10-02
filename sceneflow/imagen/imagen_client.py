"""Low-level Gemini image generation using gemini-*-image models."""
from __future__ import annotations

import base64
import logging

import requests

log = logging.getLogger(__name__)

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Best available image model on the free Gemini API
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"


def generate_image(
    api_key: str,
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
) -> bytes:
    """
    Call a Gemini image-generation model and return raw image bytes.

    Returns
    -------
    bytes
        Raw PNG/JPEG image bytes.

    Raises
    ------
    RuntimeError
        On non-2xx responses or unexpected response structure.
    """
    url = _ENDPOINT.format(model=model)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }

    log.debug("Image gen API call: model=%s prompt=%.80s…", model, prompt)
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if not response.ok:
        raise RuntimeError(
            f"Gemini Imagen API error {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
        # Image comes back as inlineData in the first part
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])
        raise RuntimeError(f"No image in response. Parts: {parts}")
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Unexpected image response structure: {exc}\n"
            f"Response body: {response.text[:500]}"
        ) from exc
