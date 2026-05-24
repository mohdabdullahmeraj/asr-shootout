"""
Sarvam Saaras v3 ASR client.

Uses the Sarvam REST API for Hindi and other Indic language transcription.
No SDK required — plain HTTP POST with multipart form data.

API docs: https://docs.sarvam.ai/api-reference/speech-to-text
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"       # Latest model (saarika:v2.5 also available)
SARVAM_LANGUAGE = "hi-IN"        # Hindi India (also supports kn-IN for Kannada)


def get_api_key() -> str:
    """Get Sarvam API key from environment."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key or api_key == "your_sarvam_api_key_here":
        raise ValueError(
            "SARVAM_API_KEY not set. "
            "Sign up at https://app.sarvam.ai to get ₹1000 free credits."
        )
    return api_key


def transcribe_file(
    audio_path: str | Path,
    api_key: Optional[str] = None,
    language: str = SARVAM_LANGUAGE,
    model: str = SARVAM_MODEL,
) -> dict:
    """
    Transcribe a single audio file using Sarvam Saaras API.

    Args:
        audio_path: Path to audio file (WAV, MP3, M4A, OGG)
        api_key: Sarvam API key (reads from env if not provided)
        language: Language code ('hi-IN', 'kn-IN', 'ta-IN', etc.)
        model: Sarvam model ('saaras:v3')

    Returns:
        dict with transcript, latency_ms, confidence, error
    """
    if api_key is None:
        api_key = get_api_key()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "error": f"File not found: {audio_path}"}

    headers = {
        "api-subscription-key": api_key,
    }

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Determine MIME type
        suffix = audio_path.suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(suffix, "audio/wav")

        files = {
            "file": (audio_path.name, audio_bytes, mime_type),
        }
        data = {
            "model": model,
            "language_code": language,
            "mode": "transcribe",
        }

        start_time = time.perf_counter()
        response = requests.post(
            SARVAM_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=30,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return {
                "transcript": "",
                "confidence": None,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        result = response.json()
        transcript = result.get("transcript", "")

        # Sarvam returns per-chunk transcripts in some modes
        # Handle both list and string responses
        if isinstance(transcript, list):
            transcript = " ".join(t.get("transcript", "") for t in transcript)

        return {
            "transcript": transcript,
            "confidence": result.get("language_probability", None),   # If returned by API
            "latency_ms": latency_ms,
            "raw_response": result,
            "error": None,
        }

    except requests.Timeout:
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "error": "Request timed out after 30s"}
    except Exception as e:
        logger.error(f"Sarvam error on {audio_path.name}: {e}")
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "error": str(e)}


def transcribe_batch(
    audio_paths: list[str | Path],
    language: str = SARVAM_LANGUAGE,
    model: str = SARVAM_MODEL,
    verbose: bool = True,
) -> list[dict]:
    """
    Transcribe a list of audio files via the Sarvam API.
    Runs sequentially to respect API rate limits.
    """
    api_key = get_api_key()
    results = []

    for i, path in enumerate(audio_paths):
        if verbose:
            print(f"  [Sarvam] {i+1}/{len(audio_paths)}: {Path(path).name}")

        result = transcribe_file(path, api_key=api_key, language=language, model=model)
        result["filename"] = Path(path).name

        if result["error"]:
            logger.warning(f"  Failed: {result['error']}")

        results.append(result)

    return results
