"""
Deepgram Nova-3 ASR client — SDK v6 compatible.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from deepgram import DeepgramClient

logger = logging.getLogger(__name__)


def get_client() -> DeepgramClient:
    """Initialize Deepgram client from environment variable."""
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key or api_key == "your_deepgram_api_key_here":
        raise ValueError("DEEPGRAM_API_KEY not set.")
    return DeepgramClient(api_key)


def transcribe_file(
    audio_path: str | Path,
    client: Optional[DeepgramClient] = None,
    language: str = "hi",
    model: str = "nova-3",
) -> dict:
    if client is None:
        client = get_client()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "words": [], "error": f"File not found: {audio_path}"}

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        start_time = time.perf_counter()

        response = client.listen.v1.media.transcribe_file(
            request=audio_data,
            model=model,
            language=language,
            smart_format=True,
            punctuate=True,
            words=True,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript or ""
        confidence = alt.confidence

        words = []
        if hasattr(alt, "words") and alt.words:
            words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in alt.words
            ]

        return {
            "transcript": transcript,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "words": words,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Deepgram error on {audio_path.name}: {e}")
        return {
            "transcript": "",
            "confidence": None,
            "latency_ms": None,
            "words": [],
            "error": str(e),
        }


def transcribe_batch(
    audio_paths: list[str | Path],
    language: str = "hi",
    model: str = "nova-3",
    verbose: bool = True,
) -> list[dict]:
    client = get_client()
    results = []

    for i, path in enumerate(audio_paths):
        if verbose:
            print(f"  [Deepgram] {i+1}/{len(audio_paths)}: {Path(path).name}")

        result = transcribe_file(path, client=client, language=language, model=model)
        result["filename"] = Path(path).name
        results.append(result)

        if result["error"]:
            logger.warning(f"  Failed: {result['error']}")

    return results


def transcribe_chunk_latency_test(
    audio_path: str | Path,
    chunk_durations_s: list[float] = [3.0, 5.0, 10.0],
) -> list[dict]:
    """
    Special test: measure latency on audio chunks of varying duration.

    This simulates how telephony systems work — VAD detects end of utterance
    and sends a short chunk (typically 2-5 seconds) to ASR.

    Returns list of {chunk_duration_s, latency_ms, transcript} dicts.
    """
    import librosa
    import soundfile as sf
    import tempfile

    client = get_client()
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for duration in chunk_durations_s:
            n_samples = int(duration * sr)
            chunk = audio[:n_samples]  # Take first N seconds

            chunk_path = Path(tmpdir) / f"chunk_{duration}s.wav"
            sf.write(str(chunk_path), chunk, sr)

            result = transcribe_file(chunk_path, client=client)
            results.append({
                "chunk_duration_s": duration,
                "latency_ms": result["latency_ms"],
                "transcript": result["transcript"],
                "error": result["error"],
            })

    return results
