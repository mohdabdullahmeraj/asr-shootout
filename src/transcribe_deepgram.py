"""
Deepgram Nova-3 ASR client — REST API fallback.
"""

import os
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def transcribe_file(
    audio_path: str | Path,
    client=None,  # kept for compatibility
    language: str = "hi",
    model: str = "nova-3",
) -> dict:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key or api_key == "your_deepgram_api_key_here":
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "words": [], "error": "DEEPGRAM_API_KEY not set"}

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "words": [], "error": f"File not found: {audio_path}"}

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        start_time = time.perf_counter()

        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": model,
            "language": language,
            "smart_format": "true",
            "punctuate": "true",
            "words": "true"
        }
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav" # Deepgram auto-detects actual type
        }

        response = requests.post(url, params=params, headers=headers, data=audio_data)
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return {"transcript": "", "confidence": None, "latency_ms": latency_ms,
                    "words": [], "error": f"Deepgram API Error: {response.text}"}

        result = response.json()
        channels = result.get("results", {}).get("channels", [])
        if not channels:
            return {"transcript": "", "confidence": None, "latency_ms": latency_ms,
                    "words": [], "error": "No channels in response"}
            
        alt = channels[0].get("alternatives", [{}])[0]
        transcript = alt.get("transcript", "")
        confidence = alt.get("confidence", None)
        words = alt.get("words", [])

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
    results = []

    for i, path in enumerate(audio_paths):
        if verbose:
            print(f"  [Deepgram] {i+1}/{len(audio_paths)}: {Path(path).name}")

        result = transcribe_file(path, language=language, model=model)
        result["filename"] = Path(path).name
        results.append(result)

        if result["error"]:
            logger.warning(f"  Failed: {result['error']}")

    return results


def transcribe_chunk_latency_test(
    audio_path: str | Path,
    chunk_durations_s: list[float] = [3.0, 5.0, 10.0],
) -> list[dict]:
    import librosa
    import soundfile as sf
    import tempfile

    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for duration in chunk_durations_s:
            n_samples = int(duration * sr)
            chunk = audio[:n_samples]  # Take first N seconds

            chunk_path = Path(tmpdir) / f"chunk_{duration}s.wav"
            sf.write(str(chunk_path), chunk, sr)

            result = transcribe_file(chunk_path)
            results.append({
                "chunk_duration_s": duration,
                "latency_ms": result["latency_ms"],
                "transcript": result["transcript"],
                "error": result["error"],
            })

    return results
