"""
Whisper large-v3 ASR client using faster-whisper backend.

faster-whisper uses CTranslate2 for inference — same accuracy as openai-whisper
but significantly faster and lower memory usage.

Includes per-segment confidence scoring and hallucination flagging
via no_speech_prob threshold.
"""

import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level model cache — only load once per session
_whisper_model = None


def get_model(model_size: str = "medium", device: str = "auto", compute_type: str = "auto"):
    """
    Load faster-whisper model. Cached in module scope to avoid
    re-loading on every call (model load takes ~30s for large-v3).

    Args:
        model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
                    Default is 'medium' for Colab free tier compatibility 
                    (large-v3 needs ~4.5GB VRAM and may OOM with other models loaded).
                    Consider 'distil-large-v3' for production latency.
        device: 'cuda', 'cpu', or 'auto' (auto-detects GPU)
        compute_type: 'float16' (GPU), 'int8' (CPU/quantized), or 'auto'
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        )

    # Auto-detect compute settings
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"  [Whisper] Loading {model_size} model on {device} ({compute_type})...")
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"  [Whisper] Model loaded.")
    return _whisper_model


def transcribe_file(
    audio_path: str | Path,
    model=None,
    model_size: str = "medium",
    language: str = "hi",
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
    word_timestamps: bool = False,
) -> dict:
    """
    Transcribe a single audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model: Optional pre-loaded WhisperModel (pass to reuse)
        model_size: Model variant if model not provided
        language: ISO language code ('hi' for Hindi)
                  None = auto-detect language
        beam_size: Beam search width (5 = good balance)
        condition_on_previous_text: False avoids hallucination propagation

    Returns:
        dict with transcript, confidence, latency_ms, segments, error
    """
    if model is None:
        model = get_model(model_size)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "segments": [], "error": f"File not found: {audio_path}"}

    try:
        start_time = time.perf_counter()

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            suppress_tokens=[],
            no_speech_threshold=0.6,     # Flag segments likely to be silence
            log_prob_threshold=-1.0,     # Flag low-confidence segments
        )

        # Materialize the generator
        segment_list = list(segments)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Build full transcript
        transcript = " ".join(s.text.strip() for s in segment_list).strip()

        # Average confidence across segments (from avg_logprob)
        # Convert log prob to rough confidence: exp(avg_logprob)
        import math
        confidences = [
            math.exp(s.avg_logprob)
            for s in segment_list
            if hasattr(s, "avg_logprob") and s.avg_logprob is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Flag potential hallucinations (no_speech_prob > 0.6)
        hallucination_flags = [
            s.no_speech_prob > 0.6
            for s in segment_list
            if hasattr(s, "no_speech_prob")
        ]
        is_likely_hallucination = any(hallucination_flags) and len(transcript.strip()) > 0

        segments_data = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "avg_logprob": getattr(s, "avg_logprob", None),
                "no_speech_prob": getattr(s, "no_speech_prob", None),
            }
            for s in segment_list
        ]

        return {
            "transcript": transcript,
            "confidence": avg_confidence,
            "latency_ms": latency_ms,
            "segments": segments_data,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "is_likely_hallucination": is_likely_hallucination,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Whisper error on {audio_path.name}: {e}")
        return {
            "transcript": "",
            "confidence": None,
            "latency_ms": None,
            "segments": [],
            "is_likely_hallucination": False,
            "error": str(e),
        }


def transcribe_batch(
    audio_paths: list[str | Path],
    model_size: str = "medium",
    language: str = "hi",
    verbose: bool = True,
) -> list[dict]:
    """
    Transcribe a list of audio files. Loads model once, reuses across calls.
    """
    model = get_model(model_size)
    results = []

    for i, path in enumerate(audio_paths):
        if verbose:
            print(f"  [Whisper] {i+1}/{len(audio_paths)}: {Path(path).name}")

        result = transcribe_file(path, model=model, model_size=model_size, language=language)
        result["filename"] = Path(path).name

        if result.get("is_likely_hallucination"):
            print(f"    ⚠️  Possible hallucination detected!")

        results.append(result)

    return results
