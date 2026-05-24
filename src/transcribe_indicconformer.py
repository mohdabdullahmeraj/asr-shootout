"""
AI4Bharat IndicConformer ASR client.

Primary backend: AI4Bharat's fork of NVIDIA NeMo toolkit.
Fallback: HuggingFace Transformers (openai/whisper-medium for Hindi).

The fallback activates automatically if NeMo is unavailable,
allowing the pipeline to degrade gracefully on CPU-only environments.
"""

import time
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level model cache
_nemo_model = None
_hf_model = None
_hf_processor = None


# ---------------------------------------------------------------------------
# Primary: NeMo-based IndicConformer
# ---------------------------------------------------------------------------

def get_nemo_model(model_name: str = None):
    """
    Load AI4Bharat IndicConformer via NVIDIA NeMo.
    Falls back to HuggingFace model if NeMo is not available.
    """
    global _nemo_model

    if _nemo_model is not None:
        return _nemo_model, "nemo"

    if model_name is None:
        model_name = os.environ.get(
            "INDICCONFORMER_MODEL",
            "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"
        )

    try:
        import torch
        import nemo.collections.asr as nemo_asr
        print(f"  [IndicConformer] Loading NeMo model: {model_name} ...")
        _nemo_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _nemo_model.freeze()
        _nemo_model = _nemo_model.to(device)
        print(f"  [IndicConformer] NeMo model loaded.")
        return _nemo_model, "nemo"

    except ImportError:
        logger.warning(
            "NeMo not installed. Falling back to HuggingFace model. "
            "To use IndicConformer, install AI4Bharat's NeMo fork:\n"
            "git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh"
        )
        return None, "nemo_unavailable"

    except Exception as e:
        logger.warning(f"NeMo model load failed: {e}. Trying HuggingFace fallback.")
        return None, "nemo_failed"


def get_hf_fallback_model():
    """
    Fallback: Load Whisper Medium for Hindi via HuggingFace Transformers.
    """
    global _hf_model, _hf_processor

    if _hf_model is not None:
        return _hf_model, _hf_processor

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch

        hf_model_id = "openai/whisper-medium"
        print(f"  [IndicConformer-Fallback] Loading HuggingFace model: {hf_model_id} ...")

        _hf_processor = WhisperProcessor.from_pretrained(hf_model_id)
        _hf_model = WhisperForConditionalGeneration.from_pretrained(hf_model_id)
        _hf_model.eval()

        if torch.cuda.is_available():
            _hf_model = _hf_model.cuda()

        print(f"  [IndicConformer-Fallback] HuggingFace model loaded.")
        return _hf_model, _hf_processor

    except Exception as e:
        logger.error(f"HuggingFace fallback also failed: {e}")
        return None, None


def transcribe_file_nemo(audio_path: str | Path, model) -> dict:
    """Transcribe using NeMo IndicConformer."""
    import librosa
    import soundfile as sf
    import tempfile

    audio_path = Path(audio_path)

    # Pre-process audio to 16kHz mono WAV first
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name

    try:
        model.cur_decoder = "rnnt"

        start_time = time.perf_counter()
        transcriptions = model.transcribe(
            [tmp_path],
            batch_size=1,
            language_id="hi"
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        transcript = transcriptions[0] if transcriptions else ""

    finally:
        os.remove(tmp_path)

    return {
        "transcript": transcript,
        "confidence": None,
        "latency_ms": latency_ms,
        "backend": "nemo",
        "error": None,
    }


def transcribe_file_hf(audio_path: str | Path, model, processor) -> dict:
    """Transcribe using HuggingFace Whisper fallback."""
    import librosa
    import torch

    audio_path = Path(audio_path)
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    input_features = inputs.input_features.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    latency_ms = (time.perf_counter() - start_time) * 1000

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {
        "transcript": transcript.strip(),
        "confidence": None,
        "latency_ms": latency_ms,
        "backend": "huggingface_whisper_fallback",
        "error": None,
    }


def transcribe_file(
    audio_path: str | Path,
    model=None,
    processor=None,
    backend: str = "auto",
) -> dict:
    """
    Transcribe a single audio file using IndicConformer.
    Automatically chooses NeMo (preferred) or HuggingFace (fallback).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"transcript": "", "confidence": None, "latency_ms": None,
                "backend": None, "error": f"File not found: {audio_path}"}

    # Try NeMo first
    if backend in ("auto", "nemo") and model is None:
        model, status = get_nemo_model()
        if status == "nemo" and model is not None:
            backend = "nemo"

    if backend == "nemo" and model is not None:
        try:
            return transcribe_file_nemo(audio_path, model)
        except Exception as e:
            logger.warning(f"NeMo inference failed, trying HF: {e}")
            model = None

    # HuggingFace fallback
    if model is None or processor is not None:
        hf_model, hf_processor = get_hf_fallback_model()
        if hf_model is None:
            return {
                "transcript": "",
                "confidence": None,
                "latency_ms": None,
                "backend": "unavailable",
                "error": "Both NeMo and HuggingFace backends failed to load. "
                         "This is itself a useful data point — IndicConformer "
                         "requires significant compute to deploy.",
            }
        try:
            return transcribe_file_hf(audio_path, hf_model, hf_processor)
        except Exception as e:
            return {"transcript": "", "confidence": None, "latency_ms": None,
                    "backend": "huggingface_whisper_fallback", "error": str(e)}


def transcribe_batch(
    audio_paths: list[str | Path],
    verbose: bool = True,
) -> list[dict]:
    """
    Transcribe a list of audio files using IndicConformer.
    Loads model once, reuses across calls.
    """
    # Determine backend
    nemo_model, nemo_status = get_nemo_model()
    hf_model, hf_processor = (None, None)

    if nemo_model is None:
        hf_model, hf_processor = get_hf_fallback_model()

    results = []
    for i, path in enumerate(audio_paths):
        if verbose:
            print(f"  [IndicConformer] {i+1}/{len(audio_paths)}: {Path(path).name}")

        if nemo_model is not None:
            result = transcribe_file(path, model=nemo_model, backend="nemo")
        elif hf_model is not None:
            result = transcribe_file_hf(path, hf_model, hf_processor)
        else:
            result = {
                "transcript": "",
                "confidence": None,
                "latency_ms": None,
                "backend": "unavailable",
                "error": "No backend available",
            }

        result["filename"] = Path(path).name
        results.append(result)

    return results
