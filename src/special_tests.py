"""
Special ASR evaluation tests.

1. Hallucination Test
   Sends noise-only audio to each model to check for spurious output.
   Models should return empty string on silence/noise. Any non-empty
   output on a noise-only clip is a potential hallucination.

2. Chunk Latency Curve
   Measures API latency as a function of audio chunk duration (2s, 3s, 5s, 10s).
   Simulates VAD-gated telephony where utterance chunks are sent incrementally.
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Hallucination Test
# ---------------------------------------------------------------------------

def generate_noise_clip(
    duration_s: float = 2.0,
    sr: int = 16000,
    noise_type: str = "pink",
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Generate a synthetic noise-only audio clip (no speech).
    Used to test if models hallucinate when given only background noise.

    Args:
        duration_s: Duration in seconds
        sr: Sample rate
        noise_type: 'white' (random), 'pink' (1/f — more realistic traffic)
        output_path: Where to save. If None, saves to a temp file.

    Returns:
        Path to generated audio file
    """
    n_samples = int(duration_s * sr)

    if noise_type == "white":
        noise = np.random.randn(n_samples) * 0.1

    elif noise_type == "pink":
        # Generate pink noise (1/f) via spectral shaping
        freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
        freqs[0] = 1.0  # Avoid divide by zero
        spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        pink_spectrum = spectrum / np.sqrt(freqs)
        noise = np.fft.irfft(pink_spectrum, n=n_samples)
        noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.15

    else:
        noise = np.random.randn(n_samples) * 0.05

    # Clip to valid range
    noise = np.clip(noise, -1.0, 1.0).astype(np.float32)

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))
    else:
        output_path = Path(output_path)

    sf.write(str(output_path), noise, sr)
    return output_path


def run_hallucination_test(
    noise_types: list[str] = ["white", "pink"],
    durations_s: list[float] = [1.0, 2.0, 3.0],
    save_noise_files: bool = True,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """
    Run hallucination test across all 4 models.
    Generates noise clips and sends them to each model.

    Returns:
        dict mapping model_name → list of {duration, noise_type, transcript, flagged}
    """
    from dotenv import load_dotenv
    load_dotenv()

    import src.transcribe_deepgram as deepgram_mod
    import src.transcribe_whisper as whisper_mod
    import src.transcribe_sarvam as sarvam_mod
    import src.transcribe_indicconformer as indic_mod

    if output_dir is None:
        output_dir = Path("results/hallucination_clips")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate noise clips
    noise_clips = []
    for noise_type in noise_types:
        for duration in durations_s:
            clip_name = f"noise_{noise_type}_{duration}s.wav"
            clip_path = output_dir / clip_name if save_noise_files else None
            path = generate_noise_clip(
                duration_s=duration,
                noise_type=noise_type,
                output_path=clip_path,
            )
            noise_clips.append({
                "path": path,
                "noise_type": noise_type,
                "duration_s": duration,
            })

    print(f"Generated {len(noise_clips)} noise clips for hallucination test")

    results = {}
    model_names = ["deepgram", "whisper", "sarvam", "indicconformer"]

    for model_name in model_names:
        print(f"\n  [Hallucination Test] Running {model_name}...")
        model_results = []

        for clip in noise_clips:
            try:
                if model_name == "deepgram":
                    result = deepgram_mod.transcribe_file(clip["path"])
                elif model_name == "whisper":
                    result = whisper_mod.transcribe_file(clip["path"])
                elif model_name == "sarvam":
                    result = sarvam_mod.transcribe_file(clip["path"])
                elif model_name == "indicconformer":
                    result = indic_mod.transcribe_file(clip["path"])
                else:
                    continue

                transcript = result.get("transcript", "").strip()
                # Flagged if model returned non-empty text on a noise-only clip
                flagged_hallucination = len(transcript) > 0

                entry = {
                    "duration_s": clip["duration_s"],
                    "noise_type": clip["noise_type"],
                    "transcript": transcript if transcript else "[EMPTY — correct]",
                    "flagged_hallucination": flagged_hallucination,
                    "confidence": result.get("confidence"),
                    "latency_ms": result.get("latency_ms"),
                    "error": result.get("error"),
                }
                model_results.append(entry)

                status = "🚨 HALLUCINATED" if flagged_hallucination else "✅ silent"
                print(f"    {clip['noise_type']} {clip['duration_s']}s → {status}: '{transcript[:60]}'")

            except Exception as e:
                logger.warning(f"    {model_name} failed on noise clip: {e}")
                model_results.append({
                    "duration_s": clip["duration_s"],
                    "noise_type": clip["noise_type"],
                    "transcript": "",
                    "flagged_hallucination": False,
                    "error": str(e),
                })

        results[model_name] = model_results

    return results


# ---------------------------------------------------------------------------
# 2. Chunk Latency Curve Test
# ---------------------------------------------------------------------------

def run_chunk_latency_test(
    audio_path: str | Path,
    chunk_durations_s: list[float] = [2.0, 3.0, 5.0, 10.0],
    n_repeats: int = 3,
) -> dict:
    """
    Test how API latency scales with audio chunk duration.

    Telephony context: VAD detects utterance end → sends 2-10s chunk to ASR.
    We want the latency curve, not just a single data point.

    Args:
        audio_path: A sample audio file to use (will be trimmed to each duration)
        chunk_durations_s: Duration variants to test
        n_repeats: Number of repeats per duration (for stable measurement)

    Returns:
        dict mapping model_name → list of {duration_s, mean_latency_ms, std_latency_ms}
    """
    import librosa
    import src.transcribe_deepgram as deepgram_mod
    import src.transcribe_sarvam as sarvam_mod

    audio_path = Path(audio_path)
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    deepgram_client = deepgram_mod.get_client()

    results = {"deepgram": [], "sarvam": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        for duration in chunk_durations_s:
            print(f"\n  Testing chunk duration: {duration}s")
            n_samples = int(duration * sr)
            chunk = audio[:n_samples]
            chunk_path = Path(tmpdir) / f"chunk_{duration}s.wav"
            sf.write(str(chunk_path), chunk, sr)

            for model_name in ["deepgram", "sarvam"]:
                latencies = []
                for repeat in range(n_repeats):
                    try:
                        if model_name == "deepgram":
                            r = deepgram_mod.transcribe_file(chunk_path, client=deepgram_client)
                        else:
                            r = sarvam_mod.transcribe_file(chunk_path)

                        if r.get("latency_ms") is not None:
                            latencies.append(r["latency_ms"])
                    except Exception as e:
                        logger.warning(f"  {model_name} chunk test failed: {e}")

                if latencies:
                    mean_lat = sum(latencies) / len(latencies)
                    std_lat = float(np.std(latencies))
                    results[model_name].append({
                        "duration_s": duration,
                        "mean_latency_ms": round(mean_lat, 1),
                        "std_latency_ms": round(std_lat, 1),
                        "n_samples": len(latencies),
                    })
                    print(f"    {model_name}: {mean_lat:.0f}ms ± {std_lat:.0f}ms")

    return results


