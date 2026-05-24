"""
ASR Shootout — Main Pipeline Orchestrator
==========================================
Run all models, compute all metrics, save results.

Usage:
    python src/run_pipeline.py

    # Run specific models only
    python src/run_pipeline.py --models deepgram sarvam

    # Run special tests
    python src/run_pipeline.py --special-tests
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent dir to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.transcribe_deepgram as deepgram_mod
import src.transcribe_whisper as whisper_mod
import src.transcribe_indicconformer as indic_mod
import src.transcribe_sarvam as sarvam_mod
from src.metrics import compute_all_metrics, aggregate_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = ROOT / "recordings"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "deepgram": deepgram_mod,
    "whisper": whisper_mod,
    "indicconformer": indic_mod,
    "sarvam": sarvam_mod,
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_ground_truth() -> dict:
    """Load ground truth JSON, return dict keyed by filename."""
    gt_path = DATA_DIR / "ground_truth.json"
    with open(gt_path) as f:
        data = json.load(f)

    index = {}
    for rec in data["recordings"] + data.get("friend_recordings", []):
        index[rec["filename"]] = rec
    return index


def load_metadata() -> pd.DataFrame:
    """Load metadata CSV with all recording info."""
    meta_path = DATA_DIR / "metadata.csv"
    return pd.read_csv(meta_path)


def get_recording_files() -> list[Path]:
    """Get all audio files from recordings directory."""
    extensions = [".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"]
    files = []
    for ext in extensions:
        files.extend(RECORDINGS_DIR.glob(f"*{ext}"))
    return sorted(files)


# ---------------------------------------------------------------------------
# Per-model transcription runner
# ---------------------------------------------------------------------------

def run_model_on_recordings(
    model_name: str,
    recording_files: list[Path],
    ground_truth: dict,
) -> list[dict]:
    """
    Run a single model on all recording files and compute metrics.
    Returns list of result dicts (one per file).
    """
    module = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Running: {model_name.upper()}")
    print(f"{'='*60}")

    try:
        kwargs = {"verbose": True}
        if model_name == "whisper":
            kwargs["model_size"] = os.environ.get("WHISPER_MODEL_SIZE", "medium")
            
        raw_results = module.transcribe_batch(
            [str(f) for f in recording_files],
            **kwargs
        )
    except Exception as e:
        return [{
            "model": model_name,
            "filename": f.name,
            "condition": "unknown",
            "speaker": "unknown",
            "transcript": "",
            "error": str(e),
            "wer": None,
            "cer": None,
            "entity_accuracy": None,
            "jaro_winkler_similarity": None,
            "levenshtein_on_entity": None,
            "recoverability": "unavailable",
            "latency_ms": None,
            "confidence": None,
            "backend": model_name,
            "is_hallucination": False,
        } for f in recording_files]

    assert len(raw_results) == len(recording_files), (
        f"{model_name}: got {len(raw_results)} results for {len(recording_files)} files"
    )
    
    # Join with ground truth and compute metrics
    enriched = []
    for raw, audio_path in zip(raw_results, recording_files):
        filename = audio_path.name
        gt = ground_truth.get(filename, {})

        locality = gt.get("locality", "")
        reference = gt.get("reference_transcript", "")
        hypothesis = raw.get("transcript", "")

        metrics = compute_all_metrics(
            locality=locality,
            reference=reference,
            hypothesis=hypothesis,
            latency_ms=raw.get("latency_ms"),
            confidence=raw.get("confidence"),
        )

        record = {
            "model": model_name,
            "filename": filename,
            "condition": gt.get("condition", "unknown"),
            "speaker": gt.get("speaker", "unknown"),
            **metrics,
            "backend": raw.get("backend", model_name),
            "is_hallucination": raw.get("is_likely_hallucination", False),
            "error": raw.get("error"),
        }
        enriched.append(record)

    return enriched


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    models: list[str] = None,
    run_special_tests: bool = True,
) -> pd.DataFrame:
    """
    Main pipeline: load data → run all models → compute metrics → save results.

    Args:
        models: List of model names to run. None = all models.
        run_special_tests: Whether to run hallucination + latency tests

    Returns:
        DataFrame with all results
    """
    load_dotenv()

    if models is None:
        models = list(MODELS.keys())

    print(f"\n🚀 ASR Shootout Pipeline")
    print(f"Models: {', '.join(models)}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load recordings
    recording_files = get_recording_files()
    if not recording_files:
        raise FileNotFoundError(
            f"No audio files found in {RECORDINGS_DIR}. "
            "Please add your recordings first."
        )
    print(f"\n📂 Found {len(recording_files)} audio files in {RECORDINGS_DIR}")

    ground_truth = load_ground_truth()

    # 2. Run each model on self-recordings
    all_results = []
    for model_name in models:
        results = run_model_on_recordings(model_name, recording_files, ground_truth)
        all_results.extend(results)

    # 3. Save raw transcriptions
    df = pd.DataFrame(all_results)
    raw_path = RESULTS_DIR / "raw_transcriptions.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n💾 Saved raw transcriptions: {raw_path}")

    # 4. Compute per-model aggregate metrics
    print("\n📊 Summary Metrics:")
    print("=" * 80)

    summaries = []
    for model_name in models:
        model_df = df[df["model"] == model_name]
        model_results = model_df[model_df["error"].isna()].to_dict("records")
        agg = aggregate_results(model_results)
        agg["model"] = model_name
        summaries.append(agg)

        print(f"\n{model_name.upper()}")
        print(f"  Entity Accuracy:     {agg.get('entity_accuracy_pct', 'N/A')}%")
        print(f"  Mean WER:            {agg.get('mean_wer', 'N/A'):.4f}")
        print(f"  Mean CER:            {agg.get('mean_cer', 'N/A'):.4f}")
        print(f"  Mean Jaro-Winkler:   {agg.get('mean_jaro_winkler', 'N/A'):.4f}")
        print(f"  Mean Latency:        {agg.get('mean_latency_ms', 'N/A')} ms")
        print(f"  Median Latency:      {agg.get('median_latency_ms', 'N/A')} ms")
        print(f"  Max Latency:         {agg.get('max_latency_ms', 'N/A')} ms")
        print(f"  Recoverability:      {agg.get('recoverability', 'N/A')}")

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Saved metrics summary: {summary_path}")

    # 5. Condition breakdown
    print("\n📊 Entity Accuracy by Condition:")
    condition_breakdown = df.groupby(["model", "condition"])["entity_accuracy"].mean() * 100
    print(condition_breakdown.round(1).to_string())

    condition_path = RESULTS_DIR / "condition_breakdown.csv"
    condition_breakdown.reset_index().to_csv(condition_path, index=False)

    # 6. Special tests
    if run_special_tests:
        print("\n\n🔬 Special Tests")
        try:
            from src.special_tests import run_hallucination_test, run_chunk_latency_test
            print("  → Run hallucination + chunk latency tests via Colab notebook")
            print("     (requires real audio file for chunk latency test)")
        except ImportError:
            pass

    print(f"\n✅ Pipeline complete. Results in: {RESULTS_DIR}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Shootout Pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--special-tests",
        action="store_true",
        help="Run hallucination + chunk latency tests",
    )

    args = parser.parse_args()
    run_pipeline(
        models=args.models,
        run_special_tests=args.special_tests,
    )
