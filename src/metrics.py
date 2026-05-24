"""
Metrics for ASR evaluation on entity-heavy utterances.

Implements:
- WER  (Word Error Rate)
- CER  (Character Error Rate)
- EA   (Entity Accuracy) — binary, locality-level
- JWS  (Jaro-Winkler Similarity) — on entity span
- LED  (Levenshtein Edit Distance) — on entity span
- Recoverability — whether a fuzzy-match post-processing step could recover the error
"""

import re
import unicodedata
from typing import Optional

import jiwer
import statistics
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, normalize whitespace.
    Works for both Hindi (Devanagari) and English (Latin).
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation (but keep Devanagari characters)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entity_from_hypothesis(locality: str, hypothesis: str) -> Optional[str]:
    """
    Try to find the locality name (or a close match) inside the hypothesis.
    Returns the matched substring if found, else None.
    Uses sliding window + Jaro-Winkler to find best candidate.
    """
    locality_norm = normalize_text(locality)
    hypothesis_norm = normalize_text(hypothesis)
    locality_words = locality_norm.split()
    hyp_words = hypothesis_norm.split()

    if not hyp_words:
        return None

    # Exact match first
    if locality_norm in hypothesis_norm:
        return locality_norm

    # Sliding window over hypothesis words (same length as locality)
    best_score = 0.0
    best_span = None
    window = len(locality_words)

    # Check windows of locality-length and ±1
    for w in range(max(1, window - 1), window + 2):
        for i in range(len(hyp_words) - w + 1):
            span = " ".join(hyp_words[i : i + w])
            score = fuzz.ratio(locality_norm, span) / 100.0
            if score > best_score:
                best_score = score
                best_span = span

    # Return span only if meaningfully similar (>= 60%)
    return best_span if best_score >= 0.60 else None


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate using jiwer.
    Returns a float between 0.0 (perfect) and 1.0+ (terrible).
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not hyp:
        return 1.0
    try:
        return jiwer.wer(ref, hyp)
    except Exception:
        return 1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate using jiwer.
    Better than WER for long Indian locality names where character-level
    errors (Byatarayanapura vs Batarayanapura) are meaningful.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not hyp:
        return 1.0
    try:
        return jiwer.cer(ref, hyp)
    except Exception:
        return 1.0


def compute_entity_accuracy(locality: str, hypothesis: str) -> int:
    """
    Binary: did the locality name appear (approximately) correctly?
    Returns 1 (correct) or 0 (wrong).

    Uses fuzzy match with threshold >= 80% to allow minor spelling variations
    (e.g., "Koramangla" vs "Koramangala" still counts).
    """
    locality_norm = normalize_text(locality)
    hypothesis_norm = normalize_text(hypothesis)

    # Exact match
    if locality_norm in hypothesis_norm:
        return 1

    # Fuzzy match — partial token ratio
    score = fuzz.partial_ratio(locality_norm, hypothesis_norm) / 100.0
    return 1 if score >= 0.80 else 0


def compute_jaro_winkler(locality: str, hypothesis: str) -> float:
    """
    Jaro-Winkler similarity between the locality name and the
    best matching span found in the hypothesis.

    Returns a float between 0.0 (no match) and 1.0 (perfect match).
    Higher scores indicate closer phonetic match to the target locality.
    """
    locality_norm = normalize_text(locality)
    hypothesis_norm = normalize_text(hypothesis)

    if not hypothesis_norm:
        return 0.0

    # Try to find the entity span in hypothesis
    entity_span = extract_entity_from_hypothesis(locality, hypothesis)
    if entity_span is None:
        # Fall back to best partial match across full hypothesis
        entity_span = hypothesis_norm

    return JaroWinkler.similarity(locality_norm, entity_span)


def compute_levenshtein_on_entity(locality: str, hypothesis: str) -> int:
    """
    Raw Levenshtein edit distance between the locality name and its
    best matching span in the hypothesis.

    Lower = better. 0 = perfect. > 5 = likely catastrophic failure.
    """
    locality_norm = normalize_text(locality)
    entity_span = extract_entity_from_hypothesis(locality, hypothesis)

    if entity_span is None:
        return len(locality_norm)  # Max possible distance

    return Levenshtein.distance(locality_norm, entity_span)


def compute_recoverability(locality: str, hypothesis: str) -> str:
    """
    Classify the transcription error into one of four tiers:
    - "correct"      : EA=1
    - "recoverable"  : EA=0, JWS >= 0.75 (close enough for fuzzy post-processing)
    - "degraded"     : EA=0, JWS 0.5–0.75
    - "catastrophic" : EA=0, JWS < 0.5 (hallucination or completely wrong entity)
    """
    ea = compute_entity_accuracy(locality, hypothesis)
    if ea == 1:
        return "correct"

    jws = compute_jaro_winkler(locality, hypothesis)
    if jws >= 0.75:
        return "recoverable"
    elif jws >= 0.50:
        return "degraded"
    else:
        return "catastrophic"


# ---------------------------------------------------------------------------
# Aggregate Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    locality: str,
    reference: str,
    hypothesis: str,
    latency_ms: Optional[float] = None,
    confidence: Optional[float] = None,
) -> dict:
    """
    Compute all metrics for a single sample.
    Returns a flat dict suitable for appending to a pandas DataFrame.
    """
    return {
        "locality": locality,
        "reference": reference,
        "hypothesis": hypothesis if hypothesis else "[NO OUTPUT]",
        "wer": round(compute_wer(reference, hypothesis), 4),
        "cer": round(compute_cer(reference, hypothesis), 4),
        "entity_accuracy": compute_entity_accuracy(locality, hypothesis),
        "jaro_winkler_similarity": round(compute_jaro_winkler(locality, hypothesis), 4),
        "levenshtein_on_entity": compute_levenshtein_on_entity(locality, hypothesis),
        "recoverability": compute_recoverability(locality, hypothesis),
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


def aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate a list of per-sample metric dicts into summary statistics.
    """
    if not results:
        return {}

    wers = [r["wer"] for r in results]
    cers = [r["cer"] for r in results]
    eas = [r["entity_accuracy"] for r in results]
    jwss = [r["jaro_winkler_similarity"] for r in results]
    leds = [r["levenshtein_on_entity"] for r in results]
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    confidences = [r["confidence"] for r in results if r.get("confidence") is not None]

    recoverability_counts = {
        "correct": sum(1 for r in results if r["recoverability"] == "correct"),
        "recoverable": sum(1 for r in results if r["recoverability"] == "recoverable"),
        "degraded": sum(1 for r in results if r["recoverability"] == "degraded"),
        "catastrophic": sum(1 for r in results if r["recoverability"] == "catastrophic"),
    }

    n = len(results)
    return {
        "n_samples": n,
        "mean_wer": round(sum(wers) / n, 4),
        "mean_cer": round(sum(cers) / n, 4),
        "entity_accuracy_pct": round(sum(eas) / n * 100, 1),
        "mean_jaro_winkler": round(sum(jwss) / n, 4),
        "mean_levenshtein_entity": round(sum(leds) / n, 2),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "max_latency_ms": round(sorted(latencies)[-1], 1) if latencies else None,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "recoverability": recoverability_counts,
    }
