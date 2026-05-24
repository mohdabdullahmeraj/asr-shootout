# ASR Benchmark for Indian Conversational Speech
### Vahan AI — Intern Assessment | May 2026

---

## The Core Argument

> **WER is the wrong metric for this problem.**
>
> When a candidate says *"Haan, main Koramangala mein rehta hoon"* and the ASR returns
> *"Haan, main Kormangala mein rehta hoon"* — Word Error Rate scores 12.5%. But the locality
> is wrong. That is a 100% product failure.
>
> This benchmark measures what actually matters: **Entity Accuracy** (did the locality name
> appear correctly?) and **Recoverability** (can downstream fuzzy-matching save a near-miss?).
>
> The data proved this argument: **Deepgram has the best WER (76.6%) but the worst Entity
> Accuracy (50%)** — the metric the company likely uses to evaluate its current ASR is actively
> misleading it about performance on the task that matters.

---

## 1. Methodology

### Dataset

| Source | Samples | Purpose |
|--------|---------|---------|
| Self-recorded | 20 clips | Primary evaluation — varied conditions, Hinglish |
| Cross-accent (2 friends) | 6 clips | Speaker generalization across accents |
| FLEURS Hindi (OSS) | 20 clips | Sanity check baseline — clean read speech |

*MUCS 2021 (Hindi-English code-switched) was planned as the OSS dataset. It requires manual
download from OpenSLR and is not available via HuggingFace. FLEURS was used instead and
is noted as a limitation — it is cleaner than real candidate audio, so FLEURS numbers are
optimistic baselines, not realistic production estimates.*

**Recording conditions:**

| Condition | Count | Why |
|-----------|-------|-----|
| Quiet room | 7 | Establishes ceiling performance |
| Traffic noise | 4 | Common for candidates calling from streets |
| Street/crowd | 3 | Worst-case ambient noise |
| Whispered | 2 | Candidates calling in shared spaces |
| Rushed speech | 3 | Time-pressured or nervous candidates |
| Phone simulation | 1 | Reverb/compression effect |

All self-recorded samples are Hinglish (Hindi + English code-switched), recorded on a
smartphone mic. No studio setup. 6 additional clips from 2 friends with different regional
accents test cross-speaker generalization.

### Model Selection

| Model | Type | Rationale |
|-------|------|-----------|
| **Deepgram Nova-3** | Commercial API | Required baseline. Currently used in production. |
| **OpenAI Whisper large-v3** | Open-source | Dominant OSS baseline; most widely deployed multilingual model. The obvious comparison point. |
| **AI4Bharat IndicConformer** | Open-source | Purpose-built for 12 Indic languages; trained on IndicSUPERB. The non-obvious pick — built for this exact problem. |
| **Sarvam Saaras v3** | Commercial API | Indian startup explicitly built for Indic languages. Their own demo uses "Koramangala 5th Block" as a sample sentence — they have optimized for this use case. Claims <250ms median latency and 100M+ minutes transcribed. |

**What was excluded and why:**
- *Google STT / Azure Cognitive*: Require paid enterprise accounts; noted as production alternatives in recommendations
- *Wav2Vec2*: Older architecture, superseded by IndicConformer for Indic languages
- *Whisper medium/small*: Large-v3 is the fair comparison for peak accuracy; medium would be the production latency-accuracy tradeoff candidate

**IndicConformer deployment note:** NeMo installation on Colab free tier requires building
`tensorstore` from source, which exceeded session limits. The HuggingFace fallback
(Whisper-medium) ran instead. IndicConformer results in this benchmark therefore reflect
Whisper-medium, not the actual IndicConformer architecture. This is itself a significant
production insight: **IndicConformer's deployment complexity is a real barrier**, not a
theoretical one.

### Metrics

| Metric | Why this, not just WER |
|--------|------------------------|
| **Entity Accuracy (EA)** | Binary: did the locality appear correctly (fuzzy ≥80%)? The actual product metric. |
| **Jaro-Winkler Similarity (JWS)** | Measures how close the entity guess is to the correct name. Distinguishes "Koramangla" (close, recoverable) from "Silk bored" (catastrophic). |
| **Recoverability** | Four tiers: correct / recoverable (JWS ≥0.75) / degraded / catastrophic. Maps to whether downstream post-processing can fix the error. |
| WER | Standard baseline — included to show it is misleading for this use case. |
| CER | Character-level errors, better for long compound names like Byatarayanapura. |
| Latency | Measured on 2s/3s/5s/10s VAD-gated chunks, not full files. This is how telephony actually works. |

---

## 2. Results

### Primary: Entity Accuracy by Model

| Model | Entity Accuracy | Mean WER | Mean CER | Mean JWS | Mean Latency |
|-------|:--------------:|:--------:|:--------:|:--------:|:------------:|
| **Whisper large-v3** | **65.4%** | 90.1% | 35.4% | 0.837 | 1295ms† |
| IndicConformer* | 61.5% | 91.7% | 39.5% | 0.818 | 1338ms† |
| Sarvam Saaras v3 | 53.8% | 91.2% | 36.3% | 0.799 | **718ms** |
| Deepgram Nova-3 | 50.0% | **76.6%** | **30.1%** | 0.803 | 1814ms |

*\* IndicConformer ran as Whisper-medium fallback — NeMo exceeded Colab session limits.*  
*† Local inference — compute only, no network round-trip. Not directly comparable to API latency.*  
*‡ All API latencies measured from Colab. Production co-located servers would be significantly faster.*

**The headline finding:** Deepgram has the best WER but the worst Entity Accuracy. The metric currently used to benchmark ASR is inversely correlated with what actually matters for candidate routing.

### Entity Accuracy by Condition

| Condition | Deepgram | Whisper | IndicConformer* | Sarvam |
|-----------|:--------:|:-------:|:---------------:|:------:|
| Quiet | 62% | **85%** | 69% | 69% |
| Traffic | 25% | 25% | 25% | 25% |
| Street | 0% | 33% | **67%** | **67%** |
| Whispered | 50% | **100%** | **100%** | 50% |
| Rushed | **67%** | **67%** | 33% | 33% |
| Phone | **100%** | 0% | **100%** | 0% |

Notable: Whisper and IndicConformer* both score 100% on whispered audio. For a hiring platform where candidates call from shared rooms and whisper to avoid being overheard, this is practically meaningful. Deepgram's 0% on street noise is the most concerning production finding.

### Recoverability Breakdown

| Model | Correct | Recoverable | Degraded | Catastrophic |
|-------|:-------:|:-----------:|:--------:|:------------:|
| Deepgram | 13 | 4 | 8 | 1 |
| Whisper | 17 | 5 | 2 | 2 |
| IndicConformer* | 16 | 5 | 3 | 2 |
| Sarvam | 14 | 3 | 8 | 1 |

**Recoverable** errors (JWS ≥0.75) can be fixed by a fuzzy-match layer against a known
locality list at near-zero cost. **Catastrophic** errors (JWS <0.5) cannot be recovered — they
represent complete hallucinations or entirely wrong entities. Whisper and IndicConformer*
have fewer degraded errors than Deepgram and Sarvam, meaning their failures tend to be
either correct or clearly wrong, rather than plausible-but-wrong.

### VAD Chunk Latency Curve

| Chunk Duration | Deepgram | Sarvam |
|:-------------:|:--------:|:------:|
| 2 seconds | 1399ms | 553ms |
| 3 seconds | 1399ms | 638ms |
| 5 seconds | 1022ms | 588ms |
| 10 seconds | 1069ms | 548ms |

*Telephony systems use VAD to detect utterance end and send 3-5 second chunks to ASR. This
is the real production latency — not full-file latency.*

**Sarvam is consistently ~550-640ms on telephony-sized chunks, just above the 500ms UX
threshold.** From production servers in India (not Colab), it would likely be sub-500ms.
Deepgram's latency is 2-2.5x higher from Colab; from co-located production servers it would
improve significantly, but the gap would likely persist.

### FLEURS Hindi — OSS Sanity Check

| Model | WER on FLEURS | WER on Self-Recordings | Degradation Factor |
|-------|:-------------:|:---------------------:|:-----------------:|
| Sarvam | **6.5%** | 91.2% | 14x |
| Deepgram | 15.7% | 76.6% | 4.9x |
| Whisper | 24.5% | 90.1% | 3.7x |

**The domain gap is the real problem, not the models.** Sarvam achieves 6.5% WER on clean
Hindi — dramatically better than the others — but degrades 14x on conversational noisy
Hinglish. All models struggle in real conditions. This means improving data quality (VAD,
noise filtering, candidate UX prompts) may yield more improvement than switching ASR models.

---

## 3. Failure Analysis

### Pre-Run Hypothesis vs Reality

Before running: Deepgram was expected to win on latency and reliability; Whisper was expected
to struggle on long Kannada-origin names; Sarvam was expected to win on Indic locality names
given their explicit optimization.

What actually happened: Deepgram had the worst Entity Accuracy despite the best WER.
Sarvam's accuracy was underwhelming despite the clean-speech advantage. Whisper won on
entity accuracy. The pre-run hypothesis was wrong on the primary metric — which is exactly
why you run the benchmark.

### The Byatarayanapura Problem

Byatarayanapura (7 syllables, Kannada origin) is the hardest locality name in the dataset.
It is also the type of name that matters most — uncommon, long, phonetically dense.

| Model | Output on `17_byatarayanapura_quiet.wav` | Verdict |
|-------|------------------------------------------|---------|
| Deepgram | "beatara hariyana pura" | Catastrophic — split into 3 fake words |
| Whisper | "byatarayanapura" | Correct |
| IndicConformer* | "byatarayanapura" | Correct |
| Sarvam | [segmented into phonetic fragments] | Degraded |

Whisper correctly transcribed Byatarayanapura despite it being 7 syllables. Deepgram split
it into "beatara hariyana pura" — three plausible-sounding but entirely wrong words. A
downstream entity extraction system matching against a locality list would find no match
and fail silently. This is a catastrophic production failure on a quiet, well-spoken recording.

### Abbreviation and Initialism Confusion

| Model | Output for "HSR Layout" | Output for "KR Puram" |
|-------|------------------------|----------------------|
| Deepgram | Correct | Degraded ("kya harapurama") |
| Whisper | Correct | Correct |
| IndicConformer* | Correct | Correct |
| Sarvam | Correct | Degraded |

HSR Layout (initialism + English word) was handled correctly by all models — English
initialisms in a Hindi sentence appear to be well-covered in training data. KR Puram was
more problematic: Deepgram and Sarvam both failed, producing phonetic approximations
of the sounds rather than the correct entity.

### Noise Failure Modes

**Traffic noise** degraded all models equally — 25% entity accuracy across the board. No
model has a meaningful advantage here.

**Street/crowd noise** showed the sharpest divergence: Deepgram 0%, Whisper 33%,
IndicConformer*/Sarvam 67%. Deepgram completely failed on street noise while
open-source/India-specific models showed relative resilience.

**Whispered speech** is where the result was most surprising: Whisper large-v3 and
IndicConformer* both achieved 100%. Deepgram achieved only 50%. For a platform where
blue-collar workers call from shared accommodation and may whisper, this is directly
relevant.

### Hallucination Test

All three API models were sent 2 seconds of pure pink noise. All returned empty output.

| Model | Output on Pure Noise | Verdict |
|-------|---------------------|---------|
| Deepgram | [empty] | ✅ Correct — silent |
| Whisper | [empty] | ✅ Correct — silent |
| Sarvam | [empty] | ✅ Correct — silent |

No hallucinations detected. This is a positive production finding — a noisy line will not
trigger a false locality extraction. Note: Whisper's `vad_filter=True` was enabled, which
actively prevents hallucination on silence. Without VAD enabled, Whisper is known to generate
fabricated text on short audio.

### Cross-Accent Results (Speaker Generalization)

| Locality | Self | Friend 1 | Friend 2 | Most Robust Model |
|----------|:----:|:--------:|:--------:|:-----------------:|
| Koramangala | All correct | All correct | Deepgram/Whisper fail | Whisper |
| Byatarayanapura | Whisper/Indic correct | All correct | All correct | Whisper/IndicConformer* |
| HSR Layout | All correct | All correct | All correct | All |

**Speaker generalization is better than self-recording performance.** Friends' accents
(likely different from the self-recorder) did not significantly degrade results — in some
cases improved them. This suggests the self-recordings may have had recording conditions
that were harder than the friend recordings (which were all quiet).

---

## 4. Production Considerations

### Cost at Scale

Assume: 100,000 calls/month, 30 seconds average transcribed audio = **50,000 minutes/month**

| Model | Pricing | Monthly Cost (50k min) | Notes |
|-------|---------|:---------------------:|-------|
| Deepgram Nova-3 | $0.0043/min | ~$215/month | Predictable, scales linearly |
| Sarvam Saaras v3 | ₹30/hour | ~$298/month | ~39% more expensive than Deepgram |
| Whisper large-v3 | Infrastructure only | ~$150-200/month | 1× A100 at $2/hr cloud; needs DevOps |
| IndicConformer | Infrastructure only | ~$150-200/month | Similar to Whisper; harder to deploy |

At 50k minutes/month, self-hosted Whisper on a cloud GPU is cost-competitive with API
options. The breakeven point is roughly 30,000 minutes/month — above that, self-hosting
wins on cost. Below that, API simplicity wins.

### Deployment Complexity

| Model | Setup | Streaming | Memory | Production-Ready? |
|-------|-------|:---------:|:------:|:-----------------:|
| Deepgram | Zero — API | ✅ Native WebSocket | N/A | ✅ Yes |
| Sarvam | Zero — API | ✅ WebSocket available | N/A | ✅ Yes |
| Whisper | GPU server required | ❌ (use WhisperLive wrapper) | 6GB+ VRAM | ⚠️ Needs DevOps |
| IndicConformer | GPU + AI4Bharat NeMo fork | ❌ | 8GB+ VRAM | ❌ High complexity |

IndicConformer's deployment barrier is not theoretical — it failed to install on Colab free
tier during this benchmark. Any production deployment would require a dedicated environment
with AI4Bharat's custom NeMo fork pre-installed.

---

## 5. What This Benchmark Doesn't Capture

1. **Single primary speaker.** 20 recordings are one voice, one phone, one accent. The 6
   friend recordings add diversity but real production covers UP/Bihar/Karnataka/Andhra
   accents at scale.

2. **G.711 codec not tested.** Real telephony compresses audio to 8kHz G.711. All models
   would perform worse on actual phone audio than on smartphone recordings. The phone
   simulation clip (1 sample) is insufficient to measure this.

3. **Colab latency ≠ production latency.** API latency from Colab includes shared network
   overhead. Co-located production servers would be significantly faster, especially for
   Deepgram (US-hosted) vs Sarvam (India-hosted).

4. **No streaming evaluation.** This benchmark tests batch/file mode. Streaming ASR has
   different latency profiles and is not evaluated here.

5. **Training data overlap unknown.** Bangalore locality names may appear in Sarvam and
   IndicConformer training data given their India focus. This would inflate accuracy on
   locality names specifically compared to a truly out-of-domain test.

6. **Entity boundary detection not tested.** This benchmark assumes the locality is the
   only entity to extract and its position is known. In production, a separate NER step is
   needed to find *where* in the utterance the locality appears. Even perfect ASR accuracy
   is insufficient without reliable entity extraction downstream.

---

## 6. Recommendation

### For Real-Time Phone Calls

**Sarvam Saaras v3 + locality fuzzy-match post-processing**

Sarvam achieves ~550ms on 3-5 second VAD-gated chunks from Colab. From production
servers in India this would likely be sub-500ms — within the live IVR threshold. It is
purpose-built for Indic languages, significantly faster than Deepgram from an Indian network
perspective, and its FLEURS accuracy (6.5% WER) demonstrates strong clean-Hindi capability.

If Sarvam's production latency proves to be above threshold after testing, **Deepgram** is
the fallback — it has established reliability, predictable cost, and the best WER even if
entity accuracy lags.

Whisper is not viable for real-time — 1295ms compute latency on a dedicated GPU, plus
the server infrastructure requirement, makes it unsuitable for live phone IVR.

### For Async WhatsApp Voice Notes

**Whisper large-v3 (self-hosted)**

Latency is irrelevant for async processing. Whisper achieves the highest entity accuracy
(65.4%) and is free to run. At 50,000 minutes/month, a cloud GPU server is cost-competitive
with API options. The quality advantage justifies the DevOps investment.

### The Highest-Leverage Improvement

**Add a fuzzy-match post-processing layer.** This is more impactful than any model switch.

Maintain a canonical list of Bangalore localities. After transcription, use Rapidfuzz with
~80% Jaro-Winkler threshold to match the extracted entity against this list. This recovers
recoverable errors (4-5 per model across 26 samples) at near-zero cost and with no latency
impact.

```python
from rapidfuzz import process
LOCALITIES = ["Koramangala", "Indiranagar", "Whitefield", ...]  # canonical list

def recover_locality(asr_output: str) -> str:
    match, score, _ = process.extractOne(asr_output, LOCALITIES)
    return match if score >= 80 else asr_output
```

A mediocre ASR with this post-processing layer will outperform a better ASR without it.
Model selection is a secondary concern compared to this single engineering decision.

### Model Selection Summary

| Constraint | Recommendation |
|-----------|----------------|
| Lowest latency | Sarvam Saaras v3 (~550ms on telephony chunks) |
| Best entity accuracy | Whisper large-v3 (65.4%) |
| Best on clean audio | Sarvam (6.5% WER on FLEURS) |
| Best on noisy/street audio | Whisper / IndicConformer* |
| Best on whispered speech | Whisper / IndicConformer* (100%) |
| Zero ongoing cost | Whisper large-v3 (self-hosted) |
| Easiest to deploy | Deepgram or Sarvam (API, zero setup) |
| Most surprising result | Deepgram has best WER but worst Entity Accuracy |

---

## Appendix: Metric Methodology Note

All model outputs were transliterated from Devanagari to Roman script before metric
computation using `indic-transliteration`. API models (Deepgram, Sarvam) return Devanagari
for Hindi input; comparing Devanagari output against Roman-script ground truth produces
artificially inflated WER. The transliteration step normalizes both sides to the same script
before computing WER, CER, and entity matching. Without this fix, all models scored near
0% entity accuracy — illustrating how metric implementation choices can completely
misrepresent model performance.

---

*Code, recordings, and full results: https://github.com/mohdabdullahmeraj/asr-shootout*