# ASR Shootout — Benchmark for Indian Conversational Speech

## Benchmarking Deepgram, Whisper, IndicConformer, and Sarvam Saaras on Bangalore locality names in Hinglish conversational speech.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/asr-shootout.git
cd asr-shootout

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your Deepgram + Sarvam API keys

# 4. Recordings are included in the repo (recordings/ folder)
#    26 samples: 20 self-recorded + 6 cross-accent

# 5. Run the pipeline
python src/run_pipeline.py
```

**For Colab (recommended):** Open `notebooks/asr_benchmark.ipynb` → Runtime → T4 GPU → Run All

---

## Project Structure

```
asr-shootout/
├── recordings/                    # 26 audio samples (self-recorded + cross-accent)
├── data/
│   ├── ground_truth.json          # Reference transcripts for all 26 samples
│   └── metadata.csv               # Audio file metadata and conditions
├── src/
│   ├── metrics.py                 # WER, CER, Entity Accuracy, Jaro-Winkler, Recoverability
│   ├── transcribe_deepgram.py     # Deepgram Nova-3 (API, baseline)
│   ├── transcribe_whisper.py      # Whisper large-v3 (OSS, faster-whisper)
│   ├── transcribe_indicconformer.py  # AI4Bharat IndicConformer (NeMo + HF fallback)
│   ├── transcribe_sarvam.py       # Sarvam Saaras v3 (API, India-specific)
│   ├── run_pipeline.py            # Main orchestrator
│   └── special_tests.py           # Hallucination test + chunk latency curve
├── notebooks/
│   └── asr_benchmark.ipynb        # Full Colab-ready notebook (recommended entry point)
├── results/                       # Benchmark outputs — CSVs, charts, latency curves
│   ├── raw_transcriptions.csv
│   ├── metrics_summary.csv
│   ├── condition_breakdown.csv
│   ├── main_comparison.png
│   ├── condition_heatmap.png
│   └── chunk_latency_curve.png
├── report/
│   └── report.md                  # Fill with your numbers after running notebook
├── requirements.txt
└── .env.example                   # Copy to .env and add your API keys
```

---

## Dataset

**Self-recorded (20 clips):** Bangalore locality names in Hinglish conversational sentences. Varied conditions:
- Quiet room (7), Traffic noise (4), Street/crowd (3), Whispered (2), Rushed (3), Phone simulation (1)

**Cross-accent (6 clips):** Same 3 localities recorded by 2 friends with different accents — tests speaker generalization.

**Open-source:** FLEURS-Hi (20 clips) as sanity-check baseline.

---

## Models Evaluated

| Model | Type | Free | Why |
|-------|------|------|-----|
| **Deepgram Nova-3** | Commercial API | ✅ 200 hrs/month | Required baseline |
| **Whisper large-v3** | OSS | ✅ Always | Dominant OSS benchmark |
| **IndicConformer** | OSS | ✅ Always | India-specific, trained on IndicSUPERB |
| **Sarvam Saaras v3** | Commercial API | ✅ ₹1000 free credits | Built for Indian languages; demo uses Bangalore localities |

---

## Metrics

Standard metrics alone don't capture what matters for this use case.

| Metric | Why |
|--------|-----|
| **Entity Accuracy** | Did the locality appear correctly? Binary. *The actual product metric.* |
| **Jaro-Winkler Similarity** | How similar is the entity guess to the correct name? Recoverable vs catastrophic. |
| **Recoverability** | Can a fuzzy-match post-processing layer fix the error? |
| WER / CER | Standard baselines — shown to be misleading for entity-heavy utterances |
| Latency | API round-trip time; tested on 2/3/5/10s chunks (VAD-gated telephony) |

---

## Special Tests

1. **Hallucination Test** — Sends pure background noise to all models. Whisper is known to generate fabricated text on short/noisy audio. This is a production safety concern.

2. **VAD Chunk Latency Curve** — Tests API latency on 2s, 3s, 5s, 10s audio chunks. Real telephony systems use VAD to send utterance-length chunks, not full files.

---

## API Key Setup

**Deepgram** (free, 200 hrs/month):
1. Sign up at [console.deepgram.com](https://console.deepgram.com)
2. Dashboard → API Keys → Create new key
3. Copy to `.env` as `DEEPGRAM_API_KEY`

**Sarvam** (free, ₹1000 credits ≈ 33 hours audio):
1. Sign up at [app.sarvam.ai](https://app.sarvam.ai)
2. Get API key from dashboard
3. Copy to `.env` as `SARVAM_API_KEY`

Your 26 recordings (~3 min total × 10 test runs) ≈ ₹15 of Sarvam credits used.

---

## Running Individual Models

```bash
# Deepgram only
python src/run_pipeline.py --models deepgram

# Multiple models
python src/run_pipeline.py --models deepgram sarvam
```

---

## Reproducing Results on Colab

1. Clone this repo: git clone https://github.com/mohdabdullahmeraj/asr-shootout.git
2. Open `notebooks/asr_benchmark.ipynb` in Colab
3. Set runtime to T4 GPU
4. Update `REPO_URL` in the first cell
5. Paste API keys in the config cell
6. Run All — recordings are included in the repo

---

## Report

See `report/report.md` for the complete benchmark findings, failure analysis, and production recommendations.

Key finding preview: **Entity Accuracy > WER** for this use case. A model can score 85% WER but still fail to capture the locality name — which is the only thing that matters for candidate routing.
