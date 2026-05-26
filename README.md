# Evidence-Based RAG for Multi-Hop Q&A

> A hallucination-resistant Retrieval-Augmented Generation pipeline for multi-hop question answering, evaluated on the full HotpotQA distractor dev set (7,405 examples) in a completely zero-shot setting.

---

## Results

| Metric | Score |
|---|---|
| Exact Match (EM) | 46.8% |
| Token F1 | 59.5% |
| Joint F1 | 43.2% |
| BERTScore F1 | **94.3%** |
| Answers ≥ 80% semantic similarity | **99.1%** |

> Joint F1 improved from **19.6% → 43.2%** across pipeline iterations. Retrieval architecture improvements contributed **5× more gain** than model upgrades alone.

---

## Overview

Large language models hallucinate when asked multi-hop questions that require chaining evidence across multiple documents. This project builds a modular, evidence-grounded RAG pipeline that:

- **Grounds every answer in retrieved evidence** — the LLM never generates from parametric memory alone
- **Enforces citation selection** — the model selects from pre-enumerated numbered facts instead of generating potentially hallucinated references
- **Verifies factual consistency** via NLI before accepting a generated response
- **Handles multi-hop reasoning** using iterative two-hop retrieval for bridge questions

---

## Pipeline Architecture

```
HotpotQA JSON
      │
      ▼
┌─────────────────┐
│  1. Data Loader  │  Parses JSON → typed dataclasses
└────────┬────────┘
         ▼
┌──────────────────────────┐
│  2. Hybrid Retriever      │  BM25 + FAISS via Reciprocal Rank Fusion
│     (Multi-hop)           │  Hop 1 → entity extraction → Hop 2 (fuzzy title + hybrid)
└────────────┬─────────────┘
             ▼
   ┌───────────────────┐
   │  3. Reranker       │  Cross-encoder (BGE-reranker-v2-m3)
   │  + Sentence Select │  Sentence-level scoring for supporting fact attribution
   └─────────┬─────────┘
             ▼
 ┌─────────────────────────┐
 │  4. Prompt Builder       │  Evidence-first prompt with numbered facts + citation selection
 └───────────┬─────────────┘
             ▼
   ┌───────────────────┐
   │  5. Generator      │  gemma4:31b via Ollama (fully local, zero cloud dependency)
   └─────────┬─────────┘
             ▼
   ┌───────────────────┐
   │  6. Verifier       │  NLI entailment (DeBERTa) — checks answer is supported by evidence
   └─────────┬─────────┘
             ▼
   ┌───────────────────┐
   │  7. Decider        │  Confidence blending → accept / retry / abstain
   └─────────┬─────────┘
             ▼
      Predictions + Evaluation
      (EM / F1 / SP F1 / Joint F1 / BERTScore)
```

---

## Key Design Decisions

**Hybrid Retrieval with RRF** — BM25 catches exact keyword matches; FAISS dense retrieval captures semantic similarity. Reciprocal Rank Fusion merges both using rank positions, making fusion robust to score-scale differences.

**Multi-Hop with Fuzzy Title Matching** — Hop 1 retrieves with the original question. An LLM extracts bridge entities; Hop 2 uses RapidFuzz `token_set_ratio` for fuzzy title matching, handling surface-form variations ("US" vs "United States"). Critical for bridge questions.

**Citation Selection** — Instead of asking the LLM to generate citations freely, it selects from a pre-numbered list of retrieved facts. This eliminates hallucinated references entirely.

**NLI-Based Verification** — `cross-encoder/nli-deberta-v3-small` scores each `(supporting_fact, answer)` pair, aggregating entailment probabilities into a continuous support score before the decider accepts the answer.

---

## Repository Structure

```
.
├── configs/
│   ├── default.yaml        # Pipeline config (retriever, reranker, generator, verifier)
│   └── prompts.yaml        # All LLM prompt templates
├── data/                   # HotpotQA JSON files (download separately — see below)
├── pipeline/
│   ├── data_loader.py
│   ├── embedder.py
│   ├── indexer.py          # BM25 + FAISS hybrid retriever
│   ├── reranker.py         # Cross-encoder reranking + sentence selection
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── verifier.py
│   ├── decider.py
│   └── eval.py             # End-to-end pipeline runner with checkpointing
├── scripts/
│   ├── config.py
│   ├── hotpot_evaluate_v1.py   # Official HotpotQA evaluation script
│   ├── evaluate_custom.py      # BERTScore, fuzzy match, containment metrics
│   └── analyze_predictions.py  # Error analysis by type, difficulty, verification
├── results/
└── requirements.txt
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ajinkya-Nagarkar/evidence-based-rag-multihop-qa.git
cd evidence-based-rag-multihop-qa
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install rapidfuzz bert-score requests
```

### 4. Download the Data

Download HotpotQA from [hotpotqa.github.io](https://hotpotqa.github.io/) and place the files in the `data/` folder:

```
data/hotpot_dev_distractor_v1.json
data/hotpot_dev_fullwiki_v1.json
data/hotpot_test_fullwiki_v1.json
```

### 5. Install and Configure Ollama

```bash
# Install Ollama from https://ollama.com, then pull required models:
ollama serve

ollama pull qwen3-embedding:8b   # Dense retrieval embeddings
ollama pull gemma4:31b           # Final generator (20 GB)
ollama pull qwen2.5:7b           # Fallback generator + entity extraction
```

---

## Usage

### Quick Test (Single Example)

```bash
python -m scripts.test_pipeline
```

### Full Evaluation

```bash
# First 100 examples
python -m pipeline.eval

# Full dev set (7,405 examples)
python -m pipeline.eval --limit 0

# With official evaluation metrics
python -m pipeline.eval --eval data/hotpot_dev_distractor_v1.json \
                         --output results/predictions/predictions.json
```

### Evaluation Scripts

```bash
# Official HotpotQA metrics (EM, F1, SP F1, Joint F1)
python scripts/hotpot_evaluate_v1.py \
    results/predictions/predictions.json \
    data/hotpot_dev_distractor_v1.json

# Extended metrics (BERTScore, fuzzy match, containment)
python scripts/evaluate_custom.py \
    --predictions results/predictions/predictions.json \
    --gold data/hotpot_dev_distractor_v1.json \
    --output results/metrics/metrics_custom.json

# Error analysis
python scripts/analyze_predictions.py \
    results/predictions/predictions.json \
    data/hotpot_dev_distractor_v1.json
```

---

## License

MIT License