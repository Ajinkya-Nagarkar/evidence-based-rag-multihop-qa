# Hallucination-Resistant Multi-Hop Question Answering

**CSCI 566 – Deep Learning Course Project**

> An evidence-grounded RAG pipeline that answers multi-hop questions while resisting hallucination through hybrid retrieval, cross-encoder reranking, claim verification, and strict citation enforcement.

---

## Motivation

Large Language Models (LLMs) often produce **hallucinations** — factually incorrect or unsupported answers — especially for multi-hop question answering (QA) tasks that require reasoning across multiple documents. This project tackles the problem by building a hallucination-resistant QA pipeline that:

- **Grounds every answer in retrieved evidence** — the LLM never generates from parametric memory alone
- **Enforces evidence-first reasoning** — context is presented before the question to prime factual extraction
- **Verifies factual consistency** before accepting a generated response
- **Routes questions by complexity** — simple queries use a lightweight model; complex bridge/comparison queries use a larger one
- **Avoids unsupported generations** through post-verification decision logic with optional abstention

We evaluate primarily on the [HotpotQA](https://hotpotqa.github.io/) dataset, which contains multi-hop questions with gold supporting-fact annotations across distractor and full-wiki settings.

---

## Pipeline Architecture

The system is a seven-stage pipeline orchestrated by `pipeline/eval.py`. Each stage is a self-contained module with a `from_config()` factory for YAML-driven instantiation.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         HotpotQA JSON                                │
│                  (dev_distractor / dev_fullwiki)                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  1. Data Loader  │  Parses JSON → Passage / HotpotQAExample dataclasses
              └────────┬────────┘
                       │
                       ▼
         ┌──────────────────────────┐
         │  2. Hybrid Retriever     │  BM25 (sparse) + FAISS (dense) via Reciprocal Rank Fusion
         │     (Multi-hop)          │  Hop 1 → entity extraction → Hop 2 (fuzzy title + hybrid)
         └────────────┬─────────────┘
                      │  top-20 candidates
                      ▼
            ┌───────────────────┐
            │  3. Reranker       │  Cross-encoder (BGE-reranker-v2-m3) re-scores passages
            │     + Sentence     │  Cross-encoder sentence selection for SP attribution
            │     Selection      │
            └─────────┬─────────┘
                      │  top-5 reranked passages + selected sentences
                      ▼
          ┌─────────────────────────┐
          │  4. Prompt Builder       │  Evidence-first prompt with numbered facts
          │     + Complexity Router  │  Complexity score → routes to small or large model
          └───────────┬─────────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  5. Generator      │  Ollama LLM inference (Llama / Qwen / Gemma)
            │     (+ Specialist  │  Optional specialist mode: large model selects facts,
            │      Mode)         │  small model generates answer
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  6. Verifier       │  Checks if answer is supported by evidence
            │     (overlap/NLI/  │  Lexical overlap · NLI entailment · QA confidence
            │      QA modes)     │  Yes/no scoring with evidence diversity signals
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  7. Decider        │  Post-verification decision layer
            │                    │  Optional retry, confidence scoring, fact dedup
            └─────────┬─────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Predictions   │  {answer, supporting_facts, verification, confidence}
              │  + Evaluation  │  Official HotpotQA EM / F1 / SP EM / Joint F1
              └───────────────┘
```

### Stage Details

| #   | Module                       | Key Technique                                                                                                                                                                                                                        |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `pipeline/data_loader.py`    | Parses HotpotQA JSON into typed `Passage`, `SupportingFact`, `HotpotQAExample` dataclasses with lazy loading and deduplication                                                                                                       |
| 2   | `pipeline/indexer.py`        | **BM25** (term-frequency) + **FAISS** (dense cosine similarity via Ollama embeddings) fused with **Reciprocal Rank Fusion (RRF)**. Multi-hop iterative retrieval with LLM entity extraction and **fuzzy title matching** (RapidFuzz) |
| 3   | `pipeline/reranker.py`       | **Cross-encoder** passage re-ranking (`BAAI/bge-reranker-v2-m3`) + cross-encoder sentence-level selection with rank-aware guarantees and title-overlap boosting                                                                      |
| 4   | `pipeline/prompt_builder.py` | Evidence-first prompt construction with numbered fact citation, complexity scoring (query length, bridge keywords, confidence gap, sentence count), and model routing                                                                |
| 5   | `pipeline/generator.py`      | Ollama LLM backend with **specialist mode** (large model for fact selection, small model for answer extraction), multi-stage JSON parsing (clean → targeted regex → free-form), repair retries, and answer normalization             |
| 6   | `pipeline/verifier.py`       | Three verification modes: lexical overlap, NLI entailment (RoBERTa-large-MNLI), QA confidence (DistilBERT-squad). Evidence-quality scoring for yes/no questions with diversity and substance checks                                  |
| 7   | `pipeline/decider.py`        | Post-verification decision layer: optional retry with feedback, confidence scoring (verification × reranker), fact deduplication, configurable abstention policy                                                                     |
| —   | `pipeline/embedder.py`       | Ollama-based embedding client with batched encoding, L2 normalization, and connection pooling                                                                                                                                        |
| —   | `pipeline/eval.py`           | End-to-end orchestrator with multi-threaded execution, checkpoint/resume, stage-by-stage recall diagnostics, and verification statistics                                                                                             |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml              # Central pipeline config (retriever, reranker, generator, verifier, etc.)
│   └── prompts.yaml              # All LLM prompt templates (indexer, citation, standard, specialist, yes/no)
├── data/
│   ├── hotpot_dev_distractor_v1.json   # 7,405 examples (10 context passages each, 2 gold + 8 distractors)
│   ├── hotpot_dev_fullwiki_v1.json     # Full-wiki setting
│   └── hotpot_test_fullwiki_v1.json    # Test set (no gold answers)
├── pipeline/
│   ├── __init__.py
│   ├── data_loader.py            # HotpotQA parser → Passage / HotpotQAExample dataclasses
│   ├── embedder.py               # Ollama embedding client (batched, L2-normalized)
│   ├── indexer.py                # BM25 + FAISS hybrid retriever with multi-hop & fuzzy title matching
│   ├── reranker.py               # Cross-encoder passage reranking + sentence selection
│   ├── prompt_builder.py         # Evidence-first prompt construction + complexity routing
│   ├── generator.py              # Ollama LLM generation (standard + specialist mode)
│   ├── verifier.py               # Answer verification (overlap / NLI / QA modes)
│   ├── decider.py                # Post-verification decision layer (retry, confidence, abstention)
│   └── eval.py                   # End-to-end pipeline runner with checkpointing & diagnostics
├── scripts/
│   ├── __init__.py
│   ├── config.py                 # Typed YAML config loader → nested dataclasses with validation
│   ├── logger.py                 # Thread-safe colored terminal logger
│   ├── test_pipeline.py          # Quick single-example pipeline test
│   ├── hotpot_evaluate_v1.py     # Official HotpotQA evaluation script
│   ├── evaluate_custom.py        # Extended evaluation (BERTScore, fuzzy match, title metrics)
│   └── analyze_predictions.py    # Deep error analysis (by type, difficulty, verification status)
├── index_cache_global/           # Persisted FAISS + BM25 indices (auto-generated)
├── results/
│   ├── archive/                  # Historical prediction runs
│   ├── predictions/              # Model prediction JSONs
│   └── metrics/                  # Evaluation metrics outputs
└── reports/
    ├── CSCI 566 Course Project Pre-Proposal.pdf
    └── Mid_Term_Report.pdf
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kabrashrey/hallucination_resistant_multihop_qna.git
cd hallucination_resistant_multihop_qna
```

### 2. Create a Virtual Environment

Using **venv**:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or using **conda**:

```bash
conda create --name rag_multihop python=3.10
conda activate rag_multihop
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Additional packages used by the extended evaluation and retriever:

```bash
pip install rapidfuzz bert-score requests
```

### 4. Install and Configure Ollama

The pipeline uses [Ollama](https://ollama.com/) for both embedding and LLM generation. Install Ollama, then pull the required models:

```bash
# Start the Ollama server
ollama serve

# Embedding model (used for dense retrieval + sentence selection)
ollama pull qwen3-embedding:8b

# Generator models (routed by complexity)
ollama pull llama3.2:latest       # Small model — simple questions
ollama pull qwen3:8b              # Large model — complex questions

# Entity extraction model (used during multi-hop retrieval)
ollama pull qwen2.5:7b
```

> **GPU Parallelism**: For throughput, set `OLLAMA_NUM_PARALLEL=3` (matching `eval.parallel_workers` in `default.yaml`) before starting `ollama serve`.

---

## Usage

### Quick Test (Single Example)

Runs one example through the full pipeline to verify setup:

```bash
python -m scripts.test_pipeline
```

### Full Evaluation

Run the pipeline on the HotpotQA dev-distractor split:

```bash
# Run on first 100 examples (default)
python -m pipeline.eval

# Run on all 7,405 examples
python -m pipeline.eval --limit 0

# Run on dev-fullwiki split
python -m pipeline.eval --split dev_fullwiki --output results/predictions/predictions_fullwiki.json

# Run with official evaluation
python -m pipeline.eval --eval data/hotpot_dev_distractor_v1.json --output results/predictions/predictions.json

# Re-run specific failed IDs
python -m pipeline.eval --ids results/failed_ids.json --output results/predictions/predictions_retry.json
```

The pipeline supports **checkpoint/resume** — if interrupted, it will automatically pick up from the last saved checkpoint.

### Evaluation Scripts

**Official HotpotQA metrics** (EM, F1, SP EM, SP F1, Joint EM, Joint F1):

```bash
python scripts/hotpot_evaluate_v1.py results/predictions/predictions.json data/hotpot_dev_distractor_v1.json
```

**Extended custom metrics** (BERTScore, fuzzy match, title metrics, containment):

```bash
python scripts/evaluate_custom.py \
    --predictions results/predictions/predictions.json \
    --gold data/hotpot_dev_distractor_v1.json \
    --output results/metrics/metrics_custom.json
```

**Deep error analysis** (error breakdown by type/difficulty, SP analysis, verification analysis):

```bash
python scripts/analyze_predictions.py \
    results/predictions/predictions.json \
    data/hotpot_dev_distractor_v1.json
```

---

## Configuration

All system parameters are driven by two YAML files in `configs/`:

| File           | Purpose                                                                                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `default.yaml` | Pipeline logic: retriever alpha weights, RRF-k, candidate pool size, reranker thresholds, complexity routing, generator model selection, verifier mode, evaluation parallelism |
| `prompts.yaml` | LLM instruction templates: entity extraction, citation-based QA, standard QA, specialist answer generation, yes/no reasoning                                                   |

### Key Configuration Parameters

```yaml
# Hybrid retrieval scoring
retriever.alpha: 0.7 # Dense weight (1-α = BM25 weight)
retriever.alpha_bridge: 0.5 # Per-type override for bridge questions
retriever.rrf_k: 20 # RRF smoothing constant

# Reranker
reranker.model_name: "BAAI/bge-reranker-v2-m3"
reranker.top_k: 5 # Passages fed to prompt builder
reranker.sentence_passage_limit: 3 # Passages with sentence selection

# Model routing
prompt_builder.complexity_routing_threshold: 0.50
generator.model_small: "llama3.2:latest"
generator.model_large: "qwen3:8b"
generator.specialist_mode: false # Two-call pipeline: large selects facts, small answers

# Verification
verifier.mode: "overlap" # "overlap", "nli", or "qa"
verifier.support_threshold: 0.55
```

### Programmatic Config Access

```python
from scripts.config import load_config

cfg = load_config("configs/default.yaml")

print(cfg.retriever.alpha)                  # 0.7
print(cfg.generator.specialist_mode)        # False
print(cfg.verifier.mode)                    # "overlap"
print(cfg.prompts.builder_citation)         # Full citation prompt template
```

---

## Key Design Decisions

### Hybrid Retrieval with RRF

BM25 catches exact keyword matches that dense retrieval misses, while FAISS dense retrieval captures semantic similarity that BM25 cannot. Reciprocal Rank Fusion merges both ranking lists using only rank positions (not raw scores), making the fusion robust to score-scale differences.

### Multi-Hop with Fuzzy Title Matching

Hop 1 retrieves with the original question. An LLM extracts bridge entities from hop-1 passages, then hop 2 uses both **fuzzy title matching** (RapidFuzz `token_set_ratio`) and hybrid retrieval with a reformulated query. This deterministic title matching is critical for comparison questions where entity names must match exactly.

### Cross-Encoder Sentence Selection

Rather than using bi-encoder cosine similarity (fast but imprecise), the reranker uses the same cross-encoder loaded for passage re-ranking to score `(query, sentence)` pairs jointly. This models token-level interactions and dramatically improves supporting fact precision at negligible latency cost (~15-25 pairs per question).

### Evidence-First Prompting with Citation Selection

The prompt places evidence _before_ the question to prime the LLM for extraction rather than generation. Facts are numbered, and the LLM outputs `supporting_fact_numbers` that map back to `(title, sentence_index)` pairs — enabling precise SP EM computation without title hallucination.

### Specialist Mode

An optional two-call pipeline where a large model selects the relevant facts and a small model extracts the answer from only those facts. This separates reasoning (which benefits from scale) from extraction (which doesn't).

### Post-Verification Decision Layer

The Decider combines verification scores with reranker confidence to produce a final confidence estimate. In evaluation mode, it keeps unsupported answers (abstaining scores 0 EM), but penalizes confidence. In production mode, it can abstain to avoid hallucinated answers.

---

## Team Members

Ajinkya Nagarkar · Augusto Rivas Constante · Avery Novick · Ishan Chakrabarti · Leonardo Robles-Angeles · Priyanka Rani · Shreyansh Kabra
