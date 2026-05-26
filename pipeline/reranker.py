import os
import re
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from pipeline.data_loader import Passage
from pipeline.embedder import OllamaEmbedder
from pipeline.indexer import RetrievalResult
from scripts.config import get_best_device
from scripts.logger import get_logger

log = get_logger("reranker")


@dataclass
class RerankResult:
    passage: Passage
    score: float                              
    rank: int                                 
    retrieval_rank: int                       
    retrieval_score: float                    
    supporting_sentences: List[str] = field(default_factory=list)
    supporting_sentence_indices: List[int] = field(default_factory=list)
    sentence_scores: List[float] = field(default_factory=list)
    hop: int = 0

class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        sentence_model_name: str = "qwen3-embedding:8b",
        ollama_base_url: str = "http://localhost:11434",
        device: str = "auto",
        sentence_score_threshold: float = 0.25,
        max_sentences_per_passage: int = 5,
        batch_size: int = 32,
        sentence_passage_limit: int = 3,
        title_overlap_boost: float = 0.05,
    ):
        if device == "auto":
            device = get_best_device()
        self.model_name = model_name
        self.device = device
        self.sentence_score_threshold = sentence_score_threshold
        self.max_sentences_per_passage = max_sentences_per_passage
        self.batch_size = batch_size
        self.sentence_passage_limit = sentence_passage_limit
        self.title_overlap_boost = title_overlap_boost

        log.info(f"Loading cross-encoder: {model_name}")
        self._cross_encoder = CrossEncoder(model_name, device=device)

        log.info(f"Loading sentence encoder via Ollama: {sentence_model_name}")
        self._sentence_encoder = OllamaEmbedder(
            model=sentence_model_name,
            base_url=ollama_base_url,
            batch_size=batch_size,
        )

        log.success("Reranker ready.")

    @classmethod
    def from_config(cls, cfg=None) -> "Reranker":
        from scripts.config import load_config
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)
        r = cfg.reranker
        # Use retriever's Ollama URL — the reranker uses it for sentence embedding,
        # which is a retrieval-side concern, not a generator-side concern.
        ollama_url = cfg.retriever.ollama_base_url
        return cls(
            model_name=r.model_name,
            sentence_model_name=r.sentence_model_name,
            ollama_base_url=ollama_url,
            device=r.device,
            sentence_score_threshold=r.sentence_score_threshold,
            max_sentences_per_passage=r.max_sentences_per_passage,
            batch_size=r.batch_size,
            sentence_passage_limit=getattr(r, "sentence_passage_limit", 3),
            title_overlap_boost=getattr(r, "title_overlap_boost", 0.05)
        )

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
        select_sentences: bool = True,
    ) -> List[RerankResult]:
        if not candidates:
            return []
        
        pairs = [(query, c.passage.title_text) for c in candidates]

        log.step(f"Cross-encoder scoring {len(pairs)} passages...")
        scores: np.ndarray = self._cross_encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        results = [
            RerankResult(
                passage=cand.passage,
                score=float(score),
                rank=0,
                retrieval_rank=cand.rank,
                retrieval_score=cand.score,
                hop=cand.hop,
            )
            for cand, score in zip(candidates, scores)
        ]

        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        for i, r in enumerate(results):
            r.rank = i

        if select_sentences:
            self._select_supporting_sentences(query, results)

        return results

    def _normalize_tokens(self, text: str) -> set:
        return {
            re.sub(r"[^a-z0-9]+", "", t.lower())
            for t in text.split()
            if re.sub(r"[^a-z0-9]+", "", t.lower())
        }

    def _select_supporting_sentences(
        self, query: str, 
        results: List[RerankResult]
        ) -> None:
        """Select supporting sentences using cross-encoder scoring for high precision.
        
        Uses the same cross-encoder loaded for passage re-ranking to score (query, sentence) pairs. 
        This is much more precise than bi-encoder cosine similarity because it models token-level 
        interactions between query and sentence. The total number of pairs is small (~15-25), so latency is negligible.
        """

        active_results = results[: self.sentence_passage_limit]
        passage_sentence_ranges: List[Tuple[int, int]] = []
        all_sentences: List[str] = []
        # We also need to construct a mapping back to the original sentence index in the passage
        # because we only append non-empty sentences to all_sentences
        sentence_to_orig_idx_mapping: List[List[int]] = []

        for r in active_results:
            start = len(all_sentences)
            orig_indices = []
            for i, s in enumerate(r.passage.sentences):
                s_strip = s.strip()
                if s_strip:
                    all_sentences.append(s_strip)
                    orig_indices.append(i)
            passage_sentence_ranges.append((start, start + len(orig_indices)))
            sentence_to_orig_idx_mapping.append(orig_indices)

        if not all_sentences:
            for r in results:
                r.supporting_sentence_indices = []
                r.supporting_sentences = []
                r.sentence_scores = []
            return

        # Cross-encoder scoring: (query, sentence) pairs scored jointly
        pairs = [(query, sent) for sent in all_sentences]
        log.step(f"Cross-encoder scoring {len(pairs)} sentences...")
        ce_scores: np.ndarray = self._cross_encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Normalize cross-encoder scores to [0, 1] range using sigmoid
        # Raw CE scores can be arbitrary floats; sigmoid maps them to probabilities
        ce_scores = 1.0 / (1.0 + np.exp(-np.array(ce_scores, dtype=np.float64)))

        query_words = self._normalize_tokens(query)
        for r_idx, (r, (start, end)) in enumerate(zip(results, passage_sentence_ranges)):
            if end <= start:
                continue

            sents = all_sentences[start:end]
            scores = ce_scores[start:end].tolist()
            orig_indices = sentence_to_orig_idx_mapping[r_idx]

            # Title-match boost: if passage title words overlap with query, boost all sentence scores
            title_words = self._normalize_tokens(r.passage.title)
            if title_words & query_words:
                scores = [s + self.title_overlap_boost for s in scores]

            paired = sorted(zip(scores, sents, orig_indices), reverse=True)
            
            selected_sents = []
            selected_scores = []
            selected_indices = []

            # Rank-aware guaranteed minimum: top passages get more sentences
            if r.rank == 0:
                min_guarantee = 2  # Rank 0 (best passage): guarantee at least 2 sentences
            elif r.rank == 1:
                min_guarantee = 2  # Rank 1: guarantee at least 2 sentences
            else:
                min_guarantee = 1  # Lower passages: guarantee 1

            for i, (score, sent, orig_idx) in enumerate(paired):
                if i < min_guarantee:
                    selected_sents.append(sent)
                    selected_scores.append(round(float(score), 4))
                    selected_indices.append(orig_idx)
                elif score >= self.sentence_score_threshold and len(selected_sents) < self.max_sentences_per_passage:
                    selected_sents.append(sent)
                    selected_scores.append(round(float(score), 4))
                    selected_indices.append(orig_idx)
                else:
                    break  # Sorted descending — all remaining are below threshold

            # Optionally, sort the selected sentences chronologically (by orig_idx). This makes the prompt read more naturally.
            chronological = sorted(zip(selected_indices, selected_sents, selected_scores))
             
            if chronological:
                r.supporting_sentence_indices, r.supporting_sentences, r.sentence_scores = zip(*chronological)
                r.supporting_sentence_indices = list(r.supporting_sentence_indices)
                r.supporting_sentences = list(r.supporting_sentences)
                r.sentence_scores = list(r.sentence_scores)
            else:
                r.supporting_sentence_indices = []
                r.supporting_sentences = []
                r.sentence_scores = []
        
        # Clear sentence fields for the lower-ranked passages we skipped
        for r in results[self.sentence_passage_limit:]:
            r.supporting_sentence_indices = []
            r.supporting_sentences = []
            r.sentence_scores = []

    def __repr__(self) -> str:
        return (
            f"Reranker(model={self.model_name}, "
            f"threshold={self.sentence_score_threshold}, "
            f"max_sents={self.max_sentences_per_passage}, "
            f"device={self.device})"
        )


