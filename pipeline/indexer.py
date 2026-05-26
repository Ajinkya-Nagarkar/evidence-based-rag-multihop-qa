"""
Hybrid Multi-Hop Retriever: BM25 + FAISS
The hybrid approach covers both cases — BM25 catches exact term matches that dense might miss, 
and dense catches semantic similarity that BM25 can't understand.
"""

import os
# Prevent FAISS segfaults on macOS must be set before importing FAISS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import pickle
import re
import requests
import numpy as np
from pathlib import Path
import faiss
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process as fz_process
from pipeline.embedder import OllamaEmbedder
from pipeline.data_loader import Passage
from scripts.config import load_config, get_best_device
from scripts.logger import get_logger
log = get_logger("indexer")


@dataclass
class RetrievalResult:
    """
    Eg:
        passage: Passage(title="Scott Derrickson", sentences=[...])
        score: 0.87
        rank: 0
        bm25_score: 0.45
        dense_score: 0.92
        confidence: 0.73
    """
    passage: Passage     # from data_loader
    score: float         # final combined score
    rank: int            # rank of the passage (0-indexed)
    bm25_score: float    # raw BM25 score (before normalization)
    dense_score: float   # raw cosine similarity from FAISS (-1 to 1)
    confidence: float = 0.0  # how confident we are (score gap from #1 to #2)
    hop: int = 0         # which hop retrieved this passage (0 = first hop)
    # Example: score=0.929 (bm25=17.59, dense=0.541) --> high overlap + high semantic similarity


def _tokenize(text: str) -> List[str]:
    """Lowercase and split texts by whitespaces for BM25"""
    return text.lower().split()

def _rrf_score(rank: int, k: int = 60) -> float:
    """
    BM25 scores might be 0 to 25 while dense scores are -1 to 1, not comparable. RRF depends only on rank
    Reciprocal Rank Fusion score for a single rank position
    If a document is rank 0 then RRF score is 1/61, rank 1 then 1/62, rank 2 then 1/63, and so on

    Low Rank --> high RRF score

    score = 1 / (k + rank + 1)
    """
    return 1.0 / (k + rank + 1)


def _compute_confidence(scores: List[float]) -> float:
    """
    Confidence signal: how much better is top result #1 than #2?
    Returns 0-1: high = top result clearly dominates, low = top results are close (ambiguous)
    """
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    gap = (scores[0] - scores[1]) / scores[0]
    return min(max(gap, 0.0), 1.0)


class BM25Retriever:
    """
    Sparse retriever using BM25 (Okapi)
    Scores passages by term-frequency overlap with the query

    How many times does the query appear in the passage? - TF
    How important is the query term in the corpus? -  IDF
    How long is the passage? - Longer passage get penalized
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def index(self, texts: List[str]):
        """
        Build BM25 index from passage texts, stores token frequency and IDF values
        texts: list of passage strings (title + text)
        """
        self._corpus_tokens = [_tokenize(t) for t in texts] # tokenize passage
        self._bm25 = BM25Okapi(self._corpus_tokens) # build BM25 inverted index

    # def score(self, query: str) -> np.ndarray:
    #     """
    #     Score all passages for a query
    #     Returns array of shape (n_passages,), higher = more relevant
    #     WARNING: scores entire corpus — use top_k() for FullWiki scale
    #     """
    #     if self._bm25 is None:
    #         raise RuntimeError("Call .index() first")
    #     return self._bm25.get_scores(_tokenize(query)) # get scores for all passages

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k passage indices
        Returns scores without fully ranking the corpus
        Returns: list of (passage_index, bm25_score) sorted descending
        """
        if self._bm25 is None:
            raise RuntimeError("Call .index() first")
        scores = self._bm25.get_scores(_tokenize(query))
        
        # partial sort:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save(self, path: Path):
        """Save BM25 corpus tokens to disk"""
        with open(path / "bm25_tokens.pkl", "wb") as f:
            pickle.dump(self._corpus_tokens, f)

    def load(self, path: Path):
        """Load BM25 corpus tokens from disk and rebuild index"""
        with open(path / "bm25_tokens.pkl", "rb") as f:
            self._corpus_tokens = pickle.load(f)
        self._bm25 = BM25Okapi(self._corpus_tokens)

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        ollama_base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        # legacy param kept so from_config() call sites don't break
        device: str = "auto",
    ):
        if device == "auto":
            device = get_best_device()
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.encoder = OllamaEmbedder(
            model=model_name,
            base_url=ollama_base_url,
            batch_size=batch_size,
        )
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: int = self.encoder.dim

    def index(self, texts: List[str], show_progress: bool = True):
        vectors = self.encoder.encode(texts, show_progress=show_progress)
        self._dim = vectors.shape[1]

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    def score(self, query: str) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode_query(query).reshape(1, -1)
        n = self._index.ntotal
        distances, indices = self._index.search(q_vec, n)

        scores = np.zeros(n, dtype=np.float32)
        scores[indices[0]] = distances[0]
        return scores

    def search_top_k(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Call .index() first")

        q_vec = self.encoder.encode_query(query).reshape(1, -1)
        distances, indices = self._index.search(q_vec, top_k)
        return distances[0], indices[0]

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode_query(query)

    def save(self, path: Path):
        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "dense_meta.json", "w") as f:
            json.dump({"dim": self._dim, "model_name": self.model_name}, f)

    def load(self, path: Path):
        self._index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "dense_meta.json", "r") as f:
            meta = json.load(f)
        self._dim = meta["dim"]

    @property
    def is_indexed(self) -> bool:
        return self._index is not None


class HybridRetriever:
    def __init__(
        self,
        embed_model: str = "qwen3-embedding:8b",
        ollama_base_url: str = "http://localhost:11434",
        extraction_model: str = "qwen3:8b",
        alpha: float = 0.7,
        alpha_bridge: Optional[float] = None,
        alpha_comparison: Optional[float] = None,
        rrf_k: int = 20,
        candidate_pool_size: int = 200,
        device: str = "auto",   # kept for API compat, unused (Ollama handles device)
        batch_size: int = 32,
        prompts=None,
        max_bridge_entities: int = 5,
        extraction_temperature: float = 0.0,
        extraction_timeout: int = 1800,
        confidence_threshold: float = 1.0,
        fuzzy_title_threshold: int = 70,
    ):
        self.extraction_model = extraction_model
        self.ollama_base_url = ollama_base_url
        self.alpha = alpha
        self.alpha_bridge = alpha_bridge
        self.alpha_comparison = alpha_comparison
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.prompts = prompts
        self.max_bridge_entities = max_bridge_entities
        self.extraction_temperature = extraction_temperature
        self.extraction_timeout = extraction_timeout
        self._confidence_threshold = confidence_threshold
        self._fuzzy_title_threshold = fuzzy_title_threshold

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(
            model_name=embed_model,
            ollama_base_url=ollama_base_url,
            batch_size=batch_size,
        )
        self._passages: List[Passage] = []
        self._texts: List[str] = []
        self._title_list: List[str] = []
        self._title_to_idx: dict = {}
        self._session = requests.Session()  # Connection pooling for LLM calls

    @classmethod
    def from_config(cls, cfg=None) -> "HybridRetriever":
        """
        Config Retriever
        Eg: retriever = HybridRetriever.from_config("configs/fast.yaml") # custom config
        """
        if cfg is None or isinstance(cfg, (str, Path)):
            cfg = load_config(cfg)
        r = cfg.retriever
        return cls(
            embed_model=r.embed_model,
            ollama_base_url=r.ollama_base_url,
            extraction_model=getattr(r.multihop, "extraction_model", "qwen3:8b"),
            alpha=r.alpha,
            alpha_bridge=r.alpha_bridge,
            alpha_comparison=r.alpha_comparison,
            rrf_k=r.rrf_k,
            candidate_pool_size=r.candidate_pool_size,
            device=r.device,
            batch_size=r.batch_size,
            prompts=getattr(cfg, "prompts", None),
            max_bridge_entities=r.multihop.max_bridge_entities,
            extraction_temperature=getattr(r.multihop, "extraction_temperature", 0.0),
            extraction_timeout=getattr(r.multihop, "extraction_timeout", 1800),
            confidence_threshold=getattr(r.multihop, "llm_decompose_confidence_threshold", 1.0),
            fuzzy_title_threshold=getattr(r.multihop, "fuzzy_title_threshold", 70),
        )

    def _get_alpha(self, question_type: Optional[str] = None) -> float:
        if question_type == "bridge" and self.alpha_bridge is not None:
            return self.alpha_bridge
        if question_type == "comparison" and self.alpha_comparison is not None:
            return self.alpha_comparison
        return self.alpha

    def index(self, passages: Union[List[Passage], List[str]], show_progress: bool = True):
        """
        Build both BM25 and FAISS indices

        passages: either list of Passage objects (from data_loader)
                  or list of plain strings
        """
        if len(passages) == 0:
            raise ValueError("No passages to index")

        # Accept either Passage objects or raw strings
        if isinstance(passages[0], Passage):
            self._passages = list(passages)
            self._texts = [p.title_text for p in self._passages]
        else:
            self._passages = [Passage(title="", sentences=[str(p)]) for p in passages]
            self._texts = [str(p) for p in passages]

        log.info(f"\nIndexing {len(self._texts)} passages...")

        # Build both indices
        log.step("Building BM25 index...")
        self.bm25.index(self._texts)

        log.step("Building FAISS dense index...")
        self.dense.index(self._texts, show_progress=show_progress)

        self._build_title_index()

        log.success(f"{len(self._texts)} passages indexed.")

    def save(self, save_dir: Union[str, Path]):
        """
        Save both indices + passage data
        call .load() instead of re-encoding
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self.dense.save(save_dir)

        # Save BM25 tokens
        self.bm25.save(save_dir)

        # Save passage data
        passages_data = []
        for p in self._passages:
            passages_data.append({
                "title": p.title,
                "sentences": p.sentences,
                "passage_id": p.passage_id,
            })
        with open(save_dir / "passages.json", "w", encoding="utf-8") as f:
            json.dump(passages_data, f, ensure_ascii=False)

        # Save config
        config = {
            "alpha": self.alpha,
            "alpha_bridge": self.alpha_bridge,
            "alpha_comparison": self.alpha_comparison,
            "rrf_k": self.rrf_k,
            "num_passages": len(self._passages),
            "candidate_pool_size": self.candidate_pool_size,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        log.success(f"Saved index to {save_dir}/ ({len(self._passages)} passages)")

    def load(self, save_dir: Union[str, Path]):
        """
        Load indices + passages from disk (skips encoding)
        The SentenceTransformer model is still loaded for query encoding
        """
        save_dir = Path(save_dir)
        if not (save_dir / "faiss.index").exists():
            raise FileNotFoundError(f"No saved index in {save_dir}")

        log.info(f"Loading index from {save_dir}/...")

        # Load FAISS
        self.dense.load(save_dir)

        # Load BM25
        self.bm25.load(save_dir)

        # Load passages
        with open(save_dir / "passages.json", "r", encoding="utf-8") as f:
            passages_data = json.load(f)
        self._passages = [
            Passage(title=p["title"], sentences=p["sentences"], passage_id=p.get("passage_id"))
            for p in passages_data
        ]
        self._texts = [p.title_text for p in self._passages]

        # Load config
        with open(save_dir / "config.json", "r") as f:
            config = json.load(f)
        self.alpha = config.get("alpha", self.alpha)
        self.alpha_bridge = config.get("alpha_bridge", self.alpha_bridge)
        self.alpha_comparison = config.get("alpha_comparison", self.alpha_comparison)
        self.rrf_k = config.get("rrf_k", self.rrf_k)

        self._build_title_index()

        log.success(f"Loaded index from {save_dir}/ ({len(self._passages)} passages)")

    def retrieve(self, query: str, top_k: int = 10,
                 question_type: Optional[str] = None,
                 candidate_pool_size: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve top-k passages using candidate-pool RRF scoring

          1. Get top-N candidates from BM25 (fast, uses argpartition)
          2. Get top-N candidates from FAISS (fast, native top-k)
          3. Union the candidate pools (~2N passages at most)
          4. RRF rank only within this small pool

        query: the question string
        top_k: number of passages to return
        question_type: "bridge" or "comparison" (uses per-type alpha if set)
        candidate_pool_size: how many candidates to pull from each retriever before fusing

        Returns: sorted list of RetrievalResult (highest score first)
        """
        if not self.bm25.is_indexed or not self.dense.is_indexed:
            raise RuntimeError("Call .index() first")

        alpha = self._get_alpha(question_type)
        pool_size = min(candidate_pool_size or self.candidate_pool_size, len(self._passages))

        # Get top-N candidates from each retriever (fast)
        bm25_top = self.bm25.top_k(query, pool_size)          # [(idx, score), ...]
        dense_scores_arr, dense_indices_arr = self.dense.search_top_k(query, pool_size)  # (scores, indices)

        # Build candidate pool (union of both top-N sets)
        candidate_indices = set()
        bm25_score_map = {}   # idx -> raw bm25 score
        dense_score_map = {}  # idx -> raw dense score

        for rank, (idx, score) in enumerate(bm25_top):
            candidate_indices.add(idx)
            bm25_score_map[idx] = (rank, score)   # (rank_in_bm25, raw_score)

        for rank, (idx, score) in enumerate(zip(dense_indices_arr.tolist(), dense_scores_arr.tolist())):
            candidate_indices.add(idx)
            dense_score_map[idx] = (rank, score)

        # RRF score only the candidates
        scored = []
        for idx in candidate_indices:
            # If a passage was found by one retriever but not other, assign it a worst-case rank (pool_size) for the missing retriever
            bm25_rank = bm25_score_map[idx][0] if idx in bm25_score_map else pool_size
            dense_rank = dense_score_map[idx][0] if idx in dense_score_map else pool_size
            bm25_raw = bm25_score_map[idx][1] if idx in bm25_score_map else 0.0
            dense_raw = dense_score_map[idx][1] if idx in dense_score_map else 0.0

            rrf = alpha * _rrf_score(dense_rank, self.rrf_k) + \
                  (1 - alpha) * _rrf_score(bm25_rank, self.rrf_k)

            scored.append((idx, rrf, bm25_raw, dense_raw))

        # Sort by RRF score and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # Compute confidence: how much #1 dominates over #2
        all_scores = [s[1] for s in scored]
        conf = _compute_confidence(all_scores)

        results = []
        for rank, (idx, rrf, bm25_raw, dense_raw) in enumerate(scored):
            results.append(RetrievalResult(
                passage=self._passages[idx],
                score=float(rrf),
                rank=rank,
                bm25_score=float(bm25_raw),
                dense_score=float(dense_raw),
                confidence=conf if rank == 0 else 0.0,  # confidence only on top result
                hop=0,
            ))
        return results

    def _build_title_index(self):
        """Build a mapping from passage titles to passage indices for fuzzy matching."""
        self._title_to_idx = {}
        for i, p in enumerate(self._passages):
            title = p.title.strip()
            if title:
                self._title_to_idx.setdefault(title, []).append(i)
        self._title_list = list(self._title_to_idx.keys())
        log.info(f"Built title index: {len(self._title_list)} unique titles")

    def _retrieve_by_title_fuzzy(self, entities: List[str]) -> List[RetrievalResult]:
        """Retrieve passages whose titles fuzzy-match the given entity strings."""
        if not self._title_list or not entities:
            return []
        results = []
        seen_titles = set()
        for entity in entities:
            matches = fz_process.extract(
                entity, self._title_list,
                scorer=fuzz.token_set_ratio,
                limit=3,
                score_cutoff=self._fuzzy_title_threshold,
            )
            for matched_title, score, _ in matches:
                if matched_title in seen_titles:
                    continue
                seen_titles.add(matched_title)
                for idx in self._title_to_idx[matched_title]:
                    results.append(RetrievalResult(
                        passage=self._passages[idx],
                        score=score / 100.0,
                        rank=0,
                        bm25_score=0.0,
                        dense_score=0.0,
                        confidence=0.0,
                        hop=1,
                    ))
        log.info(f"Fuzzy title matching: {len(entities)} entities → {len(results)} passages")
        return results

    def _extract_bridge_entities(self, passage: Passage, question: str) -> List[str]:
        """
        Extract potential bridge entities from a passage that could be used for hop 2

        Catches:
          - Passage title (primary source for entity names)
          - Multi-word names:   "Shirley Temple", "New York City"
          - Acronyms:           "NASA", "FBI", "UNESCO"
          - Single-name entities: "Cher", "Madonna", "Beyoncé"
          - Titles with punctuation: "Dr. Strange", "St. Louis"
          - Parenthetical context: "(1928-1992)", "(actress)"
        """
        question_lower = question.lower()
        text = passage.text

        entities = []

        # Passage title is primary bridge source (often contains entity names like "Shirley Temple")
        if passage.title and passage.title.strip():
            entities.append(passage.title)

        # Multi-word capitalized phrases ("Shirley Temple", "New York City", "Dr. Strange", "St. Louis")
        # Improved pattern: catches 2+ consecutive capitalized words
        entities += re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)

        # Acronyms — 2+ consecutive uppercase letters ("NASA", "FBI", "MCU")
        entities += re.findall(r'\b([A-Z]{2,})\b', text)

        # Single capitalized words that appear mid-sentence, exclude sentence starters by requiring a lowercase letter or punctuation before
        entities += re.findall(r'(?<=[a-z.,;:!?]\s)([A-Z][a-zà-ÿ]{2,})\b', text)

        # Parenthetical context (dates, descriptions) - "Shirley Temple (1928-1992)"
        # Extract text in parentheses that might be useful context
        entities += re.findall(r'\(([^)]+)\)', text)[:3]  # limit to 3 parenthetical contexts

        # Filter out entities already in the question
        bridge = []
        seen = set()
        for entity in entities:
            entity_clean = entity.strip().rstrip('.')
            if (entity_clean.lower() not in question_lower
                    and entity_clean not in seen
                    and len(entity_clean) > 1
                    and entity_clean != passage.title):  # don't duplicate title
                seen.add(entity_clean)
                bridge.append(entity_clean)

        # Always include passage title as fallback if not already added
        if passage.title and passage.title.strip() and passage.title not in bridge:
            bridge.insert(0, passage.title)

        return bridge[:self.max_bridge_entities]

    def _extract_entities_llm(self, passages: List[Passage], question: str) -> List[str]:
        if not passages:
            return []
        
        system_prompt = self.prompts.indexer_system.strip() if self.prompts and self.prompts.indexer_system else (
            "You are a precise entity extractor. Respond ONLY with a valid JSON array of strings, with no additional text and no markdown formatting (do not use ```json)."
        )
        
        context = ""
        for i, p in enumerate(passages):
            # Truncate text to 800 chars to save prompt context and speed up extraction
            context += f"Passage {i+1} Title: {p.title}\nPassage {i+1} Text: {p.text[:800]}\n\n"
            
        if self.prompts and self.prompts.indexer_user:
            prompt = self.prompts.indexer_user.format(question=question, context=context)
        else:
            prompt = (
                f"Question: {question}\n\n"
                f"Context:\n{context}\n"
                f"Extract the most important named entities (people, places, organizations, specific concepts) from the context that are relevant to finding the final answer to the question. "
                f"Focus particularly on entities that link different pieces of information together (bridge entities).\n"
                f"DO NOT include entities that are already explicitly mentioned in the question.\n"
                f"Output strictly a JSON list of strings, for example: [\"Entity 1\", \"Entity 2\"]. Output nothing else."
            )
        
        try:
            # Try /api/chat first (required for chat-only models like gemma4),
            # fall back to /api/generate for older models.
            chat_url = f"{self.ollama_base_url}/api/chat"
            gen_url = f"{self.ollama_base_url}/api/generate"
            chat_payload = {
                "model": self.extraction_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": self.extraction_temperature, "num_ctx": 4096},
                "keep_alive": -1,
            }
            gen_payload = {
                "model": self.extraction_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "temperature": self.extraction_temperature,
                "options": {"num_ctx": 4096},
                "keep_alive": -1,
            }
            resp = self._session.post(chat_url, json=chat_payload, timeout=self.extraction_timeout)
            if resp.status_code == 404:
                # Model doesn't support /api/chat, fall back to /api/generate
                resp = self._session.post(gen_url, json=gen_payload, timeout=self.extraction_timeout)
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
            else:
                resp.raise_for_status()
                text = resp.json().get("message", {}).get("content", "").strip()
            
            entities = []
            try:
                # Try parsing the whole text first
                entities = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r'\[.*?\]', text, re.DOTALL)
                if match:
                    try:
                        entities = json.loads(match.group(0))
                    except:
                        pass
                
            if isinstance(entities, list):
                # Also collect passage titles as robust fallbacks
                titles = [p.title for p in passages if p.title]
                entities.extend(titles)
                # Deduplicate and filter out question terms
                q_lower = question.lower()
                final_entities = []
                seen = set()
                for e in entities:
                    e_clean = str(e).strip()
                    if e_clean and e_clean.lower() not in q_lower and e_clean.lower() not in seen:
                        seen.add(e_clean.lower())
                        final_entities.append(e_clean)
                return final_entities
        except Exception as e:
            log.warning(f"LLM entity extraction failed: {e}. Falling back to regex.")
            
        # Fallback to regex
        fallback = []
        for p in passages:
            fallback.extend(self._extract_bridge_entities(p, question))
        return list(dict.fromkeys(fallback))

    def retrieve_multihop(self, query: str, hops: int = 2, top_k: int = 10,
                          top_k_per_hop: int = 20,
                          question_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Multi-hop iterative retrieval with fuzzy title matching.

        Hop 1: hybrid retrieve using the original question.
        Hop 2: extract entities from hop-1 passages, then:
               (a) fuzzy-match entity strings against passage titles
               (b) hybrid retrieve with reformulated query (question + entities)
               Merge fuzzy + hybrid results for hop-2.
        Finally: deduplicate and re-rank all results.

        Both bridge AND comparison questions run hop-2 (comparison entities
        benefit from fuzzy title matching too).
        """
        all_results: List[RetrievalResult] = []
        seen_keys: set = set()

        # ── Hop 1 ──────────────────────────────────────────────────────
        hop1_results = self.retrieve(query, top_k=top_k_per_hop,
                                     question_type=question_type)
        hop1_scores = [r.score for r in hop1_results]
        hop1_confidence = _compute_confidence(hop1_scores)

        top_passages_for_extraction = []
        for r in hop1_results:
            key = (r.passage.title, r.passage.text)
            if key not in seen_keys:
                seen_keys.add(key)
                r.hop = 0
                all_results.append(r)
            top_passages_for_extraction.append(r.passage)

        # ── Hop 2 (always runs — confidence_threshold defaults to 1.0) ─
        confidence_threshold = getattr(self, '_confidence_threshold', 1.0)
        if hop1_confidence > confidence_threshold:
            log.info(f"Hop-1 confidence {hop1_confidence:.3f} > threshold {confidence_threshold} — skipping hop-2")
        else:
            # Extract bridge entities via LLM (falls back to regex)
            bridge_entities = self._extract_entities_llm(
                top_passages_for_extraction[:5], query
            )
            log.info(f"Hop-1 extracted entities: {bridge_entities}")

            if bridge_entities:
                # (a) Fuzzy title matching on extracted entities
                fuzzy_results = self._retrieve_by_title_fuzzy(bridge_entities)
                for r in fuzzy_results:
                    key = (r.passage.title, r.passage.text)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append(r)

                # (b) Hybrid retrieve with reformulated query
                unique_entities = bridge_entities[:self.max_bridge_entities]
                entity_str = " ".join(unique_entities)
                hop2_query = f"{query} {entity_str}"
                log.info(f"Hop-2 query: {hop2_query}")

                hop2_results = self.retrieve(hop2_query, top_k=top_k_per_hop,
                                             question_type=question_type)
                for r in hop2_results:
                    key = (r.passage.title, r.passage.text)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        r.hop = 1
                        all_results.append(r)

        # ── Merge & re-rank ────────────────────────────────────────────
        all_results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(all_results):
            r.rank = i

        return all_results[:top_k]

    def retrieve_passages(self, query: str, top_k: int = 10,
                          question_type: Optional[str] = None) -> List[Passage]:
        """return just the Passage objects"""
        return [r.passage for r in self.retrieve(query, top_k, question_type)]

    def retrieve_texts(self, query: str, top_k: int = 10,
                       question_type: Optional[str] = None) -> List[str]:
        """return just the passage text strings"""
        return [r.passage.title_text for r in self.retrieve(query, top_k, question_type)]

    @property
    def num_passages(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return (f"HybridRetriever(alpha={self.alpha}, "
                f"alpha_bridge={self.alpha_bridge}, "
                f"alpha_comparison={self.alpha_comparison}, "
                f"rrf_k={self.rrf_k}, "
                f"indexed={self.bm25.is_indexed}, "
                f"passages={len(self._passages)})")


