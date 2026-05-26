from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pipeline.reranker import RerankResult
from pipeline.verifier import VerificationResult
from scripts.logger import get_logger

log = get_logger("decider")


@dataclass
class DecisionOutput:
    answer: str
    supporting_passage_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verifier_supported: bool = False
    support_score: float = 0.0


class Decider:
    """Post-verification decision layer.
    
    Responsibilities:
      1. Optionally retry generation when the verifier rejects an answer
      2. Compute a confidence score combining verification + reranker signals
      3. Deduplicate supporting facts
      4. Extract supporting passage identifiers
      5. Decide whether to keep the answer or abstain (configurable)
    
    By default, abstention is DISABLED — the Decider keeps the original answer
    even when unsupported, since abstaining always scores 0 EM. Confidence is
    lowered instead to signal uncertainty.
    """

    def __init__(self, abstain_on_unsupported: bool = False):
        """
        abstain_on_unsupported: If True, replace unsupported answers with
            "Insufficient evidence". If False (default), keep the original
            answer but lower confidence. Set to False for evaluation (maximizes
            EM), set to True for production (avoids hallucinated answers).
        """
        self.abstain_on_unsupported = abstain_on_unsupported

    def decide(
        self,
        answer: str,
        verification: Optional[VerificationResult],
        reranked_results: Sequence[RerankResult],
        supporting_facts: Optional[Sequence[Sequence[Any]]] = None,
        verifier: Optional["Verifier"] = None,
        retry_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        attempt_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DecisionOutput, Dict[str, Any]]:
        attempt: Dict[str, Any] = dict(attempt_metadata or {})
        attempt.update(
            {
                "answer": answer,
                "supporting_facts": supporting_facts,
                "reranked_results": reranked_results,
                "full_response": attempt.get("full_response"),
            }
        )

        current_verification = verification
        is_supported = bool(getattr(current_verification, "is_supported", False))

        # --- Retry logic (only when enabled and verification failed) ---
        if current_verification is not None and not is_supported and verifier and retry_fn:
            log.info(f"Decider: answer unsupported, attempting retry...")
            retry = retry_fn()
            if retry:
                attempt.update(retry)
                try:
                    current_verification = verifier.verify(
                        answer=attempt["answer"],
                        evidence=attempt["reranked_results"],
                        supporting_facts=attempt["supporting_facts"],
                    )
                    is_supported = bool(getattr(current_verification, "is_supported", False))
                    if is_supported:
                        log.info(f"Decider: retry succeeded, answer now supported")
                    else:
                        log.info(f"Decider: retry did not improve support score")
                except Exception as e:
                    log.warning(f"Decider: retry verification failed: {e}")

        # --- Deduplicate supporting facts ---
        deduped_facts = self._deduplicate_facts(attempt.get("supporting_facts"))
        attempt["supporting_facts"] = deduped_facts

        # --- Extract passage IDs ---
        supporting_ids = self._extract_supporting_passage_ids(
            attempt["reranked_results"], deduped_facts
        )

        # --- Compute confidence score ---
        support_score = float(getattr(current_verification, "support_score", 0.0) or 0.0)
        reranker_confidence = self._compute_reranker_confidence(reranked_results)
        
        # Weighted combination: verification is primary, reranker is secondary signal
        if current_verification is not None:
            confidence = 0.7 * support_score + 0.3 * reranker_confidence
        else:
            confidence = reranker_confidence  # No verifier — use reranker only

        # --- Final answer decision ---
        final_answer = (attempt["answer"] or "").strip()
        
        if current_verification is not None and not is_supported:
            if self.abstain_on_unsupported:
                final_answer = "Insufficient evidence"
                log.info(f"Decider: abstaining (answer unsupported, abstain mode ON)")
            else:
                # Keep the answer but penalize confidence
                confidence *= 0.5
                log.debug(f"Decider: keeping unsupported answer with reduced confidence {confidence:.3f}")

        return (
            DecisionOutput(
                answer=final_answer,
                supporting_passage_ids=supporting_ids,
                confidence=round(confidence, 4),
                verifier_supported=is_supported,
                support_score=round(support_score, 4),
            ),
            attempt,
        )

    def _deduplicate_facts(
        self, supporting_facts: Optional[Sequence[Sequence[Any]]]
    ) -> List[List[Any]]:
        """Deduplicate [title, sent_idx] pairs while preserving order."""
        if not supporting_facts:
            return []
        
        seen = set()
        deduped = []
        for fact in supporting_facts:
            if not isinstance(fact, (list, tuple)) or len(fact) < 2:
                continue
            key = (fact[0], int(fact[1]))
            if key not in seen:
                seen.add(key)
                deduped.append([fact[0], int(fact[1])])
        return deduped

    def _compute_reranker_confidence(
        self, reranked_results: Sequence[RerankResult]
    ) -> float:
        """Compute a confidence signal from reranker scores.
        
        High confidence = top passages have high, concentrated scores.
        Low confidence = all passages have similar/low scores.
        """
        if not reranked_results:
            return 0.0
        
        scores = [r.rerank_score for r in reranked_results if hasattr(r, 'rerank_score')]
        if not scores:
            scores = [r.score for r in reranked_results]
        if not scores:
            return 0.0
        
        top_score = max(scores)
        # Normalize using sigmoid-like scaling (scores can be arbitrary)
        import math
        confidence = 1.0 / (1.0 + math.exp(-top_score))
        return min(confidence, 1.0)

    def _extract_supporting_passage_ids(
        self,
        reranked_results: Sequence[RerankResult],
        supporting_facts: Optional[Sequence[Sequence[Any]]],
    ) -> List[str]:
        if not reranked_results:
            return []

        if not supporting_facts:
            return []

        ids: List[str] = []
        for fact in supporting_facts:
            if not isinstance(fact, (list, tuple)) or len(fact) < 2:
                continue
            title = fact[0]
            try:
                sent_idx = int(fact[1])
            except Exception:
                continue

            for r in reranked_results:
                if r.passage.title != title:
                    continue
                if sent_idx in (r.supporting_sentence_indices or []):
                    ids.append(self._passage_identifier(r))
                    break
                if 0 <= sent_idx < len(r.passage.sentences):
                    ids.append(self._passage_identifier(r))
                    break

        # De-duplicate while preserving order
        seen = set()
        ordered = []
        for pid in ids:
            if pid and pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered

    def _passage_identifier(self, result: RerankResult) -> str:
        passage = result.passage
        if getattr(passage, "passage_id", None):
            return str(passage.passage_id)
        if passage.title:
            return str(passage.title)
        return f"passage_{result.rank}"

