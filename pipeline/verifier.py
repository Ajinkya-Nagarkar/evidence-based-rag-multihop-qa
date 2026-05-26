import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("verifier")


@dataclass
class ClaimCheck:
    claim: str
    support_score: float
    is_supported: bool
    best_evidence_id: Optional[str] = None
    best_evidence_text: str = ""


@dataclass
class VerificationResult:
    support_score: float
    is_supported: bool
    unsupported_claims: List[str] = field(default_factory=list)
    claim_checks: List[ClaimCheck] = field(default_factory=list)
    evidence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Verifier:
    """
    Verification modes:
      - "overlap": lexical support scoring
      - "nli": entailment model scoring with lexical fallback if unavailable
      - "qa": question-answering confidence with lexical fallback
    """

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with", "who", "what", "when", "where", "which",
        "whom", "whose", "why", "how", "this", "these", "those", "their", "there",
        "than", "then", "into", "about", "after", "before", "during", "while",
    }

    def __init__(
        self,
        mode: str = "overlap",
        support_threshold: float = 0.55,
        claim_threshold: float = 0.45,
        min_supported_claim_ratio: float = 1.0,
        max_claims: int = 6,
        nli_model_name: str = "roberta-large-mnli",
        nli_device: int = -1,
        qa_model_name: str = "distilbert-base-cased-distilled-squad",
        qa_device: int = -1,
        qa_min_answer_score: float = 0.2,
    ):
        self.mode = mode
        self.support_threshold = support_threshold
        self.claim_threshold = claim_threshold
        self.min_supported_claim_ratio = min_supported_claim_ratio
        self.max_claims = max_claims
        self.nli_model_name = nli_model_name
        self.nli_device = nli_device
        self.qa_model_name = qa_model_name
        self.qa_device = qa_device
        self.qa_min_answer_score = qa_min_answer_score
        self._nli_pipe = None
        self._qa_pipe = None
        self._qa_loader_failed = False

        log.success(
            f"Verifier ready. mode={mode}, support_threshold={support_threshold}, "
            f"claim_threshold={claim_threshold}"
        )

    @classmethod
    def from_config(cls, cfg=None) -> "Verifier":
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)

        vcfg = getattr(cfg, "verifier", None)
        if vcfg is None:
            return cls()

        return cls(
            mode=getattr(vcfg, "mode", getattr(vcfg, "backend", "overlap")),
            support_threshold=float(getattr(vcfg, "support_threshold", 0.55)),
            claim_threshold=float(getattr(vcfg, "claim_threshold", 0.45)),
            min_supported_claim_ratio=float(getattr(vcfg, "min_supported_claim_ratio", 1.0)),
            max_claims=int(getattr(vcfg, "max_claims", 6)),
            nli_model_name=getattr(vcfg, "nli_model_name", "roberta-large-mnli"),
            nli_device=int(getattr(vcfg, "nli_device", -1)),
            qa_model_name=getattr(vcfg, "qa_model_name", "distilbert-base-cased-distilled-squad"),
            qa_device=int(getattr(vcfg, "qa_device", -1)),
            qa_min_answer_score=float(getattr(vcfg, "qa_min_answer_score", 0.2)),
        )

    def verify(
        self,
        answer: str,
        evidence: Sequence[Any],
        supporting_facts: Optional[Sequence[Sequence[Any]]] = None,
    ) -> VerificationResult:
        answer = (answer or "").strip()
        evidence_items = self._flatten_evidence(evidence)

        if not answer:
            return VerificationResult(
                support_score=0.0,
                is_supported=False,
                unsupported_claims=[""],
                claim_checks=[],
                evidence_count=len(evidence_items),
                metadata={"reason": "empty_answer"},
            )

        if not evidence_items:
            return VerificationResult(
                support_score=0.0,
                is_supported=False,
                unsupported_claims=[answer],
                claim_checks=[],
                evidence_count=0,
                metadata={"reason": "no_evidence"},
            )

        # Yes/no answers have no meaningful lexical content to overlap with
        # evidence, so we check whether the generator had supporting facts
        # selected and whether those facts have substance.
        if answer.lower() in ("yes", "no", "noanswer"):
            claim_checks = self._score_yesno(answer, evidence_items, supporting_facts)
        else:
            claims = self._extract_claims(answer)
            claim_checks = [self._score_claim(c, evidence_items) for c in claims]

            if not claim_checks:
                claim_checks = [self._score_claim(answer, evidence_items)]

        support_score = sum(c.support_score for c in claim_checks) / max(len(claim_checks), 1)
        supported_claims = sum(1 for c in claim_checks if c.is_supported)
        support_ratio = supported_claims / max(len(claim_checks), 1)
        unsupported_claims = [c.claim for c in claim_checks if not c.is_supported]
        is_supported = (
            support_score >= self.support_threshold
            and support_ratio >= self.min_supported_claim_ratio
        )

        metadata = {
            "mode": self.mode,
            "num_claims": len(claim_checks),
            "supported_claim_ratio": round(support_ratio, 4),
            "supporting_fact_coverage": self._supporting_fact_coverage(
                supporting_facts, evidence_items
            ),
        }

        return VerificationResult(
            support_score=round(float(support_score), 4),
            is_supported=bool(is_supported),
            unsupported_claims=unsupported_claims,
            claim_checks=claim_checks,
            evidence_count=len(evidence_items),
            metadata=metadata,
        )

    def _extract_claims(self, answer: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", answer).strip()
        if not cleaned:
            return []

        parts = re.split(r"(?<=[\.\?!;])\s+", cleaned)
        claims = []
        for part in parts:
            piece = part.strip(" -")
            if piece:
                claims.append(piece)
            if len(claims) >= self.max_claims:
                break
        return claims

    def _score_yesno(
        self,
        answer: str,
        evidence_items: List[Tuple[str, str]],
        supporting_facts: Optional[Sequence[Sequence[Any]]],
    ) -> List[ClaimCheck]:
        """Score yes/no answers based on evidence quality (not fixed heuristics).
        
        Checks:
          1. Sufficient evidence quantity (facts + evidence items)
          2. Evidence diversity (multiple source titles = stronger for comparison)
          3. Evidence substance (enough tokens to support a boolean judgment)
          4. Presence of comparison/contrast signals (important for yes/no reasoning)
        """
        has_facts = supporting_facts is not None and len(supporting_facts) >= 2
        has_evidence = len(evidence_items) >= 2

        # Base score from evidence availability
        if has_facts and has_evidence:
            score = 0.65  # Start lower than before — must earn the rest
        elif has_facts or has_evidence:
            score = 0.40
        else:
            score = 0.15

        # Evidence diversity bonus: yes/no questions (esp. comparisons) need
        # info from multiple sources. Check how many distinct titles are present.
        if evidence_items:
            # Extract unique titles from evidence IDs (format: "title::sent_idx")
            titles = set()
            for ev_id, _ in evidence_items:
                parts = ev_id.split("::")
                if len(parts) >= 2:
                    titles.add(parts[0])
            if len(titles) >= 2:
                score += 0.10  # Multiple source titles = stronger
            elif len(titles) == 1:
                score += 0.03

        # Evidence substance check: do the evidence sentences have enough
        # content to support a boolean judgment?
        total_evidence_tokens = 0
        for _, ev_text in evidence_items:
            total_evidence_tokens += len(self._tokenize(ev_text))
        if total_evidence_tokens >= 15:
            score += 0.10  # Enough substance
        elif total_evidence_tokens >= 8:
            score += 0.05

        # Comparison/contrast signal bonus: look for words that indicate
        # the evidence contains the kind of reasoning needed for yes/no
        comparison_signals = {
            "both", "also", "same", "different", "unlike", "whereas",
            "however", "but", "although", "while", "than", "more", "less",
            "not", "never", "neither", "nor",
        }
        evidence_text_combined = " ".join(ev_text for _, ev_text in evidence_items).lower()
        signal_count = sum(1 for w in comparison_signals if w in evidence_text_combined)
        if signal_count >= 3:
            score += 0.10
        elif signal_count >= 1:
            score += 0.05

        score = min(max(score, 0.0), 1.0)

        # Find best evidence item
        best_id = evidence_items[0][0] if evidence_items else None
        best_text = evidence_items[0][1] if evidence_items else ""

        return [
            ClaimCheck(
                claim=answer,
                support_score=round(score, 4),
                is_supported=score >= self.claim_threshold,
                best_evidence_id=best_id,
                best_evidence_text=best_text,
            )
        ]

    def _flatten_evidence(self, evidence: Sequence[Any]) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for idx, ev in enumerate(evidence):
            # RerankResult-like object
            if hasattr(ev, "passage") and hasattr(ev, "supporting_sentences"):
                title = getattr(getattr(ev, "passage", None), "title", f"passage_{idx}")
                sent_indices = getattr(ev, "supporting_sentence_indices", []) or []
                sents = getattr(ev, "supporting_sentences", []) or []
                for j, sent in enumerate(sents):
                    sid = sent_indices[j] if j < len(sent_indices) else j
                    text = (sent or "").strip()
                    if text:
                        items.append((f"{title}::{sid}", text))
                continue

            # Passage-like object with raw sentences
            if hasattr(ev, "title") and hasattr(ev, "sentences"):
                title = getattr(ev, "title", f"passage_{idx}")
                for j, sent in enumerate(getattr(ev, "sentences", []) or []):
                    text = (sent or "").strip()
                    if text:
                        items.append((f"{title}::{j}", text))
                continue

            # Dict with sentence/text payload
            if isinstance(ev, dict):
                title = str(ev.get("title", f"passage_{idx}"))
                sent_idx = ev.get("sentence_index", ev.get("idx", 0))
                if "sentence" in ev or "text" in ev:
                    text = str(ev.get("sentence", ev.get("text", ""))).strip()
                    if text:
                        items.append((f"{title}::{sent_idx}", text))
                continue

            # Plain string evidence
            if isinstance(ev, str) and ev.strip():
                items.append((f"evidence::{idx}", ev.strip()))

        return items

    def _score_claim(self, claim: str, evidence_items: List[Tuple[str, str]]) -> ClaimCheck:
        best_id = None
        best_text = ""
        best_score = 0.0

        for ev_id, ev_text in evidence_items:
            score = self._score_pair(claim, ev_text)
            if score > best_score:
                best_score = score
                best_id = ev_id
                best_text = ev_text

        return ClaimCheck(
            claim=claim,
            support_score=round(float(best_score), 4),
            is_supported=best_score >= self.claim_threshold,
            best_evidence_id=best_id,
            best_evidence_text=best_text,
        )

    def _score_pair(self, claim: str, evidence_text: str) -> float:
        overlap_score = self._lexical_support(claim, evidence_text)

        if self.mode == "qa":
            qa_score = self._qa_support(claim, evidence_text)
            if qa_score is None:
                return overlap_score
            # QA score can be noisy for declarative claims, so keep lexical anchor.
            return 0.45 * overlap_score + 0.55 * qa_score

        if self.mode != "nli":
            return overlap_score

        nli_score = self._nli_support(claim, evidence_text)
        if nli_score is None:
            return overlap_score

        # Blend lexical overlap + entailment confidence for robustness.
        return 0.35 * overlap_score + 0.65 * nli_score

    def _lexical_support(self, claim: str, evidence_text: str) -> float:
        claim_toks = self._tokenize(claim)
        evidence_toks = self._tokenize(evidence_text)
        if not claim_toks or not evidence_toks:
            return 0.0

        overlap = len(claim_toks & evidence_toks)
        precision = overlap / max(len(claim_toks), 1)
        recall = overlap / max(len(evidence_toks), 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Favor coverage of claim tokens more than sentence compactness.
        return min(1.0, 0.75 * precision + 0.25 * f1)

    def _nli_support(self, claim: str, evidence_text: str) -> Optional[float]:
        pipe = self._load_nli_pipeline()
        if pipe is None:
            return None

        try:
            outputs = pipe({"text": evidence_text, "text_pair": claim}, truncation=True)
            if not isinstance(outputs, list):
                outputs = [outputs]
            label_scores = {o.get("label", "").lower(): float(o.get("score", 0.0)) for o in outputs}

            entail_keys = ["entailment", "label_2", "entails"]
            contradiction_keys = ["contradiction", "label_0", "contradicts"]

            entail = max((label_scores.get(k, 0.0) for k in entail_keys), default=0.0)
            contradiction = max((label_scores.get(k, 0.0) for k in contradiction_keys), default=0.0)
            return max(0.0, entail - 0.5 * contradiction)
        except Exception as e:
            log.warning(f"NLI inference failed, falling back to overlap: {e}")
            return None

    def _load_nli_pipeline(self):
        if self._nli_pipe is not None:
            return self._nli_pipe

        try:
            from transformers import pipeline

            log.info(f"Loading NLI model: {self.nli_model_name}")
            self._nli_pipe = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                top_k=None,
                device=self.nli_device,
            )
            return self._nli_pipe
        except Exception as e:
            log.warning(f"Could not load NLI mode ({self.nli_model_name}): {e}")
            self._nli_pipe = None
            return None

    def _qa_support(self, claim: str, evidence_text: str) -> Optional[float]:
        qa_mode_runner = self._load_qa_pipeline()
        if qa_mode_runner is None:
            return None

        try:
            if callable(qa_mode_runner):
                qa_out = qa_mode_runner(question=claim, context=evidence_text)
                raw_score = float(qa_out.get("score", 0.0))
                answer_text = str(qa_out.get("answer", "")).strip()
            else:
                raw_score, answer_text = self._qa_manual_inference(
                    claim=claim,
                    evidence_text=evidence_text,
                    qa_bundle=qa_mode_runner,
                )

            if not answer_text:
                return 0.0

            # Reject weak spans from noisy extraction.
            if raw_score < self.qa_min_answer_score:
                return raw_score * 0.5

            answer_overlap = self._lexical_support(answer_text, claim)
            # Reward confident extracted span + alignment of extracted span with claim.
            return min(1.0, 0.7 * raw_score + 0.3 * answer_overlap)
        except Exception as e:
            log.warning(f"QA inference failed, falling back to overlap: {e}")
            return None

    def _load_qa_pipeline(self):
        if self._qa_loader_failed:
            return None
        if self._qa_pipe is not None:
            return self._qa_pipe

        try:
            from transformers import pipeline

            log.info(f"Loading QA model: {self.qa_model_name}")
            self._qa_pipe = pipeline(
                "question-answering",
                model=self.qa_model_name,
                tokenizer=self.qa_model_name,
                device=self.qa_device,
            )
            return self._qa_pipe
        except Exception as primary_error:
            # Newer transformers builds may not expose question-answering pipeline.
            try:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                log.info(
                    "Falling back to manual QA model loading (AutoModelForQuestionAnswering)."
                )
                tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
                model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
                model.eval()
                self._qa_pipe = {"tokenizer": tokenizer, "model": model}
                return self._qa_pipe
            except Exception as fallback_error:
                log.warning(
                    f"Could not load QA mode ({self.qa_model_name}): "
                    f"{primary_error}; fallback failed: {fallback_error}"
                )
                self._qa_pipe = None
                self._qa_loader_failed = True
                return None

    def _qa_manual_inference(
        self,
        claim: str,
        evidence_text: str,
        qa_bundle: Dict[str, Any],
    ) -> Tuple[float, str]:
        try:
            import torch
        except Exception as e:
            log.warning(f"PyTorch unavailable for manual QA inference: {e}")
            return 0.0, ""

        tokenizer = qa_bundle["tokenizer"]
        model = qa_bundle["model"]

        encoded = tokenizer(
            claim,
            evidence_text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )

        with torch.no_grad():
            outputs = model(**encoded)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        start_idx = int(torch.argmax(start_logits).item())
        end_idx = int(torch.argmax(end_logits).item())

        if end_idx < start_idx:
            end_idx = start_idx
        # Keep span bounded to avoid very long / noisy extraction.
        if end_idx - start_idx > 16:
            end_idx = start_idx + 16

        input_ids = encoded["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        start_prob = torch.softmax(start_logits, dim=0)[start_idx].item()
        end_prob = torch.softmax(end_logits, dim=0)[end_idx].item()
        raw_score = float((start_prob * end_prob) ** 0.5)
        return raw_score, answer_text

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        return {t for t in tokens if t not in self._STOPWORDS and len(t) > 1}

    def _supporting_fact_coverage(
        self,
        supporting_facts: Optional[Sequence[Sequence[Any]]],
        evidence_items: List[Tuple[str, str]],
    ) -> Optional[float]:
        if not supporting_facts:
            return None

        available_ids = {ev_id for ev_id, _ in evidence_items}
        requested_ids = set()
        for sf in supporting_facts:
            if not isinstance(sf, (list, tuple)) or len(sf) < 2:
                continue
            title = str(sf[0])
            try:
                idx = int(sf[1])
            except (TypeError, ValueError):
                idx = sf[1]
            requested_ids.add(f"{title}::{idx}")

        if not requested_ids:
            return None

        covered = len(requested_ids & available_ids) / len(requested_ids)
        return round(float(covered), 4)

    def __repr__(self) -> str:
        return (
            f"Verifier(mode={self.mode}, support_threshold={self.support_threshold}, "
            f"claim_threshold={self.claim_threshold}, min_supported_claim_ratio={self.min_supported_claim_ratio}, "
            f"nli_model={self.nli_model_name}, qa_model={self.qa_model_name})"
        )
