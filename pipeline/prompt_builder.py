from dataclasses import dataclass
from typing import List, Optional, Dict
import re

from pipeline.data_loader import Passage, SupportingFact
from pipeline.indexer import RetrievalResult
from pipeline.reranker import RerankResult
from scripts.config import load_config
from scripts.logger import get_logger

log = get_logger("prompt_builder")


@dataclass
class PromptBuilderOutput:
    prompt: str                           
    complexity_score: float               
    target_model: str                     
    passage_titles: List[str]             
    supporting_fact_indices: Dict         
    fact_mapping: Dict                    
    metadata: Dict                        


class PromptBuilder:
    def __init__(self, cfg=None, prompts=None):
        if cfg is None or isinstance(cfg, str):
            cfg = load_config(cfg).prompt_builder
        self.cfg = cfg
        self.prompts = prompts
        log.success("PromptBuilder initialized.")

    @classmethod
    def from_config(cls, cfg=None) -> "PromptBuilder":
        if cfg is None or isinstance(cfg, (str,)):
            cfg = load_config(cfg)
        return cls(cfg.prompt_builder, getattr(cfg, "prompts", None))

    def build(
        self,
        query: str,
        reranked_results: List[RerankResult],
        include_metadata: bool = True,
        use_citation_selection: bool = True,
        question_type: str = "",
    ) -> PromptBuilderOutput:
        if not reranked_results:
            raise ValueError("No reranked results provided")

        # Sort by reranker score (highest first) for better LLM prioritization
        reranked_results = sorted(reranked_results, key=lambda r: r.score, reverse=True)

        passage_titles = [r.passage.title for r in reranked_results]
        num_passages = len(reranked_results)
        evidence_block = self._format_evidence_block(reranked_results)
        complexity_score = self._compute_complexity_score(query, reranked_results)

        supporting_fact_indices = self._extract_supporting_fact_indices(reranked_results)
        facts_list_str, fact_mapping = self._format_facts_list(reranked_results)

        # Ensure complexity_routing_threshold is a float (defensive cast for config loading issues)
        threshold = float(self.cfg.complexity_routing_threshold) if hasattr(self.cfg, 'complexity_routing_threshold') else 0.6
        target_model = "complex" if complexity_score > threshold else "simple"

        if use_citation_selection:
            instructions = self._build_instructions(
                query, facts_list_str=facts_list_str, fact_mapping=fact_mapping,
                question_type=question_type,
            )
        else:
            instructions = self._build_instructions(query, question_type=question_type)

        if self.cfg.evidence_first:
            prompt = f"{evidence_block}\n\n{instructions}"
        else:
            prompt = f"{instructions}\n\n{evidence_block}"

        if len(prompt) > self.cfg.max_evidence_chars:
            log.warning(f"Prompt too long ({len(prompt)} chars), truncating to {self.cfg.max_evidence_chars}")
            prompt = prompt[:self.cfg.max_evidence_chars] + "\n[... truncated ...]"

        metadata = {} if not include_metadata else {
            "num_passages": num_passages,
            "prompt_length": len(prompt),
            "complexity_raw": complexity_score,
            "total_sentences": sum(len(r.supporting_sentences) for r in reranked_results),
            "avg_retrieval_score": sum(r.retrieval_score for r in reranked_results) / num_passages,
            "avg_rerank_score": sum(r.score for r in reranked_results) / num_passages,
            "num_facts": len(fact_mapping),
        }

        return PromptBuilderOutput(
            prompt=prompt,
            complexity_score=complexity_score,
            target_model=target_model,
            passage_titles=passage_titles,
            supporting_fact_indices=supporting_fact_indices,
            fact_mapping=fact_mapping,
            metadata=metadata,
        )

    def _format_evidence_block(self, results: List[RerankResult]) -> str:
        lines = ["Evidence:"]

        for passage_idx, result in enumerate(results):
            passage = result.passage
            lines.append(f"Passage {passage_idx}: [{passage.title}]")

            # Rank-aware: top-2 passages get 3 sentences, rest get 2
            max_sents = 3 if passage_idx < 2 else 2
            for sent_str, sent_idx in zip(result.supporting_sentences[:max_sents], result.supporting_sentence_indices[:max_sents]):
                display_sent = sent_str[:180] + "..." if len(sent_str) > 180 else sent_str
                lines.append(f"  [{sent_idx}] {display_sent}")

        return "\n".join(lines)

    def _compute_complexity_score(self, query: str, results: List[RerankResult]) -> float:
        score = 0.0

        # Factor 1: Query length
        has_long_query = len(query) > self.cfg.complexity_length_threshold
        score += self.cfg.complexity_length_weight * float(has_long_query)

        # Factor 2: Bridge keywords
        query_lower = query.lower()
        has_bridge_kw = any(kw in query_lower for kw in getattr(self.cfg, "bridge_keywords", ["who", "which", "where"]))
        score += self.cfg.complexity_keywords_weight * float(has_bridge_kw)

        # Factor 3: Confidence (how ambiguous is the retrieval?)
        # Better proxy than retrieval_score: use reranker top-score gap
        if len(results) >= 2:
            top = float(results[0].score)
            second = float(results[1].score)
            denom = abs(top) + 1e-6
            gap = max(min((top - second) / denom, 1.0), 0.0)
            confidence_penalty = 1.0 - gap
        else:
            confidence_penalty = 0.5

        # avg_confidence = sum(r.retrieval_score for r in results) / len(results) if results else 0.5
        # # Lower confidence = harder (invert the score)
        # confidence_penalty = 1.0 - min(avg_confidence, 1.0)
        score += self.cfg.complexity_confidence_weight * confidence_penalty

        # Factor 4: Supporting sentence count (many sentences = complex reasoning)
        total_sentences = sum(len(r.supporting_sentences) for r in results)
        has_many_sentences = total_sentences > self.cfg.complexity_sentence_threshold
        score += self.cfg.complexity_sentences_weight * float(has_many_sentences)

        return min(max(score, 0.0), 1.0)

    def _extract_supporting_fact_indices(self, results: List[RerankResult]) -> Dict:
        fact_indices = {}

        for passage_idx, result in enumerate(results):
            passage = result.passage
            facts = [
                (passage.title, sent_idx)
                for sent_idx in result.supporting_sentence_indices
            ]
            fact_indices[passage_idx] = facts

        return fact_indices

    def _format_facts_list(self, results: List[RerankResult]) -> str:
        lines = ["AVAILABLE FACTS:"]
        fact_mapping = {}  # fact_number -> (title, sentence_idx)
        fact_number = 0

        for passage_idx, result in enumerate(results):
            passage = result.passage
            # Rank-aware: top-2 passages get 3 facts, rest get 2
            max_facts = 3 if passage_idx < 2 else 2
            for sent_str, sent_idx in list(zip(result.supporting_sentences, result.supporting_sentence_indices))[:max_facts]:
                display_sent = sent_str[:180] + "..." if len(sent_str) > 180 else sent_str
                lines.append(f"Fact {fact_number}: [{passage.title}, {sent_idx}] {display_sent}")
                fact_mapping[fact_number] = (passage.title, sent_idx)
                fact_number += 1
        return "\n".join(lines), fact_mapping

    def _is_yesno_question(self, query: str, question_type: str = "") -> bool:
        """Detect yes/no questions via syntax heuristics.
        
        IMPORTANT: Do NOT assume all comparison questions are yes/no.
        In HotpotQA, many comparison questions expect entity answers:
          - "Which writer was from England?" → entity answer
          - "Who is older, X or Y?" → entity answer
          - "Are both X and Y from Z?" → yes/no answer
        
        Only route to yes/no when the syntax clearly indicates a boolean question.
        """
        q_lower = query.strip().lower()
        
        # Entity-seeking questions are NEVER yes/no, even within comparison type
        entity_starters = (
            "which ", "who ", "what ", "where ", "when ", "how ",
            "name ", "in which ",
        )
        if q_lower.startswith(entity_starters):
            return False
        
        # Syntax-based detection: questions starting with boolean indicators
        yesno_starters = (
            "is ", "are ", "was ", "were ", "did ", "does ", "do ",
            "has ", "have ", "had ", "can ", "could ", "will ", "would ",
            "should ", "shall ",
        )
        return q_lower.startswith(yesno_starters)

    def _build_instructions(self, query: str, facts_list_str: str = "",
                            fact_mapping: Dict = None, question_type: str = "") -> str:
        is_yesno = self._is_yesno_question(query, question_type)

        if fact_mapping is not None and facts_list_str:
            # Yes/No questions get a specialized prompt for boolean reasoning
            if is_yesno and self.prompts and self.prompts.builder_citation_yesno:
                log.info("Using yes/no specialized prompt")
                return self.prompts.builder_citation_yesno.format(
                    query=query, facts_list_str=facts_list_str
                )
            # Citation selection approach
            if self.prompts and self.prompts.builder_citation:
                return self.prompts.builder_citation.format(query=query, facts_list_str=facts_list_str)
            else:
                instructions = f"""QUESTION: {query}

{facts_list_str}

TASK: Answer the question using ONLY the available facts. Think step by step.

RESPOND WITH THIS JSON FORMAT - NOTHING ELSE:
{{
  "supporting_fact_numbers": [0, 1, 3],
  "answer": "Short answer here"
}}

RULES:
- Output ONLY raw valid JSON. Do NOT wrap it in markdown backticks (```json).
- "supporting_fact_numbers": list ALL fact numbers that support your answer based on AVAILABLE FACTS.
- "answer": short and direct (a few words).
- For yes/no questions, answer "yes" or "no" (lowercase).
- You MUST provide an answer using the available facts. Even if you only have partial evidence, give your best logical guess.
- NEVER output "Cannot determine from evidence", always make a prediction.
- For 'who/which is older/larger/first' questions, answer with the ENTITY NAME.
"""
        else:
            if self.prompts and self.prompts.builder_standard:
                return self.prompts.builder_standard.format(query=query)
            else:
                instructions = f"""Question: {query}

Instructions:
1. Read the evidence above carefully.
2. Identify which evidence passages (Passage 0, 1, 2, etc.) are relevant to the question.
3. Reason step-by-step using ONLY the evidence provided.
4. List the supporting facts by citing the passage number and sentence index: [passage_num, sentence_idx].
5. Generate a clear, concise answer based on the evidence.

Output Format (JSON):
{{
  "supporting_facts": [[0, 0], [1, 3], [2, 1]],
  "answer": "Your answer here"
}}

Important:
- Output ONLY raw valid JSON, no markdown formatting or backticks around it.
- supporting_facts is a list of [passage_number, sentence_index] pairs (NOT title).
- Passage numbers are 0-4 as shown in the evidence block above.
- You MUST provide an answer using the available facts. Even if you only have partial evidence, give your best logical guess.
- NEVER output "Cannot determine from evidence" unless the facts are completely unrelated to the question.
"""
        return instructions

    def __repr__(self) -> str:
        return (
            f"PromptBuilder(evidence_first={self.cfg.evidence_first}, "
            f"complexity_threshold={self.cfg.complexity_routing_threshold}, "
            f"max_chars={self.cfg.max_evidence_chars})"
        )


