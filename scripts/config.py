"""
Config Loader
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union

def get_best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class DataConfig:
    dev_distractor: str = "data/hotpot_dev_distractor_v1.json"
    dev_fullwiki: str = "data/hotpot_dev_fullwiki_v1.json"
    test_fullwiki: str = "data/hotpot_test_fullwiki_v1.json"
    limit: Optional[int] = None


@dataclass
class PromptsConfig:
    indexer_system: str = ""
    indexer_user: str = ""
    builder_citation: str = ""
    builder_standard: str = ""
    generator_specialist: str = ""
    builder_citation_yesno: str = ""

@dataclass
class MultihopConfig:
    hops: int = 2
    top_k_per_hop: int = 20
    max_bridge_entities: int = 5
    hop2_strategy: str = "both"
    llm_decompose_confidence_threshold: float = 1.0
    llm_decompose_ollama_model: str = "qwen3:8b"
    extraction_temperature: float = 0.0
    extraction_timeout: int = 1800
    fuzzy_title_threshold: int = 70


@dataclass
class RetrieverConfig:
    embed_model: str = "qwen3-embedding:8b"
    ollama_base_url: str = "http://localhost:11434"
    device: str = field(default_factory=get_best_device)
    batch_size: int = 32
    alpha: float = 0.7
    alpha_bridge: Optional[float] = 0.6
    alpha_comparison: Optional[float] = 0.5
    rrf_k: int = 20
    candidate_pool_size: int = 200
    top_k: int = 10
    multihop: MultihopConfig = field(default_factory=MultihopConfig)
    index_cache_dir: str = "index_cache"


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    sentence_model_name: str = "qwen3-embedding:8b"
    device: str = field(default_factory=get_best_device)
    top_k: int = 5                        
    sentence_score_threshold: float = 0.25
    max_sentences_per_passage: int = 5    
    batch_size: int = 32
    sentence_passage_limit: int = 3
    title_overlap_boost: float = 0.05


@dataclass
class PromptBuilderConfig:
    # Formatting options
    include_passage_numbers: bool = True       
    include_sentence_indices: bool = True      
    evidence_first: bool = True                
    max_evidence_chars: int = 9000
    bridge_keywords: list = field(default_factory=lambda: ["who", "which", "when", "where", "what", "how", "portrayed", "actor", "character", "played"])             

    complexity_length_weight: float = 0.10
    complexity_length_threshold: int = 50
    complexity_keywords_weight: float = 0.35
    complexity_confidence_weight: float = 0.20
    complexity_sentences_weight: float = 0.35
    complexity_sentence_threshold: int = 5     
    complexity_routing_threshold: float = 0.50  

    temperature_small_model: float = 0.1       
    temperature_large_model: float = 0.2       


@dataclass
class GeneratorConfig:
    ollama_base_url: str = "http://localhost:11434"
    model_small: str = "gemma3:12b"
    model_large: str = "qwen3:32b"
    request_timeout: int = 1800
    validate_citations: bool = True
    retry_on_parse_failure: bool = True
    specialist_mode: bool = False


@dataclass
class VerifierConfig:
    enabled: bool = True
    retry_on_failure: bool = True
    max_verification_retries: int = 1
    retry_score_threshold: float = 0.4
    mode: str = "overlap"  # "overlap", "nli", or "qa"
    support_threshold: float = 0.55
    claim_threshold: float = 0.45
    min_supported_claim_ratio: float = 1.0
    max_claims: int = 6
    nli_model_name: str = "roberta-large-mnli"
    nli_device: int = -1
    qa_model_name: str = "distilbert-base-cased-distilled-squad"
    qa_device: int = -1
    qa_min_answer_score: float = 0.2


@dataclass
class EvalConfig:
    limit: Optional[int] = 100
    predictions_dir: str = "results/predictions"
    metrics_dir: str = "results/metrics"
    parallel_workers: int = 3  # ThreadPoolExecutor workers for pipeline parallelism


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    prompt_builder: PromptBuilderConfig = field(default_factory=PromptBuilderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

    def validate(self) -> None:
        """Raise ValueError early if config values are out of sensible range."""
        r = self.retriever
        if not (0.0 <= r.alpha <= 1.0):
            raise ValueError(f"retriever.alpha must be in [0, 1], got {r.alpha}")
        if r.alpha_bridge is not None and not (0.0 <= r.alpha_bridge <= 1.0):
            raise ValueError(f"retriever.alpha_bridge must be in [0, 1], got {r.alpha_bridge}")
        if r.alpha_comparison is not None and not (0.0 <= r.alpha_comparison <= 1.0):
            raise ValueError(f"retriever.alpha_comparison must be in [0, 1], got {r.alpha_comparison}")
        if r.candidate_pool_size <= 0:
            raise ValueError(f"retriever.candidate_pool_size must be > 0, got {r.candidate_pool_size}")

        rr = self.reranker
        if rr.top_k <= 0:
            raise ValueError(f"reranker.top_k must be > 0, got {rr.top_k}")
        if not (0.0 <= rr.sentence_score_threshold <= 1.0):
            raise ValueError(f"reranker.sentence_score_threshold must be in [0, 1], got {rr.sentence_score_threshold}")

        pb = self.prompt_builder
        if not (0.0 <= pb.complexity_routing_threshold <= 1.0):
            raise ValueError(f"prompt_builder.complexity_routing_threshold must be in [0, 1], got {pb.complexity_routing_threshold}")
        if pb.max_evidence_chars <= 0:
            raise ValueError(f"prompt_builder.max_evidence_chars must be > 0, got {pb.max_evidence_chars}")

        ev = self.eval
        if ev.parallel_workers <= 0:
            raise ValueError(f"eval.parallel_workers must be > 0, got {ev.parallel_workers}")


def _dict_to_dataclass(cls, d: dict):
    if d is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in d.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            # Check if the field type is itself a dataclass
            if isinstance(ft, type) and hasattr(ft, '__dataclass_fields__') and isinstance(val, dict):
                kwargs[key] = _dict_to_dataclass(ft, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: Union[str, Path, None] = None) -> Config:
    """
    Load config from YAML file
    path: path to YAML file (default: configs/default.yaml)
    Returns: Config dataclass
    """
    if path is None:
        # Look for default.yaml relative to project root
        path = Path("configs/default.yaml")

    path = Path(path)
    if not path.exists():
        from scripts.logger import get_logger
        get_logger("config").warning(f"Config not found at {path}, using defaults")
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    if "data" in raw:
        cfg.data = _dict_to_dataclass(DataConfig, raw["data"])

    if "retriever" in raw:
        ret = raw["retriever"]
        if ret.get("device") in ("auto", "cpu"):
            ret["device"] = get_best_device()
        multihop_raw = ret.pop("multihop", None)
        cfg.retriever = _dict_to_dataclass(RetrieverConfig, ret)
        if multihop_raw:
            cfg.retriever.multihop = _dict_to_dataclass(MultihopConfig, multihop_raw)

    if "reranker" in raw:
        rr = raw["reranker"]
        if rr.get("device") in ("auto", "cpu"):
            rr["device"] = get_best_device()
        cfg.reranker = _dict_to_dataclass(RerankerConfig, rr)

    if "prompt_builder" in raw:
        cfg.prompt_builder = _dict_to_dataclass(PromptBuilderConfig, raw["prompt_builder"])

    if "generator" in raw:
        cfg.generator = _dict_to_dataclass(GeneratorConfig, raw["generator"])

    if "verifier" in raw:
        cfg.verifier = _dict_to_dataclass(VerifierConfig, raw["verifier"])

    if "eval" in raw:
        cfg.eval = _dict_to_dataclass(EvalConfig, raw["eval"])

    # Attempt to load prompts.yaml if it exists
    prompts_path = path.parent / "prompts.yaml"
    if prompts_path.exists():
        with open(prompts_path, "r") as f:
            raw_prompts = yaml.safe_load(f) or {}
        cfg.prompts = _dict_to_dataclass(PromptsConfig, raw_prompts)

    cfg.validate()
    return cfg
