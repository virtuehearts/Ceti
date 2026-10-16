"""BrainConfig — every knob for the brain framework in one place.

Low-spec defaults throughout: stdlib only, small timeouts, tiny models.
API keys are NEVER stored here — they come from the environment:
    OPENROUTER_API_KEY, NVIDIA_API_KEY
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def data_dir_default() -> str:
    """Where the brain keeps everything. CETI_DATA_DIR overrides the
    package-relative default — point it anywhere."""
    env = os.environ.get("CETI_DATA_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data")


def _data_dir() -> str:
    return data_dir_default()


@dataclass
class BrainConfig:
    # ---- tier switches -------------------------------------------------
    enable_aiml: bool = True      # instant pattern brain (~ms)
    enable_memory: bool = True    # fuzzy Q&A index over promoted answers
    enable_local: bool = True     # local Qwen via Ollama
    enable_cloud: bool = True     # OpenRouter / Nvidia fallback

    # ---- local LLM (Qwen, served by Ollama) ----------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    local_model: str = "qwen2.5-coder:1.5b"
    local_timeout: int = 120          # seconds; small local models answer fast
    local_max_tokens: int = 512
    local_temperature: float = 0.3

    # ---- cloud LLMs (OpenAI-compatible) --------------------------------
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    nvidia_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    cloud_model: str = "nvidia/nemotron-3.5-lightning:free"
    # Reasoning models need room AND time: they burn tokens on
    # their thinking trace before writing the answer, so a 512-token
    # budget starves them to null content (finish=length) and a 45s
    # timeout kills them mid-thought. Lightning answers in ~60s and
    # drives tools (verified 2026-10-16); the previous default,
    # nemotron-3-ultra-550b, 504s on the provider side.
    cloud_timeout: int = 180
    cloud_max_tokens: int = 2048
    cloud_temperature: float = 0.3
    # cloud backends tried in this order
    cloud_order: tuple = ("openrouter", "nvidia")

    # ---- relevance classifier ------------------------------------------
    promote_threshold: float = 0.65   # >= this: save to AIML automatically
    adjudicate_low: float = 0.45      # in [low, threshold): ask Qwen to judge
    adjudicate_with_llm: bool = True  # set False to skip the LLM judge call

    # ---- knowledge stores ----------------------------------------------
    data_dir: str = field(default_factory=_data_dir)
    seed_aiml: str = "seed.aiml"        # read-only starter patterns
    learned_aiml: str = "learned.aiml"  # auto-promoted categories live here
    memory_db: str = "memory.sqlite3"  # TF-IDF Q&A index (sqlite backend)
    knowledge_db: str = "knowledge.sqlite3"  # the database: qa + 768d
                                             # embeddings + aliases +
                                             # research log (vector backend)
    memory_backend: str = "auto"       # auto | vector | sqlite
    embed_url: str = "http://127.0.0.1:11435"  # embeddinggemma-2 server
    max_learned_categories: int = 500  # cap so the brain stays tiny
    memory_min_score: float = 0.35     # TF-IDF match acceptance floor
    vector_min_score: float = 0.70     # cosine match acceptance floor
                                             # measured 2026-10-12: true
                                             # paraphrases score 0.705-0.869,
                                             # unrelated "what is X" pairs
                                             # score 0.511-0.680. 0.70 splits
                                             # them; the old 0.60 let
                                             # false positives through.

    # ---- cassettes ----------------------------------------------------
    cassette_dir: str = "cassettes"    # swappable AIML topic datasets
    cassettes: str | list = "all"      # names to load, or "all"

    # ---- idle researcher ----------------------------------------------
    research_pages: int = 5            # pages per topic per cycle

    # ---- behavior -------------------------------------------------------
    learn_from_local: bool = True   # promote good local answers
    learn_from_cloud: bool = True   # promote good cloud answers
    # NOTE: learning always persists the PII-STRIPPED question/answer,
    # so nothing personal ever lands in the AIML file or the index.

    # ---- persona & handler mode ---------------------------------------
    persona_file: str = "persona.md"  # voice of the node (Jero-as-Khan)
    handler_mode: bool = True
    # When True the LLM is a HANDLER, not an oracle: on a memory hit it
    # presents the retrieved knowledge instead of free-generating.
    # The datasets answer; the model wields them.

    def aiml_paths(self) -> list[str]:
        base = os.path.abspath(self.data_dir)
        return [os.path.join(base, self.seed_aiml),
                os.path.join(base, self.learned_aiml)]

    def learned_path(self) -> str:
        return os.path.join(os.path.abspath(self.data_dir), self.learned_aiml)

    def memory_path(self) -> str:
        return os.path.join(os.path.abspath(self.data_dir), self.memory_db)

    def knowledge_path(self) -> str:
        return os.path.join(os.path.abspath(self.data_dir),
                            self.knowledge_db)

    @staticmethod
    def openrouter_key() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY") or None

    @staticmethod
    def nvidia_key() -> str | None:
        return os.environ.get("NVIDIA_API_KEY") or None
