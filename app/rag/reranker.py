import math
import os
from collections.abc import Sequence
from functools import lru_cache
from threading import Lock
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


class RerankerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    fail_open: bool = True
    model: str = DEFAULT_RERANKER_MODEL
    candidate_limit: int = Field(default=25, ge=1)
    batch_size: int = Field(default=8, ge=1)
    query_max_length: int = Field(default=512, ge=1)
    max_length: int = Field(default=1024, ge=1)
    use_fp16: bool = False
    cache_dir: str | None = None


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class BgeReranker:
    """Thread-safe adapter around FlagEmbedding's encoder-only reranker."""

    def __init__(self, model: Any, *, batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model = model
        self._batch_size = batch_size
        self._lock = Lock()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if any(not passage.strip() for passage in passages):
            raise ValueError("passages must not contain empty values")
        if not passages:
            return []

        pairs = [[query, passage] for passage in passages]
        with self._lock:
            raw_scores = self._model.compute_score(
                pairs,
                batch_size=self._batch_size,
                normalize=True,
            )

        if isinstance(raw_scores, (int, float)):
            values = [raw_scores]
        else:
            values = list(raw_scores)
        if len(values) != len(passages):
            raise RuntimeError("reranker returned an unexpected number of scores")

        scores: list[float] = []
        for value in values:
            score = float(value)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("reranker returned an invalid normalized score")
            scores.append(score)
        return scores


@lru_cache(maxsize=1)
def get_reranker_settings() -> RerankerSettings:
    load_dotenv()
    return RerankerSettings(
        enabled=_read_bool("RERANKER_ENABLED", False),
        fail_open=_read_bool("RERANKER_FAIL_OPEN", True),
        model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        candidate_limit=int(os.getenv("RERANKER_CANDIDATE_LIMIT", "25")),
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        query_max_length=int(os.getenv("RERANKER_QUERY_MAX_LENGTH", "512")),
        max_length=int(os.getenv("RERANKER_MAX_LENGTH", "1024")),
        use_fp16=_read_bool("RERANKER_USE_FP16", False),
        cache_dir=os.getenv("RERANKER_CACHE_DIR") or None,
    )


@lru_cache(maxsize=1)
def get_reranker() -> BgeReranker:
    # Imported lazily so disabled reranking does not load torch or model weights.
    from FlagEmbedding import FlagReranker

    settings = get_reranker_settings()
    model = FlagReranker(
        settings.model,
        cache_dir=settings.cache_dir,
        query_max_length=settings.query_max_length,
        max_length=settings.max_length,
        use_fp16=settings.use_fp16,
    )
    return BgeReranker(model, batch_size=settings.batch_size)
