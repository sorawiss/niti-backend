import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 8192
REQUIRED_M3_HEADS = ("sparse_linear.pt", "colbert_linear.pt")


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridEmbedding:
    dense: list[float]
    sparse: SparseEmbedding


class BgeM3HybridEncoder:
    """Generate BGE-M3 dense and lexical vectors in a single model pass."""

    def __init__(
        self,
        model: Any,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self._model = model
        self._batch_size = batch_size
        self._max_length = max_length

    def encode_documents(self, texts: list[str]) -> list[HybridEmbedding]:
        return self._encode(texts)

    def encode_query(self, text: str) -> HybridEmbedding:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[HybridEmbedding]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty values")

        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        if len(dense_vectors) != len(texts) or len(lexical_weights) != len(texts):
            raise RuntimeError("BGE-M3 returned an unexpected number of embeddings")

        embeddings: list[HybridEmbedding] = []
        for dense_vector, weights in zip(dense_vectors, lexical_weights, strict=True):
            dense = (
                dense_vector.tolist()
                if hasattr(dense_vector, "tolist")
                else list(dense_vector)
            )
            sparse_items = sorted(
                (int(token_id), float(weight))
                for token_id, weight in weights.items()
                if float(weight) > 0
            )
            embeddings.append(
                HybridEmbedding(
                    dense=[float(value) for value in dense],
                    sparse=SparseEmbedding(
                        indices=[token_id for token_id, _ in sparse_items],
                        values=[weight for _, weight in sparse_items],
                    ),
                )
            )
        return embeddings


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _resolve_cached_model(cache_dir: str) -> str:
    """Reuse LlamaIndex's BGE-M3 snapshot and fetch only missing M3 heads."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached_config = try_to_load_from_cache(
        MODEL_NAME,
        "config.json",
        cache_dir=cache_dir,
    )
    if not isinstance(cached_config, str) or not Path(cached_config).is_file():
        return MODEL_NAME

    model_path = Path(cached_config).parent
    for filename in REQUIRED_M3_HEADS:
        if not (model_path / filename).is_file():
            downloaded = Path(
                hf_hub_download(
                    repo_id=MODEL_NAME,
                    filename=filename,
                    cache_dir=cache_dir,
                )
            )
            if downloaded.parent != model_path:
                raise RuntimeError(
                    "BGE-M3 base weights and lexical heads resolved to different revisions"
                )
    return str(model_path)


@lru_cache(maxsize=1)
def get_hybrid_encoder() -> BgeM3HybridEncoder:
    # Imported lazily so lightweight commands and unit tests do not load torch.
    from FlagEmbedding import BGEM3FlagModel
    from llama_index.core.utils import get_cache_dir

    cache_dir = os.getenv("BGE_M3_CACHE_DIR") or get_cache_dir()
    model_path = _resolve_cached_model(cache_dir)
    model = BGEM3FlagModel(
        model_path,
        cache_dir=cache_dir,
        use_fp16=_env_bool("BGE_M3_USE_FP16"),
    )
    return BgeM3HybridEncoder(
        model,
        batch_size=_env_int("BGE_M3_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        max_length=_env_int("BGE_M3_MAX_LENGTH", DEFAULT_MAX_LENGTH),
    )
