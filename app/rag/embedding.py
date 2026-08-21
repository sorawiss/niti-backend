from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
    )