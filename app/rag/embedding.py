from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )