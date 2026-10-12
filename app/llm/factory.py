import os

from app.llm.base import LLMGateway
from app.llm.config import get_openrouter_settings
from app.llm.openrouter import OpenRouterGateway


def create_llm_gateway(provider: str | None = None) -> LLMGateway:
    selected_provider = provider or os.getenv("LLM_PROVIDER", "openrouter")
    normalized_provider = selected_provider.strip().lower()
    if normalized_provider == "openrouter":
        return OpenRouterGateway(get_openrouter_settings())

    raise ValueError(f"Unsupported LLM provider: {normalized_provider}")