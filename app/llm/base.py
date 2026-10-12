from abc import ABC, abstractmethod

from app.llm.models import LLMRequest, LLMResponse


class LLMGateway(ABC):
    """Provider-neutral contract used by the application layer."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one non-streaming chat response."""

    async def close(self) -> None:
        """Release provider resources when the gateway owns any."""
