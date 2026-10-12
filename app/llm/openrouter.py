from typing import Any

import httpx

from app.llm.base import LLMGateway
from app.llm.config import OpenRouterSettings
from app.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.models import LLMRequest, LLMResponse, LLMUsage


class OpenRouterGateway(LLMGateway):
    def __init__(
        self,
        settings: OpenRouterSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout_seconds)
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.post(
                f"{self._settings.base_url.rstrip('/')}/chat/completions",
                headers=self._build_headers(),
                json=self._build_payload(request),
            )
        except httpx.TimeoutException as error:
            raise LLMTimeoutError(
                "The language model request timed out",
                code="llm_timeout",
                retryable=True,
            ) from error
        except httpx.RequestError as error:
            raise LLMUpstreamError(
                "Could not connect to the language model provider",
                code="llm_connection_error",
                retryable=True,
            ) from error

        if response.is_error:
            self._raise_for_error_response(response)

        return self._parse_response(response)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterGateway":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }
        if self._settings.app_url:
            headers["HTTP-Referer"] = self._settings.app_url
        if self._settings.app_name:
            headers["X-OpenRouter-Title"] = self._settings.app_name
        return headers

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self._settings.model
        max_output_tokens = (
            request.max_output_tokens or self._settings.max_output_tokens
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "stream": False,
            "provider": {
                "zdr": self._settings.zero_data_retention,
                "allow_fallbacks": self._settings.allow_fallbacks,
            },
        }
        if model.startswith("openai/gpt-5"):
            payload["max_completion_tokens"] = max_output_tokens
            payload["reasoning"] = {"effort": "low", "exclude": True}
        else:
            payload["max_tokens"] = max_output_tokens
            payload["temperature"] = (
                request.temperature
                if request.temperature is not None
                else self._settings.temperature
            )
        if request.response_format is not None:
            payload["response_format"] = request.response_format
            payload["provider"]["require_parameters"] = True
        if request.user_id:
            payload["user"] = request.user_id
        return payload

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")

            usage_data = data.get("usage") or {}
            metadata = data.get("openrouter_metadata") or {}
            return LLMResponse(
                content=content,
                model=data["model"],
                provider=data.get("provider") or metadata.get("provider_name"),
                finish_reason=choice.get("finish_reason"),
                request_id=data.get("id"),
                usage=LLMUsage(
                    prompt_tokens=usage_data.get("prompt_tokens"),
                    completion_tokens=usage_data.get("completion_tokens"),
                    total_tokens=usage_data.get("total_tokens"),
                    cost=usage_data.get("cost"),
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise LLMInvalidResponseError(
                "The language model returned an unexpected response",
                code="llm_invalid_response",
                status_code=response.status_code,
            ) from error

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
            error = body.get("error", body)
            if isinstance(error, dict):
                return str(error.get("message") or "Language model request failed")
        except ValueError:
            pass
        return "Language model request failed"

    def _raise_for_error_response(self, response: httpx.Response) -> None:
        status = response.status_code
        message = self._error_message(response)

        if status in {401, 403}:
            raise LLMAuthenticationError(
                message,
                code="llm_authentication_error",
                status_code=status,
            )
        if status == 429:
            raise LLMRateLimitError(
                message,
                code="llm_rate_limit",
                status_code=status,
                retryable=True,
            )
        if status in {408, 524}:
            raise LLMTimeoutError(
                message,
                code="llm_timeout",
                status_code=status,
                retryable=True,
            )
        if status >= 500 or status == 529:
            raise LLMUpstreamError(
                message,
                code="llm_upstream_error",
                status_code=status,
                retryable=True,
            )
        raise LLMRequestError(
            message,
            code="llm_request_error",
            status_code=status,
        )
