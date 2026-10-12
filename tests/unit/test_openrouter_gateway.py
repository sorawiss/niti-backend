import json
import unittest

import httpx
from pydantic import SecretStr

from app.llm.config import OpenRouterSettings
from app.llm.errors import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.models import ChatMessage, ChatRole, LLMRequest
from app.llm.openrouter import OpenRouterGateway


def make_settings() -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key=SecretStr("test-key"),
        model="test/model",
        base_url="https://openrouter.test/api/v1",
        zero_data_retention=True,
        allow_fallbacks=False,
    )


class OpenRouterGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_maps_request_and_response(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "test/model")
            self.assertTrue(payload["provider"]["zdr"])
            self.assertFalse(payload["provider"]["allow_fallbacks"])
            return httpx.Response(
                200,
                json={
                    "id": "generation-1",
                    "model": "test/model",
                    "provider": "test-provider",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "คำตอบ"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                        "cost": 0.001,
                    },
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        response = await gateway.generate(
            LLMRequest(messages=[ChatMessage(role=ChatRole.USER, content="คำถาม")])
        )

        self.assertEqual(response.content, "คำตอบ")
        self.assertEqual(response.provider, "test-provider")
        self.assertEqual(response.usage.total_tokens, 14)
        self.assertEqual(response.usage.cost, 0.001)
        await client.aclose()

    async def test_response_format_requires_supported_parameters(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertTrue(payload["provider"]["require_parameters"])
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            return httpx.Response(
                200,
                json={
                    "id": "generation-2",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {"content": "{}"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        await gateway.generate(
            LLMRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="คำถาม")],
                response_format={"type": "json_schema", "json_schema": {}},
            )
        )
        await client.aclose()

    async def test_authentication_error_is_mapped(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        with self.assertRaises(LLMAuthenticationError) as context:
            await gateway.generate(
                LLMRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertFalse(context.exception.retryable)
        await client.aclose()

    async def test_rate_limit_is_retryable(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        with self.assertRaises(LLMRateLimitError) as context:
            await gateway.generate(
                LLMRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertTrue(context.exception.retryable)
        await client.aclose()

    async def test_timeout_is_mapped(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        with self.assertRaises(LLMTimeoutError):
            await gateway.generate(
                LLMRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        await client.aclose()

    async def test_invalid_success_response_is_rejected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(make_settings(), client=client)
        with self.assertRaises(LLMInvalidResponseError):
            await gateway.generate(
                LLMRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
