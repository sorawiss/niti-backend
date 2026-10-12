import json
import unittest

import httpx
from pydantic import SecretStr

from app.llm.config import OpenRouterSettings
from app.llm.models import ChatMessage, ChatRole, LLMRequest
from app.llm.openrouter import OpenRouterGateway


class OpenRouterGpt5ParameterTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpt5_uses_parameters_supported_by_zdr_endpoint(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["max_completion_tokens"], 1200)
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("temperature", payload)
            self.assertEqual(
                payload["reasoning"],
                {"effort": "low", "exclude": True},
            )
            self.assertTrue(payload["provider"]["require_parameters"])
            return httpx.Response(
                200,
                json={
                    "id": "generation-gpt5",
                    "model": "openai/gpt-5-mini",
                    "choices": [
                        {
                            "message": {"content": "{}"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        settings = OpenRouterSettings(
            api_key=SecretStr("test-key"),
            model="openai/gpt-5-mini",
            base_url="https://openrouter.test/api/v1",
            zero_data_retention=True,
            allow_fallbacks=False,
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gateway = OpenRouterGateway(settings, client=client)

        await gateway.generate(
            LLMRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="à¸„à¸³à¸–à¸²à¸¡")],
                response_format={"type": "json_schema", "json_schema": {}},
            )
        )

        await client.aclose()


if __name__ == "__main__":
    unittest.main()
