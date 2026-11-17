import asyncio
import json
import os
import unittest
from unittest.mock import patch

from app.llm.base import LLMGateway
from app.llm.config import (
    QueryRewriteSettings,
    QueryRewriteWorkflow,
    get_query_rewrite_settings,
)
from app.llm.errors import LLMInvalidResponseError, LLMRateLimitError
from app.llm.models import LLMRequest, LLMResponse, LLMUsage
from app.rag.rewrite import (
    QueryRewriteService,
    QueryRewriteStatus,
)


class FakeGateway(LLMGateway):
    def __init__(
        self,
        outcome: str | Exception = '{"rewritten_query":"????????????????"}',
        *,
        delay: float = 0,
    ) -> None:
        self.outcome = outcome
        self.delay = delay
        self.requests: list[LLMRequest] = []
        self.active = 0
        self.max_active = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return LLMResponse(
                content=self.outcome,
                model="deepseek/deepseek-v4.1-flash",
                provider="DeepSeek",
                finish_reason="stop",
                request_id="rewrite-request-1",
                usage=LLMUsage(
                    prompt_tokens=20,
                    completion_tokens=5,
                    total_tokens=25,
                    cost=0.0001,
                ),
                raw_response={
                    "id": "rewrite-request-1",
                    "model": "deepseek/deepseek-v4.1-flash",
                    "provider": "DeepSeek",
                    "choices": [
                        {
                            "message": {"content": self.outcome},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                        "cost": 0.0001,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                    "openrouter_metadata": {
                        "provider_name": "DeepSeek",
                        "generation_time": 0.15,
                    },
                },
            )
        finally:
            self.active -= 1


def settings(**overrides) -> QueryRewriteSettings:
    values = {
        "chat_enabled": True,
        "timeout_seconds": 1,
        "max_concurrency": 4,
    }
    values.update(overrides)
    return QueryRewriteSettings(**values)


class QueryRewriteSettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_query_rewrite_settings.cache_clear()

    def test_defaults_disable_every_workflow(self) -> None:
        names = [
            "QUERY_REWRITE_RAG_ENABLED",
            "QUERY_REWRITE_CHAT_ENABLED",
            "QUERY_REWRITE_CONTRACT_REVIEW_ENABLED",
            "QUERY_REWRITE_WHOLE_CONTRACT_ENABLED",
        ]
        with patch.dict(os.environ, {name: "false" for name in names}):
            get_query_rewrite_settings.cache_clear()
            loaded = get_query_rewrite_settings()

        self.assertTrue(
            all(
                not loaded.is_enabled(workflow)
                for workflow in QueryRewriteWorkflow
            )
        )

    def test_each_environment_flag_controls_only_its_workflow(self) -> None:
        mapping = {
            QueryRewriteWorkflow.RAG: "QUERY_REWRITE_RAG_ENABLED",
            QueryRewriteWorkflow.CHAT: "QUERY_REWRITE_CHAT_ENABLED",
            QueryRewriteWorkflow.CONTRACT_REVIEW: (
                "QUERY_REWRITE_CONTRACT_REVIEW_ENABLED"
            ),
            QueryRewriteWorkflow.WHOLE_CONTRACT: (
                "QUERY_REWRITE_WHOLE_CONTRACT_ENABLED"
            ),
        }
        for enabled_workflow, enabled_name in mapping.items():
            with self.subTest(workflow=enabled_workflow):
                environment = {name: "false" for name in mapping.values()}
                environment[enabled_name] = "true"
                with patch.dict(os.environ, environment):
                    get_query_rewrite_settings.cache_clear()
                    loaded = get_query_rewrite_settings()
                self.assertTrue(loaded.is_enabled(enabled_workflow))
                self.assertTrue(
                    all(
                        not loaded.is_enabled(workflow)
                        for workflow in QueryRewriteWorkflow
                        if workflow != enabled_workflow
                    )
                )


class QueryRewriteServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_returns_original_without_gateway_call(self) -> None:
        gateway = FakeGateway()
        service = QueryRewriteService(gateway, QueryRewriteSettings())

        result = await service.rewrite(
            "??????? OT ???",
            workflow=QueryRewriteWorkflow.CHAT,
        )

        self.assertEqual(result.status, QueryRewriteStatus.DISABLED)
        self.assertEqual(result.effective_query, result.original_query)
        self.assertEqual(gateway.requests, [])
        self.assertIsNone(result.usage.total_tokens)

    async def test_success_uses_model_override_and_strict_schema(self) -> None:
        gateway = FakeGateway()
        service = QueryRewriteService(gateway, settings())

        result = await service.rewrite(
            "??????? OT ???",
            workflow=QueryRewriteWorkflow.CHAT,
            user_id="user-1",
        )

        self.assertEqual(result.status, QueryRewriteStatus.REWRITTEN)
        self.assertEqual(result.effective_query, "????????????????")
        self.assertEqual(result.usage.total_tokens, 25)
        request = gateway.requests[0]
        self.assertEqual(request.model, "deepseek/deepseek-v4.1-flash")
        self.assertEqual(request.temperature, 0)
        self.assertEqual(request.max_output_tokens, 128)
        self.assertEqual(request.user_id, "user-1")
        self.assertEqual(request.response_format["type"], "json_schema")
        self.assertTrue(request.response_format["json_schema"]["strict"])
        self.assertEqual(len(request.messages), 2)
        self.assertEqual(
            json.loads(request.messages[1].content),
            {"query": "??????? OT ???"},
        )
        self.assertIn("untrusted data", request.messages[0].content)

    async def test_provider_and_validation_failures_fall_back(self) -> None:
        cases = [
            (
                LLMRateLimitError(
                    "slow down",
                    code="llm_rate_limit",
                    retryable=True,
                ),
                "llm_rate_limit",
            ),
            ("not-json", "invalid_rewrite"),
            ('{"rewritten_query":"   "}', "invalid_rewrite"),
            (
                json.dumps({"rewritten_query": "x" * 11}),
                "invalid_rewrite",
            ),
        ]
        for outcome, expected_code in cases:
            with self.subTest(error_code=expected_code, outcome=outcome):
                gateway = FakeGateway(outcome)
                service = QueryRewriteService(
                    gateway,
                    settings(max_query_chars=10),
                )
                result = await service.rewrite(
                    "original",
                    workflow=QueryRewriteWorkflow.CHAT,
                )
                self.assertEqual(
                    result.status,
                    QueryRewriteStatus.FALLBACK_ORIGINAL,
                )
                self.assertEqual(result.effective_query, "original")
                self.assertEqual(result.error_code, expected_code)

    async def test_timeout_falls_back_without_retry(self) -> None:
        gateway = FakeGateway(delay=0.05)
        service = QueryRewriteService(
            gateway,
            settings(timeout_seconds=0.001),
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            result = await service.rewrite(
                "original",
                workflow=QueryRewriteWorkflow.CHAT,
            )

        self.assertEqual(result.status, QueryRewriteStatus.FALLBACK_ORIGINAL)
        self.assertEqual(result.error_code, "query_rewrite_timeout")
        self.assertEqual(len(gateway.requests), 1)
        payload = json.loads(logs.output[0].split("query_rewrite=", 1)[1])
        self.assertEqual(payload["event"], "query_rewrite_llm_completed")
        self.assertTrue(payload["llm_called"])
        self.assertEqual(payload["model"], "deepseek/deepseek-v4.1-flash")
        self.assertIsNone(payload["request_id"])

    async def test_concurrency_is_bounded(self) -> None:
        gateway = FakeGateway(delay=0.02)
        service = QueryRewriteService(
            gateway,
            settings(max_concurrency=2),
        )

        results = await asyncio.gather(
            *(
                service.rewrite(
                    f"query-{index}",
                    workflow=QueryRewriteWorkflow.CHAT,
                )
                for index in range(5)
            )
        )

        self.assertTrue(
            all(result.status == QueryRewriteStatus.REWRITTEN for result in results)
        )
        self.assertEqual(gateway.max_active, 2)

    async def test_log_has_detailed_telemetry_and_excludes_content(self) -> None:
        gateway = FakeGateway()
        service = QueryRewriteService(gateway, settings())
        secret_query = "private-contract-clause"

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            await service.rewrite(
                secret_query,
                workflow=QueryRewriteWorkflow.CHAT,
            )

        combined = "\n".join(logs.output)
        payload = json.loads(combined.split("query_rewrite=", 1)[1])

        self.assertEqual(payload["event"], "query_rewrite_llm_completed")
        self.assertTrue(payload["llm_called"])
        self.assertEqual(payload["request_id"], "rewrite-request-1")
        self.assertEqual(payload["finish_reason"], "stop")
        self.assertEqual(payload["usage"]["cost"], 0.0001)
        self.assertEqual(
            payload["usage"]["prompt_tokens_details"]["cached_tokens"],
            3,
        )
        self.assertEqual(
            payload["provider_metadata"]["provider_name"],
            "DeepSeek",
        )
        self.assertNotIn(secret_query, combined)
        self.assertNotIn(gateway.outcome, combined)


    async def test_raw_response_logging_can_be_enabled(self) -> None:
        gateway = FakeGateway()
        service = QueryRewriteService(
            gateway,
            settings(log_raw_response=True),
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            await service.rewrite(
                "developer-debug-query",
                workflow=QueryRewriteWorkflow.CHAT,
            )

        payload = json.loads(logs.output[0].split("query_rewrite=", 1)[1])
        self.assertEqual(payload["raw_response"]["id"], "rewrite-request-1")
        self.assertEqual(
            payload["raw_response"]["choices"][0]["message"]["content"],
            gateway.outcome,
        )

    async def test_raw_invalid_gateway_response_is_logged(self) -> None:
        raw_response = '{"choices":[],"debug":"upstream-payload"}'
        gateway = FakeGateway(
            LLMInvalidResponseError(
                "unexpected response",
                code="llm_invalid_response",
                status_code=200,
                raw_response=raw_response,
            )
        )
        service = QueryRewriteService(
            gateway,
            settings(log_raw_response=True),
        )

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            result = await service.rewrite(
                "developer-debug-query",
                workflow=QueryRewriteWorkflow.CHAT,
            )

        payload = json.loads(logs.output[0].split("query_rewrite=", 1)[1])
        self.assertEqual(result.error_code, "llm_invalid_response")
        self.assertEqual(payload["raw_response"], raw_response)


if __name__ == "__main__":
    unittest.main()
