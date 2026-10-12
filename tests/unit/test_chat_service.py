import json
import unittest

from app.chat.models import LegalAnswerStatus
from app.chat.service import ChatGenerationError, ChatService
from app.llm.base import LLMGateway
from app.llm.models import LLMRequest, LLMResponse, LLMUsage


class FakeGateway(LLMGateway):
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=self.content,
            model="test/model",
            provider="test-provider",
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


def make_retriever(results: list[dict]):
    def retrieve(**kwargs):
        return results

    return retrieve


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_returns_only_valid_retrieved_citations(self) -> None:
        gateway = FakeGateway(
            json.dumps(
                {
                    "status": "answered",
                    "answer": "ลูกจ้างมีสิทธิได้รับค่าล่วงเวลา",
                    "citation_ids": ["LAW_1", "LAW_999", "LAW_1"],
                    "requires_human_review": False,
                },
                ensure_ascii=False,
            )
        )
        service = ChatService(
            gateway,
            retriever=make_retriever(
                [
                    {
                        "score": 0.91,
                        "text": "นายจ้างต้องจ่ายค่าล่วงเวลา",
                        "metadata": {
                            "law_name": "กฎหมายทดสอบ",
                            "section": "61",
                            "source_file": "law.jsonl",
                        },
                    }
                ]
            ),
        )

        result = await service.answer("ต้องได้ค่าล่วงเวลาหรือไม่")

        self.assertEqual(result.status, LegalAnswerStatus.ANSWERED)
        self.assertEqual([source.source_id for source in result.citations], ["LAW_1"])
        self.assertEqual(result.usage.total_tokens, 120)
        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(
            gateway.requests[0].response_format["type"],
            "json_schema",
        )

    async def test_answer_without_valid_citation_is_not_exposed(self) -> None:
        gateway = FakeGateway(
            json.dumps(
                {
                    "status": "answered",
                    "answer": "คำตอบที่ไม่มีหลักฐาน",
                    "citation_ids": ["LAW_999"],
                    "requires_human_review": False,
                }
            )
        )
        service = ChatService(
            gateway,
            retriever=make_retriever(
                [
                    {
                        "score": 0.8,
                        "text": "ข้อความกฎหมาย",
                        "metadata": {"law_name": "กฎหมาย", "section": "1"},
                    }
                ]
            ),
        )

        result = await service.answer("คำถาม")

        self.assertEqual(result.status, LegalAnswerStatus.INSUFFICIENT_CONTEXT)
        self.assertEqual(result.citations, [])
        self.assertTrue(result.requires_human_review)
        self.assertNotIn("คำตอบที่ไม่มีหลักฐาน", result.answer)

    async def test_empty_retrieval_skips_the_language_model(self) -> None:
        gateway = FakeGateway("not used")
        service = ChatService(gateway, retriever=make_retriever([]))

        result = await service.answer("คำถาม")

        self.assertEqual(result.status, LegalAnswerStatus.INSUFFICIENT_CONTEXT)
        self.assertEqual(gateway.requests, [])
        self.assertIsNone(result.model)

    async def test_invalid_json_raises_generation_error(self) -> None:
        gateway = FakeGateway("not-json")
        service = ChatService(
            gateway,
            retriever=make_retriever(
                [
                    {
                        "score": 0.8,
                        "text": "ข้อความกฎหมาย",
                        "metadata": {"law_name": "กฎหมาย", "section": "1"},
                    }
                ]
            ),
        )

        with self.assertRaises(ChatGenerationError):
            await service.answer("คำถาม")


if __name__ == "__main__":
    unittest.main()
