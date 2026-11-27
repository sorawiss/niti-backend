import json
import unittest

from app.chat.models import LegalAnswerStatus
from app.chat.service import ChatGenerationError, ChatService
from app.llm.base import LLMGateway
from app.llm.models import LLMRequest, LLMResponse, LLMUsage


class FakeGateway(LLMGateway):
    def __init__(self, content: str, *, plan_content: str = '{"queries": []}') -> None:
        self.content = content
        self.plan_content = plan_content
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        schema = (request.response_format or {}).get("json_schema", {})
        return LLMResponse(
            content=(
                self.plan_content
                if schema.get("name") == "legal_chat_search_plan"
                else self.content
            ),
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
                        "retrieval_score": 0.032,
                        "rerank_score": 0.91,
                        "retrieval_rank": 4,
                        "rerank_rank": 1,
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
        self.assertEqual(result.citations[0].retrieval_score, 0.032)
        self.assertEqual(result.citations[0].rerank_score, 0.91)
        self.assertEqual(result.citations[0].retrieval_rank, 4)
        self.assertEqual(result.citations[0].rerank_rank, 1)
        self.assertEqual(result.usage.total_tokens, 120)
        self.assertEqual(len(gateway.requests), 2)
        self.assertEqual(
            gateway.requests[-1].response_format["type"],
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

    async def test_empty_retrieval_without_followup_evidence_skips_answer_generation(self) -> None:
        gateway = FakeGateway("not used")
        service = ChatService(gateway, retriever=make_retriever([]))

        result = await service.answer("คำถาม")

        self.assertEqual(result.status, LegalAnswerStatus.INSUFFICIENT_CONTEXT)
        self.assertEqual(len(gateway.requests), 1)
        self.assertIsNone(result.model)

    async def test_search_plan_can_cite_laws_outside_initial_filter(self) -> None:
        question = "overtime rate"
        calls: list[tuple[str, int, str | None]] = []
        initial = {
            "text": "ชั่วโมงทำงาน",
            "metadata": {"section": "23", "law_name": "กฎหมายฉบับแรก"},
        }
        supplemental = {
            "text": "อัตราค่าล่วงเวลา",
            "metadata": {"section": "61", "law_name": "กฎหมายอีกฉบับ"},
        }

        def retrieve(*, query, top_k, law_name):
            calls.append((query, top_k, law_name))
            return [initial, supplemental] if query == "ค่าล่วงเวลา" else [initial]

        gateway = FakeGateway(
            '{"status":"answered","answer":"คำตอบ","citation_ids":["LAW_2"],'
            '"requires_human_review":false}',
            plan_content=json.dumps(
                {"queries": ["Overtime  RATE", "ค่าล่วงเวลา", "ค่าล่วงเวลา"]}
            ),
        )
        for law_name in ("กฎหมายฉบับแรก", None):
            with self.subTest(law_name=law_name):
                calls.clear()
                gateway.requests.clear()
                result = await ChatService(gateway, retriever=retrieve).answer(
                    question, top_k=8, law_name=law_name
                )

                expected_calls = [(question, 8, law_name)]
                if law_name:
                    expected_calls.append(("Overtime  RATE", 5, None))
                expected_calls.append(("ค่าล่วงเวลา", 5, None))
                self.assertEqual(calls, expected_calls)
                self.assertIn(initial["text"], gateway.requests[0].messages[-1].content)
                self.assertEqual(
                    [source.law_name for source in result.citations], ["กฎหมายอีกฉบับ"]
                )

    async def test_search_plan_recovers_from_empty_initial_retrieval_and_failed_search(self) -> None:
        def retrieve(*, query, **kwargs):
            if query == "หมดเวลา":
                raise TimeoutError()
            if query == "มาตรา 61":
                return [{"text": "อัตราค่าล่วงเวลา"}]
            return []

        gateway = FakeGateway(
            '{"status":"answered","answer":"คำตอบ","citation_ids":["LAW_1"],'
            '"requires_human_review":false}',
            plan_content=json.dumps({"queries": ["หมดเวลา", "มาตรา 61"]}),
        )
        result = await ChatService(gateway, retriever=retrieve).answer("คำถาม")

        self.assertEqual(result.status, LegalAnswerStatus.ANSWERED)
        self.assertEqual(result.citations[0].text, "อัตราค่าล่วงเวลา")

    async def test_invalid_search_plan_uses_initial_evidence(self) -> None:
        gateway = FakeGateway(
            '{"status":"answered","answer":"คำตอบ","citation_ids":["LAW_1"],'
            '"requires_human_review":false}',
            plan_content='{"queries": ["a", "b", "c", "d"]}',
        )
        service = ChatService(
            gateway, retriever=make_retriever([{"text": "ข้อความกฎหมาย"}])
        )

        result = await service.answer("คำถาม")

        self.assertEqual(result.status, LegalAnswerStatus.ANSWERED)
        self.assertEqual(len(result.citations), 1)

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
