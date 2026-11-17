import json
import unittest
from unittest.mock import patch

from app.api.rag import RetrieveRequest, retrieve
from app.chat.service import ChatService
from app.documents.models import DocumentBlock, ParsedDocument
from app.llm.config import QueryRewriteWorkflow
from app.rag.rewrite import (
    QueryRewriteResult,
    QueryRewriteStatus,
)
from app.reviews.service import ClauseRetrievalService
from tests.unit.test_chat_service import FakeGateway


class FakeRewriter:
    def __init__(self, effective_query: str) -> None:
        self.effective_query = effective_query
        self.calls: list[tuple[str, QueryRewriteWorkflow]] = []

    async def rewrite(
        self,
        query: str,
        *,
        workflow: QueryRewriteWorkflow,
        user_id: str | None = None,
    ) -> QueryRewriteResult:
        self.calls.append((query, workflow))
        return QueryRewriteResult(
            status=QueryRewriteStatus.REWRITTEN,
            original_query=query,
            effective_query=self.effective_query,
            model="deepseek/deepseek-v4.1-flash",
            provider="DeepSeek",
        )


class QueryRewriteIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_retrieves_rewrite_but_answers_original(self) -> None:
        original = "??????? OT ???"
        effective = "???????????????? ??????? ???????"
        calls: list[str] = []

        def retriever(**kwargs):
            calls.append(kwargs["query"])
            return [
                {
                    "score": 0.9,
                    "text": "???????????????????????????????????",
                    "metadata": {
                        "law_name": "????????????????????",
                        "section": "61",
                    },
                }
            ]

        gateway = FakeGateway(
            json.dumps(
                {
                    "status": "answered",
                    "answer": "????????????????????????",
                    "citation_ids": ["LAW_1"],
                    "requires_human_review": False,
                },
                ensure_ascii=False,
            )
        )
        rewriter = FakeRewriter(effective)
        service = ChatService(
            gateway,
            retriever=retriever,
            query_rewriter=rewriter,
        )

        result = await service.answer(original)

        self.assertEqual(calls, [effective])
        self.assertEqual(
            rewriter.calls,
            [(original, QueryRewriteWorkflow.CHAT)],
        )
        self.assertEqual(result.rewrite.effective_query, effective)
        prompt_text = "\n".join(
            message.content for message in gateway.requests[0].messages
        )
        self.assertIn(original, prompt_text)
        self.assertNotIn(effective, prompt_text)

    async def test_rag_api_retrieves_with_effective_query(self) -> None:
        original = "OT"
        effective = "????????????????"
        rewriter = FakeRewriter(effective)
        calls: list[dict] = []

        def fake_retrieve_laws(**kwargs):
            calls.append(kwargs)
            return []

        with patch("app.api.rag.retrieve_laws", fake_retrieve_laws):
            response = await retrieve(
                RetrieveRequest(query=original, top_k=3),
                rewriter,
            )

        self.assertEqual(calls[0]["query"], effective)
        self.assertEqual(response.query, original)
        self.assertEqual(response.rewrite.effective_query, effective)
        self.assertEqual(
            rewriter.calls,
            [(original, QueryRewriteWorkflow.RAG)],
        )

    async def test_clause_retrieval_uses_selected_workflow_and_keeps_original(self) -> None:
        original = "Employee overtime"
        effective = "???????????????? ???????"
        rewriter = FakeRewriter(effective)
        calls: list[str] = []

        def retriever(**kwargs):
            calls.append(kwargs["query"])
            return []

        document = ParsedDocument(
            document_id="doc-1",
            filename="contract.docx",
            blocks=[
                DocumentBlock(
                    index=0,
                    type="paragraph",
                    text=original,
                )
            ],
        )
        service = ClauseRetrievalService(
            retriever,
            query_rewriter=rewriter,
        )

        preparation = await service.prepare(
            document,
            workflow=QueryRewriteWorkflow.WHOLE_CONTRACT,
        )

        query_result = preparation.clause_results[0].query_results[0]
        self.assertEqual(calls, [effective])
        self.assertEqual(
            rewriter.calls[0][1],
            QueryRewriteWorkflow.WHOLE_CONTRACT,
        )
        self.assertEqual(
            preparation.clause_results[0].queries[0].text,
            original,
        )
        self.assertEqual(query_result.rewrite.original_query, original)
        self.assertEqual(query_result.rewrite.effective_query, effective)


if __name__ == "__main__":
    unittest.main()
