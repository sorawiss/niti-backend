from pydantic import ValidationError

from app.documents.models import ParsedDocument
from app.llm.base import LLMGateway
from app.llm.models import LLMRequest, LLMResponse
from app.reviews.models import ReviewPlan
from app.reviews.prompts import BASELINE_TOPICS, build_plan_messages


class ReviewInputError(ValueError):
    """The document cannot be planned without omission or ambiguous references."""


class ReviewPlanningError(Exception):
    """The model response does not constitute a complete, valid review plan."""

    def __init__(
        self,
        message: str,
        *,
        response: LLMResponse | None = None,
        plan: ReviewPlan | None = None,
        max_queries: int | None = None,
    ) -> None:
        super().__init__(message)
        # Available to an explicit local diagnostic tool; never automatically logged.
        self.response = response
        self.plan = plan
        self.max_queries = max_queries

    @property
    def query_count(self) -> int | None:
        if self.plan is None:
            return None
        return sum(len(topic.queries) for topic in self.plan.topics)


def validate_plan(
    document: ParsedDocument, plan: ReviewPlan, *, max_queries: int = 32
) -> None:
    known = {block.index for block in document.blocks}
    topic_ids = [topic.topic_id for topic in plan.topics]
    if len(topic_ids) != len(set(topic_ids)):
        raise ReviewPlanningError("Duplicate topic IDs")
    if not BASELINE_TOPICS.keys() <= set(topic_ids):
        raise ReviewPlanningError("Missing baseline review topics")
    queries = [query for topic in plan.topics for query in topic.queries]
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ReviewPlanningError("Duplicate query IDs")
    if len(queries) > max_queries:
        raise ReviewPlanningError("Review query limit exceeded")

    reviewed = {index for topic in plan.topics for index in topic.block_indices}
    context = [item.block_index for item in plan.context_blocks]
    if len(context) != len(set(context)) or reviewed.intersection(context):
        raise ReviewPlanningError("Conflicting or duplicate context dispositions")
    referenced = reviewed | set(context)
    if referenced - known:
        raise ReviewPlanningError("Plan references unknown document blocks")
    if known - referenced:
        raise ReviewPlanningError("Plan leaves document blocks unaccounted for")


class ReviewPlanner:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        max_input_bytes: int = 60_000,
        max_output_tokens: int = 8_000,
        max_queries: int = 32,
    ) -> None:
        if min(max_input_bytes, max_output_tokens, max_queries) < 1:
            raise ValueError("Planner limits must be positive")
        self._gateway = gateway
        self._max_input_bytes = max_input_bytes
        self._max_output_tokens = max_output_tokens
        self._max_queries = max_queries

    async def plan(self, document: ParsedDocument) -> tuple[ReviewPlan, LLMResponse]:
        indices = [block.index for block in document.blocks]
        if not indices or any(index < 0 for index in indices):
            raise ReviewInputError("Document must contain blocks with nonnegative indices")
        if len(indices) != len(set(indices)):
            raise ReviewInputError("Document block indices must be unique")
        messages = build_plan_messages(document)
        messages[0].content += f"\nสร้างคำค้นรวมไม่เกิน {self._max_queries} คำค้น"
        input_bytes = sum(len(message.content.encode("utf-8")) for message in messages)
        if input_bytes > self._max_input_bytes:
            raise ReviewInputError("Document exceeds the planning input budget")
        response = await self._gateway.generate(
            LLMRequest(
                messages=messages,
                max_output_tokens=self._max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "contract_review_plan",
                        "strict": True,
                        "schema": ReviewPlan.model_json_schema(),
                    },
                },
            )
        )
        plan: ReviewPlan | None = None
        try:
            if response.finish_reason not in (None, "stop"):
                raise ReviewPlanningError("Planner response did not finish normally")
            plan = ReviewPlan.model_validate_json(response.content)
            validate_plan(document, plan, max_queries=self._max_queries)
        except ValidationError as error:
            raise ReviewPlanningError(
                "Invalid review plan JSON or schema",
                response=response,
                max_queries=self._max_queries,
            ) from error
        except ReviewPlanningError as error:
            raise ReviewPlanningError(
                str(error), response=response, plan=plan, max_queries=self._max_queries
            ) from error
        return plan, response
