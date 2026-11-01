from app.documents.models import ParsedDocument
from app.reviews.models import ReviewPreparation
from app.reviews.planner import ReviewPlanner
from app.reviews.retrieval import ReviewRetriever


class ReviewPreparationService:
    def __init__(self, planner: ReviewPlanner, retriever: ReviewRetriever) -> None:
        self._planner = planner
        self._retriever = retriever

    async def prepare(self, document: ParsedDocument) -> ReviewPreparation:
        plan, response = await self._planner.plan(document)
        retrieval = await self._retriever.retrieve(plan)
        return ReviewPreparation(
            document_id=document.document_id,
            filename=document.filename,
            status=(
                "ready"
                if all(query.status == "retrieved" for query in retrieval.queries)
                else "partial"
            ),
            plan=plan,
            retrieval=retrieval,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        )
