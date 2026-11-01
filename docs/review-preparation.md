# Whole-contract review preparation (PR1)

This service reads every parsed contract block, plans legal searches, and returns
an inspectable source registry. It does not issue legal findings or proposed
amendments. The review API, legal analysis, and LangGraph wiring belong to PR2.
Existing `/chat` and `/documents/upload` behavior is unchanged.

## Use

```python
from app.documents.models import ParsedDocument
from app.llm.factory import create_llm_gateway
from app.reviews.planner import ReviewPlanner
from app.reviews.retrieval import ReviewRetriever
from app.reviews.service import ReviewPreparationService


async def prepare_review(upload_response: dict) -> dict:
    document = ParsedDocument.model_validate(upload_response)
    gateway = create_llm_gateway()
    try:
        service = ReviewPreparationService(
            ReviewPlanner(gateway),
            ReviewRetriever(),
        )
        prepared = await service.prepare(document)
        return prepared.model_dump(mode="json")
    finally:
        await gateway.close()
```

This example makes paid LLM requests and queries the configured Qdrant instance.
Both dependencies can be replaced with fakes in tests.

## Planning and coverage

The entire document, including tables and original block indices, is sent to the
planner with a strict JSON schema. Required baseline topics are compensation,
working time, leave, and termination; the model adds contract-specific topics.
A missing clause is not automatically an unlawful clause. A baseline topic can
have no block references and still request relevant legal rules.

Every block must be referenced by a topic or explicitly classified as context
with a reason. Related blocks may appear under multiple topics. The backend
rejects omitted or unknown blocks, duplicate topic/query IDs, missing baseline
topics, and conflicting context dispositions. Structural coverage does not
establish that every legal issue has been identified.

Input is limited to 60,000 UTF-8 bytes of prompt content, including instructions;
oversized input is rejected without truncation or an LLM call. This is a
conservative application limit, not a provider token counter. Planning requests
override the chat output default with 8,000 output tokens. Non-normal finish
reasons and invalid JSON/schema/coverage raise `ReviewPlanningError`; gateway
errors propagate unchanged. No automatic paid retries are performed.

## Retrieval and identity

Each topic supplies neutral Thai legal questions, including conditions and
exceptions, rather than one embedding query for the entire contract. At most
32 queries run, with four concurrent calls and top-k five by default. Limits
are configurable. Searches do not impose an automatically guessed law filter.

`source_id` is `src_` plus SHA-256 of the UTF-8 JSON array containing exact
`law_name`, `section`, and `text`. Identical sources across searches share an ID;
different texts for the same provision remain separate. Source filenames are
provenance, not identity. Sources preserve their filenames and per-query scores.
The frontend should display `law_name` and `section`; future findings should
reference the registry ID rather than generate citation text.

Results preserve plan order even when retrieval completion order differs.
Each query reports `retrieved`, `no_results`, `partial` (some malformed hits),
or `failed`. Invalid law identity, empty text, and non-finite scores are rejected.
Retrieval exceptions are recorded without exposing upstream exception messages.
`ready` means every query returned valid hits; otherwise preparation is `partial`.
Neither status is a legal assessment or proof that the retrieved sources apply.

## Limits and next steps

- No documents, plans, or results are persisted by this service.
- DOCX ingestion still has the limitations described in `docx-ingestion.md`.
- Retrieval uses the existing synchronous adapter in worker threads. Concurrency
  is bounded per preparation call; cancellation does not terminate running threads.
- Corpus completeness, legal effective dates, retrieval relevance, and reference
  entailment require evaluation before exposing legal findings.
- PR2 must budget the combined contract and retrieved laws without silently
  dropping context, and connect analysis, evidence validation, and the API.

## Network-free checks

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_review_preparation -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
