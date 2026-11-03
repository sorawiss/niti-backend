# Whole-contract review preparation (PR1)

This service reads every parsed contract block, plans legal searches, and returns
an inspectable source registry. It does not issue legal findings or proposed
amendments. The review API, legal analysis, and LangGraph wiring belong to PR2.
Existing `/chat` and `/documents/upload` behavior is unchanged.

## Run a DOCX and inspect each stage

From the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.try_review_preparation
```

This uses `docs/Thai_Employment_Contract_Mock.docx` by default. A different input
and a new output directory can be supplied explicitly:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.try_review_preparation "C:\contracts\example.docx" --output-dir tmp/example-run
```

The command uses live LLM and existing RAG services configured by the project's
`.env`. It makes one planning request with the current prompt and validation
limits, and does not retry. Retrieval runs sequentially to avoid concurrent cold
loading of the embedding model. `--top-k` defaults to five.

Each default run gets a unique folder under `tmp/review-preparation/` (the folder
name uses UTC). Explicit output directories must not exist; previous results are
never overwritten. Files are written as UTF-8:

| File | Contents |
| --- | --- |
| `document.json` | Parsed contract blocks, saved before the planning call |
| `planner-response.json` | Raw model output plus model, finish reason, and usage |
| `plan.json` | Accepted plan, saved before retrieval |
| `rejected-plan.json` | Schema-valid plan that failed workflow validation |
| `result.json` | Full preparation result, including partial retrieval results |
| `run.json` | Completion summary or failure stage and diagnostic details |

For a rejected plan, `run.json` includes the rejection reason, query count when
the plan passed schema validation, and configured query limit. Malformed JSON,
schema failures, and abnormal finish reasons retain the raw response; their
query count is `null`, because no valid plan could be constructed. Such output
is diagnostic evidence only and is never passed to retrieval. Gateway errors
retain error codes, not upstream exception messages or credentials. Diagnostics
can contain the contract and model-generated text and remain local under the
git-ignored `tmp/` directory by default.

Exit codes: `0` = preparation ready; `2` = partial retrieval result saved;
`1` = failure. A `ready` result does not establish legal relevance or compliance.

`ReviewPlanningError` now carries the rejected `plan` (when schema-valid), raw
`response`, `max_queries`, and a computed `query_count`. This allows this explicit
diagnostic script to save failures without adding logging or file writes to the
planner itself. Existing prompts, validation rules, and API behavior are unchanged.

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
