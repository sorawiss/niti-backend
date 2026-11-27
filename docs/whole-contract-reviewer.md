# Whole-contract evidence review

This path is separate from the existing clause-by-clause baseline reviewer.

```text
DOCX -> existing parser -> existing segmentation -> existing retrieval
     -> LangGraph: plan searches -> retrieve and merge evidence -> review contract
     -> backend validation
```

`POST /reviews/whole-contract` returns validated reviews for every clause,
contract-level findings, and an overall summary. The
final reviewer receives every contract clause and the shared pool of complete
legal sources. Any source may support any clause. The planner receives the
clause-to-query-to-source mappings to see which searches have already run.

The backend rejects the complete generated result when it has invalid JSON or
schema, duplicate or missing clause IDs, unknown citation IDs, duplicate
assessment IDs within a clause, unknown supporting contract clause IDs, or an
assessment that does not include its own reviewed clause as support.

`contract_findings` is a free-form opinion in Thai, or `null` when there is no
useful opinion to add. It may use paragraphs or bullets to explain priorities,
practical concerns, and uncertainty. `overall_summary` is a short recap.
Clause-level assessments still use structured citations and clause references.

## Planner output budget

The planner uses `WHOLE_CONTRACT_PLANNER_MAX_OUTPUT_TOKENS` from the
environment or `.env`, with a default of `8192`. Restart the API or rerun
the script after changing it. The reviewer's output budget is configured
independently. In the trace, compare `context_planning.max_output_tokens`
with `context_planning.usage.completion_tokens` and `finish_reason` when
diagnosing a truncated response.

## Workflow trace

The endpoint uses two model requests. The first sees the contract and initial
retrieval, then proposes up to eight searches with `query`, `law_name`, and
`top_k` (1 to 5). Initial retrieval runs without LLM query rewriting on this
endpoint. The planner receives the unique `law_name` values from the active Qdrant
collection. If the catalog lookup fails, it uses the names from initial
retrieval. The search step drops unrecognized filters and runs the proposed
query without a law-name filter. If a filtered
search yields no usable evidence, the retriever repeats it without the law
filter. The second model request reviews the contract against the merged
evidence. A failed planner leaves initial evidence available for review.

Add `?debug=true` to inspect the initial retrieval, proposed searches and gaps,
per-search outcomes, source IDs sent to the reviewer, retrieval provenance,
and review validation. Merged mappings appear under `retrieval_provenance`:

```powershell
curl.exe -F "file=@docs/mock.docx" "http://localhost:8000/reviews/whole-contract?debug=true"
```

With `?debug=true`, the API also writes an INFO line to the server log
prefixed with `whole_contract_workflow_debug=`. Its JSON includes
`document_id` and the same `workflow_debug` object as the response.

The trace includes token counts and provider-reported cost under
`context_planning.usage` and `legal_review.usage`. For OpenRouter, cost
is reported in credits (1 credit equals 1 USD). `total_llm_cost` sums the
two reported costs and is `null` when either cost is unavailable.

The trace uses source IDs that resolve to the statute text in
`unique_legal_sources`. Use the trace with test contracts or in a trusted
environment because search text can include contract details.

## Complete LLM response for debugging

Raw provider output is deliberately not returned by the public HTTP endpoint.
Run the local script instead:

```powershell
.\.venv\Scripts\python.exe scripts\try_whole_contract_review.py docs\mock.docx
```

Use `--output-dir tmp\my-whole-review` to choose a fixed new directory. The
script writes:

- `document.json`: parsed contract blocks;
- `retrieval.json`: all per-clause retrieval results;
- `review.json`: validated assessments, unique sources, and evidence mappings;
- `workflow-debug.json`: step-by-step retrieval and review trace;
- `llm-response.json`: acceptance state, validation errors, normalized response
  fields, exact model content, and the full upstream provider JSON.

These artifacts contain contract text and legal analysis. Keep them under the
ignored local `tmp/` directory.
