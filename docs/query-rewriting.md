# LLM query rewriting

The retrieval workflows can optionally rewrite each query with
`deepseek/deepseek-v4.1-flash` through the existing OpenRouter gateway. Query
rewriting is disabled by default and is controlled independently for each API.

## Configuration

```env
QUERY_REWRITE_RAG_ENABLED=false
QUERY_REWRITE_CHAT_ENABLED=false
QUERY_REWRITE_CONTRACT_REVIEW_ENABLED=false
QUERY_REWRITE_WHOLE_CONTRACT_ENABLED=false

QUERY_REWRITE_MODEL=deepseek/deepseek-v4.1-flash
QUERY_REWRITE_MAX_OUTPUT_TOKENS=128
QUERY_REWRITE_TIMEOUT_SECONDS=8
QUERY_REWRITE_MAX_CONCURRENCY=4
QUERY_REWRITE_MAX_QUERY_CHARS=1000
QUERY_REWRITE_LOG_RAW_RESPONSE=false
```

The workflow flags map to these endpoints:

| Variable | Endpoint |
| --- | --- |
| `QUERY_REWRITE_RAG_ENABLED` | `POST /rag/retrieve` |
| `QUERY_REWRITE_CHAT_ENABLED` | `POST /chat` |
| `QUERY_REWRITE_CONTRACT_REVIEW_ENABLED` | `POST /reviews` |
| `QUERY_REWRITE_WHOLE_CONTRACT_ENABLED` | Direct calls using the whole-contract rewrite workflow. The [whole-contract endpoint](whole-contract-reviewer.md) uses its LangGraph planner for supplemental searches. |

Changing a variable requires restarting the application. Clients cannot
override these policies with request parameters.

## Behavior

The rewriter produces one query for the BGE-M3 dense and sparse retrieval
pipeline. Chat answer generation continues to receive the original user
question. Contract review retains each original `ClauseQuery` and attaches the
rewrite result to its corresponding `QueryRetrievalResult`.

Responses use one of these rewrite statuses:

- `disabled`: the workflow flag is off and OpenRouter is not called.
- `rewritten`: retrieval used the validated rewritten query.
- `fallback_original`: rewriting failed and retrieval continued with the
  original query.

Timeouts, provider failures, invalid JSON, blank output, and output exceeding
`QUERY_REWRITE_MAX_QUERY_CHARS` fall back without retrying. Every attempted LLM
call emits one structured `query_rewrite` log after completion, including the
workflow, status, latency, OpenRouter request ID, provider, model, finish reason,
detailed token/cost usage, provider metadata, and sanitized error code. Disabled
workflows emit a separate record with `llm_called=false`.

For local debugging, set `QUERY_REWRITE_LOG_RAW_RESPONSE=true` and restart the
application. Each attempted rewrite log will then contain `raw_response`: the
parsed OpenRouter response object on success, or the exact upstream response
text when gateway parsing fails. Keep this disabled in production because raw
responses can contain user queries, generated content, provider diagnostics,
and other sensitive data.

## Response metadata

`POST /chat` and `POST /rag/retrieve` include a top-level `rewrite` object.
Contract-review responses include it under
`clause_reviews[].query_results[].rewrite`. The object contains the original
and effective queries, model/provider identity, token usage, cost, and a
sanitized error code.

## Retrieval evaluation

The evaluation harness uses the configured OpenRouter and Qdrant services, so it is
not part of the network-free unit test suite:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_query_rewriting
```

It compares baseline and rewritten Recall@5, rejects overall regression, and
requires at least 5% relative improvement for the ambiguous/colloquial subset.
