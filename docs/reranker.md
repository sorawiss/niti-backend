# Hybrid retrieval reranker

The optional second-stage reranker improves the ordering produced by Qdrant's
dense+sparse reciprocal-rank fusion (RRF). It does not replace BGE-M3
embeddings and does not require rebuilding the Qdrant collection.

## Request flow

```text
effective query
  -> BGE-M3 dense and sparse retrieval
  -> Qdrant RRF candidate set
  -> BAAI/bge-reranker-v2-m3
  -> final top_k results
```

Reranking is disabled by default. When disabled, Qdrant receives the requested
`top_k` and the behavior matches the retrieval baseline. When enabled, Qdrant
returns at least `RERANKER_CANDIDATE_LIMIT` fused candidates, the cross-encoder
scores every query-passage pair, and only the final `top_k` are returned.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RERANKER_ENABLED` | `false` | Enable second-stage reranking |
| `RERANKER_FAIL_OPEN` | `true` | Return RRF order if reranking fails |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | FlagEmbedding reranker model |
| `RERANKER_CANDIDATE_LIMIT` | `25` | Number of fused candidates to rerank |
| `RERANKER_BATCH_SIZE` | `8` | Query-passage scoring batch size |
| `RERANKER_QUERY_MAX_LENGTH` | `512` | Maximum query tokens |
| `RERANKER_MAX_LENGTH` | `1024` | Maximum combined pair tokens |
| `RERANKER_USE_FP16` | `false` | Use FP16 on compatible hardware |
| `RERANKER_CACHE_DIR` | unset | Optional Hugging Face cache directory |

The model loads lazily on the first enabled retrieval. A process-local lock
serializes scoring against the shared model instance so concurrent clause
retrieval threads do not invoke it simultaneously.

## Result scores

Retrieval results include both stages:

```json
{
  "score": 0.94,
  "retrieval_score": 0.0317,
  "rerank_score": 0.94,
  "retrieval_rank": 7,
  "rerank_rank": 1,
  "text": "...",
  "metadata": {}
}
```

`score` is always the score used for final ordering. With reranking enabled it
equals `rerank_score`; otherwise it equals `retrieval_score`, and the reranker
fields are null. Reranker scores are normalized to the range 0 through 1 for
observability, but should not be treated as calibrated legal confidence.

Chat citations expose the same fields. Clause-review source matches retain the
per-query retrieval and reranker ranks so deduplicated legal sources remain
traceable.

## Failure behavior

With the default fail-open setting, a model-loading or scoring failure is
logged without query or passage content and the request returns the original
RRF ordering. Set `RERANKER_FAIL_OPEN=false` only when callers should receive a
retrieval failure instead.

Before enabling reranking by default, compare RRF and reranked results on a
versioned Thai labor-law relevance set and track candidate recall, Recall@5,
MRR@5, nDCG@5, p95 latency, and peak process memory.
