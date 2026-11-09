# Qdrant hybrid search

The active RAG retriever uses BGE-M3 dense and lexical embeddings stored in a
separate Qdrant collection named `thai_labor_law_hybrid`. The legacy
`thai_labor_law` collection is not modified and remains available for tests.

## Build the hybrid collection

Install dependencies, configure `QDRANT_URL` and `QDRANT_API_KEY`, then run:

```powershell
.\.venv\Scripts\python.exe -m app.rag.qdrant_index
```

Indexing uses deterministic point IDs, so rerunning the command upserts the
same chunks. To deliberately clear and rebuild only the hybrid collection:

```powershell
.\.venv\Scripts\python.exe -m app.rag.qdrant_index --recreate
```

The command refuses to target the legacy collection.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `QDRANT_COLLECTION_NAME` | `thai_labor_law_hybrid` | Active hybrid collection |
| `BGE_M3_BATCH_SIZE` | `4` | BGE-M3 inference batch size |
| `BGE_M3_MAX_LENGTH` | `8192` | Maximum model input tokens |
| `BGE_M3_USE_FP16` | `false` | Enable FP16 inference on compatible GPUs |
| `HYBRID_PREFETCH_LIMIT` | `50` | Dense and sparse candidates sent to RRF |
| `HYBRID_RRF_K` | `60` | Reciprocal Rank Fusion constant |

At query time, Qdrant retrieves candidates from the named `dense` and `sparse`
vectors, applies native RRF, and returns the existing `score`, `text`, and
`metadata` result contract used by chat and clause review.
