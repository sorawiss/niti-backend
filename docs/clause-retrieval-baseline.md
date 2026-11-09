# Clause retrieval baseline

This package prepares clause-level legal evidence from a parsed DOCX. It does
not decide whether a contract is compliant, identify a violation, or suggest a
revision.

## Pipeline

1. The existing DOCX parser returns ordered `DocumentBlock` objects.
2. The clause segmenter uses headings and Thai or Arabic clause numbering.
3. A coverage validator requires every non-empty block exactly once.
4. The query builder creates at least one deterministic query per clause.
5. `ClauseRetriever` invokes the existing `retrieve_laws()` function.
6. Results retain the clause ID, block indices, query provenance, law name, and
   section.

If a document has no detectable headings or clause numbers, each parser block
becomes a clause. This is a conservative fallback that preserves coverage and
makes the limitation visible in `boundary_reason`.

Headings immediately followed by numbered provisions are attached as context
to the first provision. Dotted clause numbers such as `1.2` and `๑.๒` are
linked to an earlier parent clause when that parent exists.

## Run the local experiment

From the backend directory:

```powershell
.venv\Scripts\python.exe scripts\try_clause_retrieval.py path\to\contract.docx
```

Optional arguments:

```text
--top-k 5
--max-query-chars 4000
--max-concurrency 1
--output-dir tmp\my-clause-run
```

The default output is a new, unique directory under
`tmp/clause-retrieval/`. The script refuses to overwrite an existing output
directory and writes:

- `document.json`: parser blocks
- `clauses.json`: deterministic clause boundaries
- `queries.json`: exact retrieval queries
- `result.json`: per-clause legal sources and retrieval status
- `run.json`: input, output, status, and summary counts

Exit codes are `0` for a ready run, `2` for a partial run, and `1` for a
failed run or invalid input.

`ready` only means that every query returned at least one structurally valid
source. It is not a statement that the sources are legally sufficient or that
the contract complies with the law.

## Extension boundary

The retriever is injected behind a callable that accepts `query`, `top_k`, and
`law_name`. Dense search can therefore be replaced by hybrid search and a
reranker without changing segmentation or result provenance.

An adaptive clause agent should be added only after this baseline is evaluated.
It can generate additional queries, request related clauses, and retry weak
retrieval. The raw deterministic query should remain available as a coverage
baseline.

