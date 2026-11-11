# Evidence-grounded clause reviewer

This phase adds legal review after the deterministic clause retrieval baseline:

```text
DOCX -> blocks -> clauses -> hybrid retrieval -> LLM review -> validated findings
```

It reviews clauses that exist in the contract. It does not yet detect a clause
that is entirely missing or certify that the whole contract complies with every
applicable law.

## Trust boundary

The model receives one clause and the legal sources retrieved for that clause.
It may return `no_issue_found`, `potential_violation`,
`needs_more_evidence`, or `not_applicable`.

The application creates `review_failed`; the model cannot select it. Before a
finding is exposed, the application requires:

- the returned `clause_id` to match the requested clause;
- every `contract_excerpt` to occur verbatim in the clause;
- every citation to reference a retrieved `source_id`;
- unique finding IDs; and
- a payload that satisfies the status-specific schema rules.

If any check fails, the complete clause response becomes `review_failed`. No
untrusted partial finding is exposed. The application hydrates accepted source
IDs with `law_name`, `section`, legal text, and source-file provenance.

When retrieval returns no sources, the application skips the model and returns
`needs_more_evidence`.

## Status meaning

| Status | Meaning |
| --- | --- |
| `potential_violation` | At least one evidence-supported issue was found. |
| `no_issue_found` | No issue was found from the supplied evidence; this is not a whole-contract compliance claim. |
| `needs_more_evidence` | Retrieval evidence is missing, irrelevant, or insufficient. |
| `not_applicable` | The unit contains no reviewable contract substance. |
| `review_failed` | The provider call or application validation failed. |

## Run with a real DOCX

Configure an ignored `.env` with at least:

```env
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openai/gpt-5-mini
```

The vector database and hybrid retrieval configuration must also be available.
From the backend directory, run:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py docs\mock.docx
```

For a specific output directory and conservative sequential execution:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py docs\mock.docx `
  --top-k 5 `
  --retrieval-concurrency 1 `
  --review-concurrency 1 `
  --max-output-tokens 2000 `
  --output-dir tmp\my-clause-review
```

The script uses a unique directory under `tmp/clause-review/` by default and
writes:

- `document.json`: parser blocks;
- `retrieval.json`: clauses, queries, and retrieved sources;
- `review.json`: validated clause reviews and hydrated citations; and
- `run.json`: input path and retrieval/review summaries.

Exit codes are `0` for ready, `2` for partial, and `1` for failed or invalid
input. A ready run can still contain `needs_more_evidence`; ready means the
workflow completed without a technical or validation failure.

The command calls the configured embedding service, Qdrant, and LLM provider.
It can consume provider credits. Unit tests do not make network calls.

## Run the automated checks

Focused reviewer tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
```

Full suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Syntax compilation:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
```

## Debugging order

1. Read `run.json` to determine whether the run was ready, partial, or failed.
2. Inspect the affected clause in `retrieval.json` and verify that its sources
   are relevant and sufficient.
3. Inspect `error_code` in `review.json`.
4. For `unknown_citation`, compare the model citation with retrieved source IDs.
5. For `contract_excerpt_mismatch`, compare the returned excerpt byte-for-byte
   with `clause.text`.
6. For `invalid_model_output`, reproduce with the recorded clause and sources;
   the application intentionally does not expose malformed output as a finding.

The current result records model identity, provider, and token/cost usage that
the provider reports. It does not persist the raw malformed LLM response because
contract text can be sensitive; add an explicitly protected diagnostic sink if
raw-response retention becomes necessary.
