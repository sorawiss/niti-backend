# Evidence-grounded clause reviewer

The pipeline is:

```text
DOCX -> blocks -> clauses -> hybrid retrieval -> LLM assessments -> validated review
```

It reviews clauses present in the contract. Missing-clause detection and
whole-contract certification remain outside this phase.

## Review model

The model receives one clause and every unique source retrieved for it. It
returns `applicable`, a Thai `applicability_reason`, and only material
`assessments`. It does not create one assessment per retrieved source.

| Assessment outcome | Meaning |
| --- | --- |
| `compliant` | The excerpt satisfies the cited requirement. |
| `potential_violation` | It may conflict with or provide less than the cited requirement. |
| `needs_more_evidence` | Available law or facts cannot resolve the issue. |

Compliant and unresolved assessments remain visible for lawyer review. The
backend derives the clause status in this order:

1. `not_applicable` when `applicable=false`;
2. `potential_violation` when any assessment has that outcome;
3. `needs_more_evidence` when no violation exists but an assessment needs
   more evidence;
4. `no_issue_found` when all assessments are compliant.

`review_failed` is application-owned.

## Model-output example

```json
{
  "clause_id": "clause-0005",
  "applicable": true,
  "applicability_reason": "ข้อสัญญากำหนดเวลาทำงานและเวลาพัก",
  "assessments": [
    {
      "assessment_id": "assessment-01",
      "outcome": "compliant",
      "issue": "เวลาทำงานปกติ",
      "reasoning": "ข้อสัญญากำหนดเวลาไม่เกินหลักฐานกฎหมายที่อ้าง",
      "citations": [{"source_id": "law:1f16784831daf20f"}],
      "evidence_gap": null,
      "suggested_revision": null,
      "confidence": "high"
    }
  ]
}
```

## Validation boundary

Before exposing an assessment, the backend requires a matching `clause_id`,
retrieved citation IDs, unique assessment IDs, valid outcome-specific fields,
and consistent applicability. Any failure makes
the complete clause response `review_failed`. Accepted source IDs are hydrated
with law name, section, text, and source-file provenance.

When retrieval has no source, the backend skips the model and returns
`needs_more_evidence`.

## Retrieval and review artifacts

`--top-k 5` applies per query. Long clauses can produce multiple queries, so
a clause may have more than five unique sources. The model sees the entire
deduplicated list and selects citations material to each assessment.

- `retrieval.json`: every retrieved source, including unused candidates.
- `review.json`: validated assessments and their cited sources.
- `llm-debug.json`: exact raw model output and validation diagnostics.
- `run.json`: retrieval and review summaries.

## Run with a real DOCX

Activation is supported:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\try_clause_review.py docs\mock.docx
```

The equivalent command without activation is:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py docs\mock.docx
```

For a fixed output directory:

```powershell
python scripts\try_clause_review.py docs\mock.docx `
  --top-k 5 `
  --retrieval-concurrency 1 `
  --review-concurrency 1 `
  --max-output-tokens 4096 `
  --output-dir tmp\my-assessment-review
```

The directory must not already exist. This real run calls embedding, Qdrant,
and LLM services and can consume provider credits.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
git diff --check
```

## Debugging order

1. Read `run.json` for the run status and counts.
2. Inspect all candidate sources for the clause in `retrieval.json`.
3. Inspect derived status and accepted assessments in `review.json`.
4. Find the same `clause_id` in `llm-debug.json`.
5. Compare `raw_model_output`, `validation_errors`, `finish_reason`, and
   `accepted`.
6. For `unknown_citation`, compare the citation with retrieved source IDs.

`llm-debug.json` can contain contract text and legal analysis. Keep it under
the ignored `tmp/` tree and do not expose it through a public API.
