# Whole-contract evidence review

This path is separate from the existing clause-by-clause baseline reviewer.

```text
DOCX -> existing parser -> existing segmentation -> existing retrieval
     -> global source deduplication -> one LLM request -> backend validation
```

`POST /reviews/whole-contract` returns validated reviews for every clause. The
request sent to the model contains every contract clause, every complete unique
legal source, and the clause-to-query-to-source mappings. Evidence is neither
summarized nor truncated by the review service.

The backend rejects the complete generated result when it has invalid JSON or
schema, duplicate or missing clause IDs, or invalid completeness coverage. The
per-request JSON schema restricts citations to the exact source IDs returned by
retrieval. If clause-level provenance still fails validation, only the affected
clause is marked `review_failed`; valid clause reviews and completeness remain
available, and the overall result is `partial`. Validation logs contain IDs and
locations but never contract or legal-source text.

The same LLM response also includes a `completeness` array for the eight fixed
employment-contract categories. Each item contains only `category`, the fixed
Thai `label`, `status`, and a Thai `summary`. The backend requires every category
exactly once in the defined order and accepts only `complete`, `partial`,
`missing`, `not_applicable`, or `uncertain` as status values. Completeness is
separate from the cited legal assessment: finding a clause does not establish
that the clause complies with law.

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
- `llm-response.json`: acceptance state, validation errors, normalized response
  fields, exact model content, and the full upstream provider JSON.

These artifacts contain contract text and legal analysis. Keep them under the
ignored local `tmp/` directory.
