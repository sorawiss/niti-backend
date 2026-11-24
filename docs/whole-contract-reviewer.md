# Whole-contract evidence review

This path is separate from the existing clause-by-clause baseline reviewer.

```text
DOCX -> existing parser -> existing segmentation -> existing retrieval
     -> global source deduplication -> one LLM request -> backend validation
```

`POST /reviews/whole-contract` returns validated reviews for every clause,
contract-level findings, and an overall summary. The
request sent to the model contains every contract clause, every complete unique
legal source, and the clause-to-query-to-source mappings. Evidence is neither
summarized nor truncated by the review service.

The backend rejects the complete generated result when it has invalid JSON or
schema, duplicate or missing clause IDs, unknown citation IDs, duplicate
assessment IDs within a clause, unknown supporting contract clause IDs, or an
assessment that does not include its own reviewed clause as support.

`contract_findings` is reserved for material issues that cannot honestly be
assigned to a single clause: a missing term, a conflict between clauses, a
contract-wide risk, or a drafting issue spanning clauses. Missing terms must
not claim clause support. Every other finding must identify at least one real
contract clause. Legal citations are checked against the same retrieved source
set used for clause reviews. `overall_summary` may summarize accepted clause
reviews and contract findings but must not introduce a new legal conclusion.

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
