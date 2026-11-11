# LLM clause reviewer implementation work log

This log records the implementation reasoning and project commands used while
building the clause reviewer. File-inspection and file-editing commands are
intentionally omitted.

## Scope agreed before implementation

- Keep deterministic clause retrieval as the baseline.
- Review one clause with only its retrieved legal evidence.
- Request strict structured output through the existing provider-neutral LLM
  gateway.
- Reject invented source IDs and excerpts that are not present in the clause.
- Keep technical failure separate from legal-review statuses.
- Preserve clause and parser-block provenance for a future frontend.
- Leave AI query generation, missing-clause detection, cross-clause planning,
  and PDF ingestion outside this change.

## Design decisions

### Model output versus application output

The model can select four semantic statuses. `review_failed` is owned by the
application because it describes provider, parsing, or validation failure. The
model cites only source IDs. The application then hydrates trusted IDs from the
retrieval result, preventing generated law names or section numbers from
entering the public result.

### Failure isolation

Each clause is reviewed independently. One failed provider call or invalid
response produces `review_failed` for that clause while other clause reviews
continue. A clause with no retrieved source skips the LLM and becomes
`needs_more_evidence`.

### Frontend traceability

The application retains the complete `ContractClause`, including `clause_id`,
clause number, title, text, and `block_indices`. Assessments refer to that parent
clause instead of duplicating an exact contract excerpt.

## Chronological project command log

Commands used only to read or write repository files are not recorded here.
Validation commands and their results will be appended as implementation
proceeds. The real-provider DOCX command will be documented but left for the
developer to run, so its output and provider cost remain visible to them.

### 1. Initial focused reviewer test run

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
```

Result before correction: 7 tests ran; 6 passed and 1 errored.

Failing test:

```text
test_gateway_error_uses_stable_error_code ... ERROR
```

Relevant error:

```text
ValidationError: 1 validation error for ClauseReviewResult
usage
  Input should be a valid dictionary or instance of LLMUsage
  [input_value=None, input_type=NoneType]
```

Diagnosis: the provider-error path correctly caught `LLMGatewayError`, but it
passed `usage=None` into a result field whose contract always contains an
`LLMUsage` object. This caused the failure handler itself to fail and prevented
per-clause isolation.

Correction: use an empty `LLMUsage()` when no provider response exists. If a
response exists but fails later validation, retain the usage reported by that
response.

### 2. Focused reviewer test after correction

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
```

Result: all 7 tests passed in 0.088 seconds. This confirms valid citation
hydration, rejection of invented citations, verbatim-excerpt validation,
no-source behavior, per-clause invalid-JSON isolation, stable gateway error
codes, and status-specific payload validation.

### 3. Full project test suite

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: all 44 tests passed in 0.488 seconds.

One existing non-failing warning appeared:

```text
UserWarning: Payload indexes have no effect in the local Qdrant.
Please use server Qdrant if you need payload indexes.
```

The warning came from the existing local-Qdrant index test. The test completed
successfully and it is unrelated to clause-review behavior.

### 4. Python compilation

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
```

Result: exit code 0 with no output. All application, script, and test modules
compiled successfully.

### 5. Experiment CLI verification

Command:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py --help
```

Result: exit code 0. The command displayed the input DOCX argument and
`--output-dir`, `--top-k`, `--max-query-chars`,
`--retrieval-concurrency`, `--review-concurrency`, and
`--max-output-tokens` options. Help mode made no external request.

### 6. Whitespace validation

Command:

```powershell
git diff --check
```

Result: exit code 0. During final review, all implementation files were still
untracked, so this command did not inspect those new files. It only confirmed there
were no whitespace errors in tracked-file changes. The explicit scan below covers
the new files.

## Real-provider command intentionally left for the developer

This command was not run during implementation because it contacts the
configured embedding/Qdrant services and OpenRouter and may consume credits:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py docs\mock.docx
```

To preserve a predictable artifact path while debugging, use:

```powershell
.\.venv\Scripts\python.exe scripts\try_clause_review.py docs\mock.docx `
  --top-k 5 `
  --retrieval-concurrency 1 `
  --review-concurrency 1 `
  --max-output-tokens 4096 `
  --output-dir tmp\my-clause-review
```

The output directory must not already exist. Read `run.json` first, then
`retrieval.json`, and finally `review.json`. Share those artifacts or the
specific clause/error code when debugging retrieval relevance or model output.

### 7. Explicit whitespace scan for new files

Command:

~~~powershell
$targets = @(
  'app\reviews\review_models.py',
  'app\reviews\review_prompts.py',
  'app\reviews\reviewer.py',
  'scripts\try_clause_review.py',
  'tests\unit\test_clause_reviewer.py',
  'docs\clause-reviewer.md',
  'docs\llm-clause-reviewer-worklog.md'
)
rg -n '[ \t]+$' $targets
~~~

Result: no trailing whitespace was found. Ripgrep exits with code 1 when it
finds no match; the validation wrapper treated that expected no-match result as
success.
## Raw LLM diagnostic implementation

The developer requested exact model output and validation detail for failed clause
reviews. The implementation keeps trusted results in `review.json` and writes
developer-only traces to `llm-debug.json` under the ignored `tmp/` tree.

Each diagnostic record contains:

- `clause_id` and whether backend validation accepted the response;
- exact `raw_model_output` from `message.content`;
- sanitized Pydantic or provenance validation errors;
- `finish_reason`, request ID, model, provider, and usage; and
- a stable error code or gateway exception type when no model content exists.

The first patch mechanism failed because the Windows workspace helper could not
refresh. The executable fallback was also denied by WindowsApps. A Git patch
fallback then placed several zero-context insertions at shifted line numbers,
temporarily making the reviewer, script, and test files syntactically invalid.
This was detected during source inspection before tests. A bulk reverse was not
performed because automatic approval review rejected it as potentially
discarding user work. The affected blocks were repaired individually with
narrow patches, preserving the developer's 4,096-token script default.

### Compilation after repair

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
```

Result: exit code 0 with no output.

### Focused diagnostic tests

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
```

Result: all 8 tests passed in 0.100 seconds. The new tests prove that accepted
responses retain their exact content and metadata, malformed JSON retains its
exact raw string with a `json_invalid` error, and status/schema failures expose
their validation error without entering trusted findings.

### Full project suite

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: all 45 tests passed in 0.485 seconds. The existing local-Qdrant payload
index warning appeared and its test passed.

### Real-provider rerun

This command was intentionally left for the developer because it calls Qdrant,
loads the embedding model, calls OpenRouter, and may consume credits:

```powershell
python scripts\try_clause_review.py docs\mock.docx `
  --output-dir tmp\my-clause-review-debug
```

The output directory must not already exist. Inspect `llm-debug.json` for the


## Legal-assessment schema migration

The lawyer-facing result now retains material compliant and unresolved analysis,
in addition to potential violations. The model returns `applicable`,
`applicability_reason`, and per-issue `assessments`. Each assessment owns an
`outcome`, citations, an optional evidence gap, and an optional suggested
revision. The backend derives clause status with violation taking priority over
an evidence gap, and an evidence gap taking priority over an all-compliant
result.

Retrieval behavior did not change. `top_k` remains per query,
`retrieval.json` preserves every candidate source, and the reviewed result
hydrates only citation IDs selected by the model.

### Implementation-tool failures observed

The first `apply_patch` attempt did not modify a file. The Windows sandbox
helper failed during setup refresh with `helper_unknown_error`. The fallback
writer's first two in-process encoding attempts also stopped before mutation
because that JavaScript runtime did not expose `TextEncoder` or `btoa`.
A small explicit base64 encoder was then used with PowerShell's
`WriteAllBytes`.

The first documentation fallback encoded a Thai JSON example with an
ASCII-only helper. This was recognized immediately and the complete file was
rewritten with an explicit UTF-8 encoder before validation.

The first JSON-schema inspection printed the top-level fields and then failed
with `KeyError: ''` because PowerShell expanded `$defs` inside the Python
command. Escaping the dollar sign fixed the inspection; no application code
changed for this command issue.

### Focused reviewer tests after migration

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_clause_reviewer -v
```

Result: all 12 tests passed in 0.147 seconds. Coverage includes retained
compliant assessments, mixed compliant/evidence-gap results, violation
priority, not-applicable output, citation hydration, provenance rejection,
no-source behavior, diagnostics, and provider-failure isolation.

### Full project suite after migration

Command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: all 49 tests passed in 0.557 seconds. The existing local-Qdrant payload
index warning appeared and its test passed.

### Compilation after migration

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
```

Result: exit code 0 with no output.

### Generated-schema inspection

Command:

```powershell
.\.venv\Scripts\python.exe -c "from app.reviews.review_models import GeneratedClauseReview; s=GeneratedClauseReview.model_json_schema(); a=s['`$defs']['GeneratedLegalAssessment']; print(sorted(s['properties'])); print(s['required']); print(sorted(a['properties'])); print(a['required'])"
```

Result: the generated model schema requires `clause_id`, `applicable`,
`applicability_reason`, and `assessments`. Every assessment requires
`assessment_id`, `outcome`, `issue`, `reasoning`,
`citations`, `evidence_gap`, `suggested_revision`, and `confidence`.
Nullable fields remain required so strict structured output must state
`null` explicitly.

### Git whitespace check

Command:

```powershell
git diff --check
```

Result: exit code 0. Git printed line-ending notices that LF will become CRLF
when it next writes the changed files; it found no whitespace error.

### Real-provider migration check left for the developer

This command was not run because it calls embedding, Qdrant, and LLM services
and can consume credits:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\try_clause_review.py docs\mock.docx `
  --top-k 5 `
  --retrieval-concurrency 1 `
  --review-concurrency 1 `
  --max-output-tokens 4096 `
  --output-dir tmp\assessment-review
```
