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

The application retains the complete `ContractClause`, including
`block_indices`. Each finding also contains an exact `contract_excerpt`. The
backend verifies that the excerpt exists verbatim before exposing the finding.

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
  --max-output-tokens 2000 `
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