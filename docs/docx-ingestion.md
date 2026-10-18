# DOCX ingestion

`POST /documents/upload` accepts one `.docx` file of at most 5 MiB and returns
top-level paragraphs, headings, and tables in their original document order.
The size limit applies to the uploaded (compressed) DOCX file.

## Current limitations

- Tables are flattened into plain text. Row and cell coordinates are not retained.
- Only top-level body content is extracted. Headers, footers, comments, text boxes,
  footnotes, and endnotes are not included.
- Tracked changes and other complex Word structures are not interpreted.
- Heading classification uses Word paragraph styles whose names begin with
  `Heading`; it does not identify legal clauses.
- The generated document ID is a response identifier only. Files and results are
  not persisted.
- The endpoint does not perform LLM calls, legal analysis, or vector indexing.
