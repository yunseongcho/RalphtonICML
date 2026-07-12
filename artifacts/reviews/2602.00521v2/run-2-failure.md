# Run 2 Failure Record

- Paper: `arxiv:2602.00521v2`
- Upstream hosted calls cached: 28
- Failed stage: initial chair review
- Chair transport attempts: 2
- Adapter error: `expected '#### **Soundness**' at line 1, got '## Soundness'`
- Root cause: both responses used the correct semantic field label but changed the Markdown heading level and omitted bold markup.

The adapter now accepts only exact reviewer field labels in the required order,
with any standard Markdown heading level and optional bold markup, then renders
the core canonical headings. Bare integer scores, score ranges, field order, and
the non-empty Comment remain strict. The complete suite passes 50 tests.

The 28 successful upstream responses remain in `backend_cache_v1`; the next run
will replay them locally and call the hosted model only for uncached chair/author
stages.
