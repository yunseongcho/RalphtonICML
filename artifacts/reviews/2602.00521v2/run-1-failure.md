# Run 1 Failure Record

- Paper: `arxiv:2602.00521v2`
- Hosted backend calls: 30 successful, 0 transport failures
- Completed stages: 19 extraction, 2 domain review, 6 criterion review, synthesis
- Failed stage: initial chair review after 2 model responses
- Parser error: `expected '' at line 2, got '2'`
- Root cause: the hosted response used the correct bare integer score but omitted a Markdown blank line required by the core canonical parser.
- Output status: `review_run.json` was not written because the pipeline writes it only after full success.

The rerun keeps the core review schema and learning-state manifest unchanged.
The hosted adapter now canonicalizes blank-line whitespace while still requiring
the exact seven headings, order, bare ASCII integer scores, valid ranges, and a
non-empty Comment. A fingerprinted local response cache was also enabled so a
late-stage failure can reuse completed model responses.
