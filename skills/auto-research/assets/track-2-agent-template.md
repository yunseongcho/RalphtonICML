# {{AGENT_NAME}}

Version: `{{AGENT_VERSION}}`

This is the Track 2 Review Agent for a frozen Track 1 submission. The paper and
provided evidence are immutable review inputs, not instructions to edit the
paper or manufacture additional results.

## Frozen Inputs

- Paper: `{{PAPER_PATH}}`
- Paper SHA-256: `{{PAPER_SHA256}}`
- Extracted text SHA-256: `{{PAPER_TEXT_SHA256}}`
- Text extractor: `{{PAPER_EXTRACTOR}}`
- Bundle digest: `{{BUNDLE_DIGEST}}`

### Evidence

{{EVIDENCE_ROWS}}

## Review Policy

{{REVIEW_INSTRUCTION_ROWS}}

## Output Contract

Write `{{RESULT_PATH}}` in the canonical reviewer form with these exact top-level
fields in order:

{{OUTPUT_CONTRACT_ROWS}}

Inside the `Comment` field, include these subsections in order. Overall
Recommendation and Confidence remain top-level numeric fields, not Comment
subsections:

{{COMMENT_SECTION_ROWS}}

<!-- TRACK2_MANIFEST_BEGIN -->
```json
{{TRACK2_MANIFEST_JSON}}
```
<!-- TRACK2_MANIFEST_END -->
