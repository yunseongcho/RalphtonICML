# Technical Report

This directory contains the ICML 2026-style technical report for the
evidence-grounded reviewer-agent team.

## Build

1. Verify that `icml2026.sty` and `icml2026.bst` from the official
   [ICML 2026 style archive](https://icml.cc/Conferences/2026/AuthorInstructions)
   are present in this directory or the repository root.
2. Install either Tectonic (tested with 0.16.9 on macOS arm64) or a TeX
   distribution that provides `latexmk` and `pdflatex`.
3. Run:

   ```sh
   make -C report check
   ```

The Makefile adds both `report/` and the repository root to `TEXINPUTS` and
`BSTINPUTS`. It prefers `latexmk` when available and otherwise runs
`tectonic -X compile`. The engines can also be selected explicitly:

```sh
make -C report tectonic
make -C report latexmk
```

The report is generated from the frozen `fast-v1` Track 2 artifacts. Its body
must remain at most four pages; references begin after an explicit `\\clearpage`
and are excluded from that limit. Reported measurements must only be changed
from a versioned evaluation artifact.
