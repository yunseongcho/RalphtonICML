LATEXMK ?= latexmk
TECTONIC ?= tectonic
TEXINPUTS := .:..:
BSTINPUTS := .:..:

.PHONY: all check clean latexmk tectonic

all: report.pdf

report.pdf: report.tex references.bib icml2026.sty icml2026.bst
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(MAKE) latexmk; \
	elif command -v $(TECTONIC) >/dev/null 2>&1; then \
		$(MAKE) tectonic; \
	else \
		echo "Neither latexmk nor tectonic is installed"; exit 1; \
	fi

latexmk:
	TEXINPUTS="$(TEXINPUTS)" BSTINPUTS="$(BSTINPUTS)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error report.tex

tectonic:
	TEXINPUTS="$(TEXINPUTS)" BSTINPUTS="$(BSTINPUTS)" $(TECTONIC) -X compile --keep-logs report.tex

check:
	@test -f icml2026.sty -o -f ../icml2026.sty || (echo "Missing icml2026.sty in report/ or repository root" && exit 1)
	@test -f icml2026.bst -o -f ../icml2026.bst || (echo "Missing icml2026.bst in report/ or repository root" && exit 1)
	$(MAKE) report.pdf

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C report.tex; \
	else \
		$(RM) report.aux report.bbl report.blg report.log report.out report.pdf report.synctex.gz; \
	fi
