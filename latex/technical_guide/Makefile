# =========================================================================
# Poraque Technical Guide
#
#   make          compile the PDF and remove every auxiliary file
#   make watch    recompile on change (auxiliaries kept while watching)
#   make clean    remove auxiliaries, keep the PDF
#   make distclean  remove auxiliaries AND the PDF
#
# `latexmk -c` deletes exactly the regenerable files -- .aux .log .out .toc
# .fls .fdb_latexmk and friends -- while keeping the .pdf. Listing extensions
# by hand would drift as packages add new ones, so the cleaning is delegated
# rather than reimplemented.
# =========================================================================
DOC := poraque_technical_guide
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean watch
# Serial only: `clean` must not run before the PDF is produced.
.NOTPARALLEL:

all: $(DOC).pdf clean

$(DOC).pdf: $(DOC).tex $(wildcard sections/*.tex)
	$(LATEXMK) $(DOC).tex

# -c removes auxiliaries but preserves the PDF; sections/ holds only sources,
# but latexmk can leave a stray .aux there when \input files are compiled.
clean:
	@latexmk -c $(DOC).tex >/dev/null 2>&1 || true
	@rm -f sections/*.aux
	@rm -f $(DOC).bbl $(DOC).blg $(DOC).synctex.gz
	@echo "cleaned: kept $(DOC).pdf, $(DOC).tex and sections/*.tex"

distclean:
	@latexmk -C $(DOC).tex >/dev/null 2>&1 || true
	@rm -f sections/*.aux
	@echo "removed auxiliaries and $(DOC).pdf"

watch:
	$(LATEXMK) -pvc $(DOC).tex
