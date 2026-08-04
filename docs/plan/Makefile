# =========================================================================
# Poraque Future Roadmap
#
#   make          compile the PDF and remove every auxiliary file
#   make watch    recompile on change (auxiliaries kept while watching)
#   make clean    remove auxiliaries, keep the PDF
#   make distclean  remove auxiliaries AND the PDF
#
# Mirrors latex/user_guide/Makefile: `latexmk -c` deletes exactly the
# regenerable files while keeping the .pdf, so the extension list does not have
# to be maintained here as packages add new ones.
# =========================================================================
DOC := future_roadmap
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean watch
# Serial only: `clean` must not run before the PDF is produced.
.NOTPARALLEL:

all: $(DOC).pdf clean

$(DOC).pdf: $(DOC).tex
	$(LATEXMK) $(DOC).tex

clean:
	@latexmk -c $(DOC).tex >/dev/null 2>&1 || true
	@rm -f $(DOC).bbl $(DOC).blg $(DOC).synctex.gz
	@echo "cleaned: kept $(DOC).pdf and $(DOC).tex"

distclean:
	@latexmk -C $(DOC).tex >/dev/null 2>&1 || true
	@echo "removed auxiliaries and $(DOC).pdf"

watch:
	$(LATEXMK) -pvc $(DOC).tex
