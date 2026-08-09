# =========================================================================
# Poraque User Guide
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
DOC := poraque_user_guide
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

# The version is read from the package, never retyped here. Both guides used
# to carry it as a literal on the title page, and both had drifted -- the user
# guide said 26.8.18 and the technical guide 26.6.4 while the package was on
# 26.8.27. A version a reader cannot trust is worse than none.
#
# `sed` rather than `python -c "import poraque"`: this must work from a clean
# checkout with nothing installed and no environment activated.
VERSION_SOURCE := ../../src/poraque/version.py
VERSION_STAMP := poraque_version.tex

.PHONY: all clean distclean watch
# Serial only: `clean` must not run before the PDF is produced.
.NOTPARALLEL:

all: $(DOC).pdf clean

# `-g` forces one full pass. Make is the arbiter of whether a rebuild is
# needed -- it is the one that knows about $(VERSION_STAMP) -- and latexmk must
# not overrule it. It otherwise would: `clean` deletes .fdb_latexmk, so on the
# next run latexmk has no dependency database, compares only against $(DOC).tex,
# announces "all targets are up-to-date" and leaves a stale version on the
# title page even though make invoked it precisely because the version changed.
$(DOC).pdf: $(DOC).tex $(VERSION_STAMP) $(wildcard sections/*.tex)
	$(LATEXMK) -g $(DOC).tex

# Regenerated whenever version.py changes, so the PDF is rebuilt with it.
$(VERSION_STAMP): $(VERSION_SOURCE)
	@v=$$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*["'\'']\([^"'\'']*\)["'\''].*/\1/p' $(VERSION_SOURCE)); \
	 test -n "$$v" || { echo "cannot read __version__ from $(VERSION_SOURCE)" >&2; exit 1; }; \
	 printf '%s\n' "% Generated from $(VERSION_SOURCE) by the Makefile. Do not edit." > $@; \
	 printf '%s\n' "\\newcommand{\\poraqueversion}{$$v}" >> $@

# -c removes auxiliaries but preserves the PDF; sections/ holds only sources,
# but latexmk can leave a stray .aux there when \input files are compiled.
clean:
	@latexmk -c $(DOC).tex >/dev/null 2>&1 || true
	@rm -f sections/*.aux
	@rm -f $(DOC).bbl $(DOC).blg $(DOC).synctex.gz
	@echo "cleaned: kept $(DOC).pdf, $(DOC).tex and sections/*.tex"

distclean:
	@latexmk -C $(DOC).tex >/dev/null 2>&1 || true
	@rm -f sections/*.aux $(VERSION_STAMP)
	@echo "removed auxiliaries and $(DOC).pdf"

watch:
	$(LATEXMK) -pvc $(DOC).tex
