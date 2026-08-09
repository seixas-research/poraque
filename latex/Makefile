# =========================================================================
# Build the Poraque documents.
#
#   make            build both guide PDFs and remove every auxiliary file
#   make user       user guide only
#   make technical  technical guide only
#   make article    the manuscript only
#   make clean      remove auxiliaries, keep the PDFs
#
# `all` deliberately excludes the manuscript. The guides document the
# released software and are rebuilt whenever it changes; a manuscript is a
# draft on its own schedule, and rebuilding it as a side effect of a release
# would churn a file nobody asked for.
# =========================================================================
.PHONY: all user technical article clean distclean
.NOTPARALLEL:

all: user technical

user:
	$(MAKE) -C user_guide

technical:
	$(MAKE) -C technical_guide

article:
	$(MAKE) -C article

clean:
	@$(MAKE) -C user_guide clean
	@$(MAKE) -C technical_guide clean
	@$(MAKE) -C article clean

distclean:
	@$(MAKE) -C user_guide distclean
	@$(MAKE) -C technical_guide distclean
	@$(MAKE) -C article distclean
