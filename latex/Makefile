# =========================================================================
# Build both Poraque guides.
#
#   make            build both PDFs and remove every auxiliary file
#   make user       user guide only
#   make technical  technical guide only
#   make clean      remove auxiliaries, keep the PDFs
# =========================================================================
.PHONY: all user technical clean distclean
.NOTPARALLEL:

all: user technical

user:
	$(MAKE) -C user_guide

technical:
	$(MAKE) -C technical_guide

clean:
	@$(MAKE) -C user_guide clean
	@$(MAKE) -C technical_guide clean

distclean:
	@$(MAKE) -C user_guide distclean
	@$(MAKE) -C technical_guide distclean
