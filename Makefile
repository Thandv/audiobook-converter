# Audiobook Converter — make targets.
#
# Most commands assume you're already inside a venv. Bootstrap one with:
#   python3 -m venv .venv && source .venv/bin/activate

.PHONY: help install install-training install-binary install-dev binary clean check sample

help:                ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:             ## Install the base package (Kokoro backend only).
	pip install -e .

install-training:    ## Install training extras (torch + coqui-tts).
	pip install -e ".[training]"

install-binary:      ## Install the deps needed to build the standalone binary.
	pip install -e ".[binary]"

install-dev:         ## Install everything (training + binary + tests).
	pip install -e ".[dev]"

binary: install-binary  ## Build a standalone `audiobook` executable into dist/.
	@which pyinstaller > /dev/null || (echo "pyinstaller not found; run `make install-binary` first" && exit 1)
	rm -rf build dist
	pyinstaller --clean --noconfirm audiobook.spec
	@echo ""
	@echo "  Binary: dist/audiobook"
	@du -sh dist/audiobook 2>/dev/null || true

pipx-install:        ## Globally install the `audiobook` command via pipx.
	pipx install --force --editable .

clean:               ## Remove build, dist, caches.
	rm -rf build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete

check:               ## Run ruff lint + a parser smoke test.
	@which ruff > /dev/null || (echo "ruff missing; run \`make install-dev\`" && exit 1)
	ruff check src/
	python -c "from audiobook.parser import parse_manuscript; print('parser ok')"

sample:              ## Render a quick sample (requires MANUSCRIPT=/path/to/file.md)
	@if [ -z "$(MANUSCRIPT)" ]; then echo "Usage: make sample MANUSCRIPT=/path/to/book.md"; exit 1; fi
	audiobook sample "$(MANUSCRIPT)" --mode single --paragraphs 6
