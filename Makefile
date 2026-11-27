PYTHON ?= python

.PHONY: install data audit test lint benchmark figures paper docx reproduce

install:
	$(PYTHON) -m pip install -e ".[dev,figures,torch]"

data:
	corridor-ingest

audit:
	corridor-audit

test:
	pytest

lint:
	ruff check .

benchmark:
	corridor-benchmark --train-seeds 5 --bootstrap-samples 1000 --bootstrap-preview-rows 20 --epochs 5

figures:
	corridor-figures

reproduce:
	corridor-reproduce

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

docx:
	$(PYTHON) paper/build_docx.py
