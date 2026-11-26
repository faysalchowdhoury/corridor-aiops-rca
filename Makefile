.PHONY: install data test lint benchmark figures paper

install:
	python -m pip install -e ".[dev,figures]"

data:
	corridor-ingest

test:
	pytest

lint:
	ruff check .

benchmark:
	corridor-benchmark --seeds 20

figures:
	corridor-figures

paper: benchmark figures
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
