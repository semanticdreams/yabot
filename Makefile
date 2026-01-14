.PHONY: test graph install

install:
	uv tool install --force --editable .

test:
	PYTHONPATH=. uv run --project . python -m pytest

render-graph:
	PYTHONPATH=. uv run --project . python scripts/render_graph.py
