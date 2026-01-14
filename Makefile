.PHONY: test graph

test:
	PYTHONPATH=. uv run --project . python -m pytest

graph:
	PYTHONPATH=. uv run --project . python scripts/render_graph.py
