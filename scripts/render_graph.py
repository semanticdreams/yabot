#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from yabot.graph import YabotGraph
from yabot.llm import LLMClient
from yabot.skills import load_skills


def _build_graph():
    api_key = os.environ.get("OPENAI_API_KEY") or "sk-local"
    llm = LLMClient(api_key=api_key)
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    skills = load_skills([skills_dir])
    return YabotGraph(
        llm=llm,
        default_model="gpt-4o-mini",
        available_models=["gpt-4o-mini"],
        max_turns=1,
        skills=skills,
    ).graph


def _render_png(graph, output_path: Path) -> None:
    viz = graph.get_graph()
    if hasattr(viz, "draw_mermaid_png"):
        png_bytes = viz.draw_mermaid_png()
    elif hasattr(viz, "draw_png"):
        png_bytes = viz.draw_png()
    else:
        raise RuntimeError("langgraph graph visualization method not found on compiled graph")
    output_path.write_bytes(png_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the current LangGraph to a PNG.")
    parser.add_argument(
        "--output",
        "-o",
        default="doc/langgraph.png",
        help="Output PNG path (default: doc/langgraph.png).",
    )
    args = parser.parse_args()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = _build_graph()
    _render_png(graph, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
