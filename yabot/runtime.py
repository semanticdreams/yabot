from __future__ import annotations

from pathlib import Path

from .checkpoint import FileBackedSaver
from .config import Config
from .graph import YabotGraph
from .llm import LLMClient
from .skills import load_skills
from .system_prompt import system_prompt
from .trace import TraceLogger


def build_graph(config: Config) -> YabotGraph:
    llm = LLMClient(api_key=config.openai_api_key)
    checkpointer = FileBackedSaver(f"{config.data_dir}/graph_state.pkl")

    builtin_skills_dir = Path(__file__).resolve().parent.parent / "skills"
    user_skills_dir = Path(config.data_dir) / "skills"
    user_skills_dir.mkdir(parents=True, exist_ok=True)
    skills = load_skills([builtin_skills_dir, user_skills_dir])
    tracer = TraceLogger(Path(config.trace_path))
    tracer.log(
        "startup",
        {
            "trace_path": str(config.trace_path),
            "data_dir": config.data_dir,
        },
    )

    return YabotGraph(
        llm=llm,
        default_model=config.default_model,
        available_models=config.available_models,
        max_turns=config.max_turns,
        skills=skills,
        checkpointer=checkpointer,
        tracer=tracer,
        system_prompt=system_prompt(),
    )
