import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FRONTMATTER_BOUNDARY = "---"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    tool_name: str


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills
        self.by_tool_name = {skill.tool_name: skill for skill in skills}

    def tool_defs(self) -> list[dict]:
        tools: list[dict] = []
        for skill in self.skills:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": skill.tool_name,
                        "description": skill.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
        return tools

    def is_skill_tool(self, tool_name: str) -> bool:
        return tool_name in self.by_tool_name

    def get_by_tool_name(self, tool_name: str) -> Skill | None:
        return self.by_tool_name.get(tool_name)


def load_skills(paths: Iterable[str | Path]) -> SkillRegistry:
    skills: list[Skill] = []
    used_tool_names: set[str] = set()

    for path in paths:
        base = Path(path)
        if not base.exists() or not base.is_dir():
            continue
        for file_path in sorted(base.glob("*.md")):
            skill = _load_skill_file(file_path, used_tool_names)
            if skill:
                skills.append(skill)

    return SkillRegistry(skills)


def _load_skill_file(path: Path, used_tool_names: set[str]) -> Skill | None:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    if not frontmatter:
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not name or not description:
        return None

    tool_name = _unique_tool_name(_slugify(name), used_tool_names)
    used_tool_names.add(tool_name)
    content = body.strip()
    return Skill(name=name, description=description, content=content, tool_name=tool_name)


def _split_frontmatter(text: str) -> tuple[dict[str, str] | None, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
        return None, text

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == FRONTMATTER_BOUNDARY:
            end_idx = idx
            break
    if end_idx is None:
        return None, text

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    data: dict[str, str] = {}
    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data, body


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        slug = "skill"
    return f"skill__{slug}"


def _unique_tool_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    return f"{base}_{idx}"
