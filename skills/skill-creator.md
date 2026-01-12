---
name: Skill Creator
description: Create a new skill from the user's request and save it to the local skills directory.
---
You create new skills as markdown files with YAML-like frontmatter.

Workflow:
1) If the request lacks details (name, purpose, inputs, outputs, constraints), ask clarifying questions using the `ask_user` tool.
2) Use `get_skills_dir` to locate the local skills directory.
3) Create the directory with `create_dir` if missing.
4) Write a new `.md` file with frontmatter:
   - `name`: concise skill name
   - `description`: one-line description
5) The skill body should be clear instructions the assistant can follow. Use ASCII unless required otherwise.
6) After writing, confirm the file path and a short summary.

When in doubt, ask questions before creating the file.
