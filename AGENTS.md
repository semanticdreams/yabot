# Agent Logging Requirements

All agent surfaces (daemon, Matrix bot, CLI, or future integrations) must log full traces to the per-user log directory.

Required trace coverage:
- Prompts, parameters, and model selection for every LLM call.
- Tool calls and tool results (including arguments).
- User inputs, approvals/denials, and command invocations.
- Final responses and relevant metadata (room/user IDs, conversation IDs, etc.).

Trace format requirements:
- JSON Lines with stable keys for `schema_version`, `event`, `trace_id`, `room_id`, `conv_id`, and `model` (when available).

Future feature requirement:
- Any new feature must include trace coverage that adapts to its new data/behavior.
- If new metadata or events are introduced, extend the trace schema to include them with stable keys and document updates here.

Logging format and location:
- JSON Lines (one event per line).
- Default path: per-user log dir for the app name, file `trace.jsonl`.
- Override via `YABOT_TRACE_PATH` if needed.

Do not add features that bypass this trace logging.

Trace schema updates:
- `plan_request`: emitted before automatic planning. Keys: `model`, `text`.
- `plan_created`: emitted after automatic planning. Keys: `steps`, `count`.

Assertion policy:
- Prefer hard `assert` statements over fallback behavior across modules to enforce invariants.
