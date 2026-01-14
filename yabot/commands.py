import re
import uuid
from typing import Any, Dict, List, Tuple

from .tokens import (
    context_window_for_model,
    estimate_message_tokens,
    estimate_messages_tokens,
    get_encoding,
    output_reserve_tokens,
)


CMD_RE = re.compile(r"^!([\w-]+)(?:\s+(.*))?$")
AGENT_NAMES = ("main", "meta")


def help_text() -> str:
    return (
        "Commands:\n"
        "!models\n"
        "!model <name>\n"
        "!become <main|meta>\n"
        "!new\n"
        "!list\n"
        "!use <id>\n"
        "!reset\n"
        "!remaining-context-percentage\n"
        "!stop\n"
    )


State = Dict[str, Any]


def parse_command(text: str) -> Tuple[str, str] | None:
    m = CMD_RE.match(text)
    if not m:
        return None
    cmd = m.group(1).lower()
    arg = (m.group(2) or "").strip()
    return cmd, arg


def ensure_state(state: State, default_model: str) -> None:
    if "agents" not in state:
        if "conversations" in state and "active" in state:
            main_state = {
                "active": state.get("active"),
                "conversations": state.get("conversations", {}),
                "tool_history": state.get("tool_history", []),
            }
        else:
            main_state = _init_agent_state(default_model)
        state["agents"] = {
            "main": main_state,
            "meta": _init_agent_state(default_model),
        }
    if "active_agent" not in state:
        state["active_agent"] = "meta"
    _sync_active_agent_state(state)
    if "approvals" not in state:
        state["approvals"] = {"shell": [], "dirs": [], "pending": None}


def set_active_agent(state: State, agent: str) -> None:
    state["active_agent"] = agent
    _sync_active_agent_state(state)


def agent_state(state: State, agent: str) -> Dict[str, Any]:
    agents = state.get("agents", {})
    return agents.get(agent, {})


def agent_active_conv(state: State, agent: str) -> Tuple[str | None, Dict[str, Any]]:
    state_agent = agent_state(state, agent)
    conv_id = state_agent.get("active")
    if not conv_id:
        return None, {}
    return conv_id, state_agent.get("conversations", {}).get(conv_id, {})


def agent_set_model(state: State, agent: str, model: str, available_models: List[str]) -> bool:
    if model not in available_models:
        return False
    _, conv = agent_active_conv(state, agent)
    if not conv:
        return False
    conv["model"] = model
    return True


def agent_record_tool_calls(state: State, agent: str, tool_calls: List[Dict[str, Any]]) -> None:
    state_agent = agent_state(state, agent)
    history = state_agent.setdefault("tool_history", [])
    for call in tool_calls:
        function = call.get("function") or {}
        history.append(
            {
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", ""),
            }
        )
    if len(history) > 50:
        state_agent["tool_history"] = history[-50:]


def agent_recent_tool_calls(state: State, agent: str, limit: int = 10) -> List[Dict[str, Any]]:
    state_agent = agent_state(state, agent)
    history = list(state_agent.get("tool_history", []))
    return history[-max(1, limit) :]


def active_conversation_meta(state: State) -> Tuple[str | None, str | None]:
    conv_id = state.get("active")
    if not conv_id:
        return None, None
    conv = state.get("conversations", {}).get(conv_id, {})
    return conv_id, conv.get("model")


def room_active_conv(state: State) -> Tuple[str, Dict[str, Any]]:
    conv_id = state["active"]
    return conv_id, state["conversations"][conv_id]


def room_new_conv(state: State, default_model: str) -> str:
    conv_id = _new_conv_id()
    state["conversations"][conv_id] = {"model": default_model, "messages": []}
    _set_active_conv(state, conv_id)
    return conv_id


def room_list_convs(state: State) -> List[Tuple[str, str, int, bool]]:
    active = state["active"]
    out = []
    for cid, c in state["conversations"].items():
        out.append((cid, c.get("model", ""), len(c.get("messages", [])), cid == active))
    out.sort(key=lambda x: (not x[3], x[0]))
    return out


def room_use_conv(state: State, conv_id: str) -> bool:
    if conv_id not in state["conversations"]:
        return False
    _set_active_conv(state, conv_id)
    return True


def room_set_model(state: State, model: str, available_models: List[str]) -> bool:
    if model not in available_models:
        return False
    _, conv = room_active_conv(state)
    conv["model"] = model
    return True


def room_reset(state: State) -> None:
    _, conv = room_active_conv(state)
    conv["messages"] = []


def handle_command(
    state: State,
    cmd: str,
    arg: str,
    available_models: List[str],
    default_model: str,
) -> str:
    if cmd in {"help", "h", "?"}:
        return help_text()
    if cmd == "models":
        return "Available models:\n" + "\n".join(f"- {x}" for x in available_models)
    if cmd == "model":
        if not arg:
            return "Usage: !model <name>\n" + "\n".join(available_models)
        if not room_set_model(state, arg, available_models):
            return f"Unknown model `{arg}`.\nUse !models."
        return f"Model set to `{arg}` for this room’s active conversation."
    if cmd == "become":
        if not arg:
            return "Usage: !become <main|meta>"
        agent = arg.lower()
        if agent not in AGENT_NAMES:
            return f"Unknown agent `{arg}`. Available: main, meta."
        set_active_agent(state, agent)
        return f"Active agent is now `{agent}`."
    if cmd == "new":
        cid = room_new_conv(state, default_model)
        return f"Started new conversation for this room: `{cid}`"
    if cmd == "list":
        items = room_list_convs(state)
        lines = ["Conversations for this room:"]
        for cid, model, nmsgs, is_active in items:
            mark = " (active)" if is_active else ""
            lines.append(f"- `{cid}` [{model}] msgs={nmsgs}{mark}")
        return "\n".join(lines)
    if cmd == "use":
        if not arg:
            return "Usage: !use <conversation_id>"
        if not room_use_conv(state, arg):
            return f"No conversation `{arg}` in this room. Use !list."
        return f"Switched active conversation to `{arg}` for this room."
    if cmd == "reset":
        room_reset(state)
        return "Cleared memory for this room’s active conversation."
    if cmd in {"remaining-context-percentage", "remaining-context", "context"}:
        return remaining_context_percentage(state, default_model)
    return "Unknown command.\n" + help_text()


def remaining_context_percentage(state: State, default_model: str) -> str:
    conv_id = state.get("active")
    conv = state.get("conversations", {}).get(conv_id, {}) if conv_id else {}
    model = conv.get("model", default_model)
    messages = conv.get("messages", [])
    if not isinstance(messages, list):
        return "Remaining context: 0%"
    context_window = context_window_for_model(model)
    reserve = output_reserve_tokens(context_window)
    input_budget = max(1, context_window - reserve)
    encoding = get_encoding(model)
    used = estimate_messages_tokens(messages, encoding)
    remaining = max(0, input_budget - used)
    percent = int((remaining / input_budget) * 100)
    return f"Remaining context: {percent}%"


def trim_messages(messages: List[Dict[str, Any]], model: str, max_turns: int) -> List[Dict[str, Any]]:
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]
    keep = other[-(max_turns * 2):]
    kept = sys_msgs + keep

    context_window = context_window_for_model(model)
    reserve = output_reserve_tokens(context_window)
    input_budget = max(0, context_window - reserve)

    encoding = get_encoding(model)
    if estimate_messages_tokens(kept, encoding) <= input_budget:
        return kept

    sys_msgs = [m for m in kept if m.get("role") == "system"]
    other = [m for m in kept if m.get("role") != "system"]

    sys_counts = [estimate_message_tokens(m, encoding) for m in sys_msgs]
    other_counts = [estimate_message_tokens(m, encoding) for m in other]
    total = sum(sys_counts) + sum(other_counts)

    while total > input_budget and other:
        total -= other_counts.pop(0)
        other.pop(0)

    while total > input_budget and sys_msgs:
        total -= sys_counts.pop(0)
        sys_msgs.pop(0)

    return sys_msgs + other


def _new_conv_id() -> str:
    return str(uuid.uuid4())[:8]


def _init_agent_state(default_model: str) -> Dict[str, Any]:
    conv_id = _new_conv_id()
    return {
        "active": conv_id,
        "conversations": {conv_id: {"model": default_model, "messages": []}},
        "tool_history": [],
    }


def _sync_active_agent_state(state: State) -> None:
    active_agent = state.get("active_agent")
    agents = state.get("agents", {})
    if active_agent not in agents:
        active_agent = "meta" if "meta" in agents else next(iter(agents), None)
        state["active_agent"] = active_agent
    if not active_agent:
        return
    agent = agents.get(active_agent, {})
    state["active"] = agent.get("active")
    state["conversations"] = agent.get("conversations", {})


def _set_active_conv(state: State, conv_id: str) -> None:
    state["active"] = conv_id
    active_agent = state.get("active_agent")
    if not active_agent:
        return
    agent = state.get("agents", {}).get(active_agent)
    if agent is None:
        return
    agent["active"] = conv_id
