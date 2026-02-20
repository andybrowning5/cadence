"""Cadence Agent Store adapter.

Speaks the Primordial Protocol (NDJSON over stdin/stdout) to run Cadence
on the Primordial AgentStore platform.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Persistent state lives at /home/user/ inside the sandbox
STATE_DIR = Path("/home/user")


def send(msg: dict) -> None:
    """Write a single Ooze Protocol message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _load_or_create_id(filename: str, prefix: str) -> str:
    """Load a persistent ID from file, or create and save a new one."""
    path = STATE_DIR / filename
    if path.exists():
        return path.read_text().strip()
    new_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    path.write_text(new_id)
    return new_id


def setup() -> tuple[Any, dict, str]:
    """Initialize Cadence's LangGraph agent with persistent local memory."""
    try:
        from src.agent import create_cadence_agent
        from src.memory import init_memory, set_ids
    except ImportError:
        sys.path.insert(0, "/home/user/agent")
        from src.agent import create_cadence_agent
        from src.memory import init_memory, set_ids

    user_id = _load_or_create_id("gus_user_id.txt", "gus-user")
    thread_id = _load_or_create_id("gus_thread_id.txt", "gus-thread")

    set_ids(user_id, thread_id)
    init_memory(user_id=user_id, thread_id=thread_id)

    agent = create_cadence_agent()
    config = {"configurable": {"thread_id": thread_id}}

    return agent, config, thread_id


def _extract_response(result: dict[str, Any]) -> str:
    """Extract assistant text from a LangGraph invoke result."""
    if "messages" not in result:
        return ""
    for msg in reversed(result["messages"]):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            content = msg.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts)
            if content:
                return content
    return ""


def handle_message(
    agent: Any,
    config: dict,
    zep_thread_id: str,
    content: str,
    message_id: str,
) -> None:
    """Process a user message through Cadence's agent graph."""
    from src.memory import get_context, save_turn  # noqa: F811

    try:
        # Build user message with timestamp and Zep context
        from src.prompts import get_current_timestamp
        parts = [f"[timestamp: {get_current_timestamp()}]"]

        zep_context = get_context()
        if zep_context:
            parts.append(f"## Memory context from past sessions\n{zep_context}")

        parts.append(f"## User message\n{content}")
        user_content = "\n\n".join(parts)

        messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]
        final_response = ""
        # Track emitted tool calls to avoid duplicates (stream_mode="values"
        # replays the full message list on each event)
        emitted_tool_calls: set[str] = set()

        if hasattr(agent, "stream"):
            for event in agent.stream(
                {"messages": messages},
                config=config,
                stream_mode="values",
            ):
                if not isinstance(event, dict) or "messages" not in event:
                    continue

                for msg in event["messages"]:
                    # Report tool calls as activity (deduplicated)
                    if (
                        getattr(msg, "type", None) == "ai"
                        and hasattr(msg, "tool_calls")
                        and msg.tool_calls
                    ):
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id") or tc.get("name", "")
                            if tc_id in emitted_tool_calls:
                                continue
                            emitted_tool_calls.add(tc_id)
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            query = tool_args.get("query") or tool_args.get("topic") or ""
                            desc = f"{tool_name}({query})" if query else tool_name
                            send({
                                "type": "activity",
                                "tool": tool_name,
                                "description": desc,
                                "message_id": message_id,
                            })

                # Extract latest AI text response from the end of the message list
                for msg in reversed(event["messages"]):
                    if getattr(msg, "type", None) != "ai":
                        continue
                    msg_content = getattr(msg, "content", "")
                    if isinstance(msg_content, str) and msg_content.strip():
                        final_response = msg_content
                        break
                    elif isinstance(msg_content, list):
                        for block in msg_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_response = block.get("text", "")
                                break
                        if final_response:
                            break
        else:
            result = agent.invoke(
                {"messages": messages},
                config=config,
            )
            final_response = _extract_response(result)

        if final_response:
            send({"type": "response", "content": final_response, "message_id": message_id, "done": True})
            save_turn(zep_thread_id, content, final_response)
        else:
            send({
                "type": "response",
                "content": "I processed your message but didn't generate a text response.",
                "message_id": message_id,
                "done": True,
            })

    except Exception as exc:
        send({"type": "error", "error": str(exc), "message_id": message_id})


def main() -> None:
    """Ooze Protocol main loop."""
    agent, config, zep_thread_id = setup()
    send({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg["type"] == "shutdown":
            break

        if msg["type"] == "message":
            handle_message(
                agent,
                config,
                zep_thread_id,
                msg["content"],
                msg["message_id"],
            )


if __name__ == "__main__":
    main()
