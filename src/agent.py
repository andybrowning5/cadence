"""Cadence: Task Prioritization Agent factory.

Creates the agent using LangChain Deep Agents framework.
"""

import os
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from .backends import get_checkpointer, get_store, make_backend
from .prompts import get_prioritizer_prompt
from .memory import get_context, init_memory, remember, save_turn

load_dotenv()


def get_model(model_name: str | None = None) -> Any:
    """Initialize the chat model based on environment.

    Args:
        model_name: Optional model identifier (e.g., "openai:gpt-4o").

    Returns:
        Initialized chat model instance.
    """
    if model_name:
        return init_chat_model(model_name)

    if os.getenv("ANTHROPIC_API_KEY"):
        return init_chat_model("anthropic:claude-sonnet-4-5-20250929")
    elif os.getenv("OPENAI_API_KEY"):
        return init_chat_model("openai:gpt-4o")
    else:
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )


def create_cadence_agent(model_name: str | None = None) -> Any:
    """Create the Cadence task prioritization agent.

    Args:
        model_name: Optional model identifier.

    Returns:
        Configured deep agent instance.
    """
    model = get_model(model_name)
    store = get_store()
    checkpointer = get_checkpointer()

    tools = [remember]

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=get_prioritizer_prompt(),
        store=store,
        checkpointer=checkpointer,
        backend=make_backend,
    )

    return agent


def run_cadence(
    model_name: str | None = None,
    thread_id: str = "default",
) -> None:
    """Run Cadence in interactive mode with TUI.

    Args:
        model_name: Optional model identifier.
        thread_id: Conversation thread ID for persistence.
    """
    from .tui import CadenceTUI, LiveActivityDisplay

    tui = CadenceTUI()
    tui.print_welcome()

    agent = create_cadence_agent(model_name)
    config = {"configurable": {"thread_id": thread_id}}

    from .memory import set_ids
    set_ids("gus-default-user", thread_id)
    init_memory(user_id="gus-default-user", thread_id=thread_id)

    while True:
        try:
            user_input = tui.get_input()

            if user_input.lower() in ("quit", "exit", "q"):
                tui.print_goodbye()
                break

            if not user_input:
                continue

            if user_input.lower() in ("/help", "?"):
                tui.print_info("Commands: q = quit, ? = help")
                continue

            response, activities = _stream_with_display(
                agent, user_input, config, tui.console
            )

            tui.show_activity_summary(activities)

            if response:
                tui.print_assistant_message(response)
                save_turn(thread_id, user_input, response)

        except KeyboardInterrupt:
            tui.print_goodbye()
            break
        except Exception as e:
            tui.print_error(str(e))
            import traceback
            traceback.print_exc()


def _stream_with_display(
    agent: Any,
    user_input: str,
    config: dict[str, Any],
    console: Any,
) -> tuple[str, list[dict]]:
    """Stream agent response with live activity display."""
    from .tui import LiveActivityDisplay, TUICallbackHandler

    final_response = ""
    activities: list[dict] = []

    with LiveActivityDisplay(console) as display:
        callback_handler = TUICallbackHandler(display)

        config_with_callbacks = {
            **config,
            "callbacks": [callback_handler],
        }

        display.set_thought("thinking...")

        # Add timestamp and Zep context to the user message
        from .prompts import get_current_timestamp
        parts = [f"[timestamp: {get_current_timestamp()}]"]
        zep_context = get_context()
        if zep_context:
            parts.append(f"## Memory context from past sessions\n{zep_context}")
        parts.append(f"## User message\n{user_input}")
        enriched_input = "\n\n".join(parts)
        messages: list[dict[str, str]] = [{"role": "user", "content": enriched_input}]

        try:
            if hasattr(agent, "stream"):
                for event in agent.stream(
                    {"messages": messages},
                    config=config_with_callbacks,
                    stream_mode="values",
                ):
                    if isinstance(event, dict) and "messages" in event:
                        event_messages = event["messages"]
                        for msg in reversed(event_messages):
                            if getattr(msg, "type", None) == "ai":
                                content = getattr(msg, "content", "")
                                if isinstance(content, str) and content.strip():
                                    final_response = content
                                    break
                                elif isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            final_response = block.get("text", "")
                                            break
                                    if final_response:
                                        break
            else:
                result = agent.invoke(
                    {"messages": messages},
                    config=config_with_callbacks,
                )
                final_response = _extract_response(result)

        except Exception:
            display.set_thought("processing...")
            try:
                result = agent.invoke(
                    {"messages": messages},
                    config=config_with_callbacks,
                )
                final_response = _extract_response(result)
            except Exception as invoke_error:
                display.set_thought("")
                raise invoke_error

        activities = display.get_summary()

    return final_response, activities


def _extract_response(result: dict[str, Any]) -> str:
    """Extract the assistant's text response from result."""
    if "messages" not in result:
        return ""

    for msg in reversed(result["messages"]):
        if (
            hasattr(msg, "type")
            and msg.type == "ai"
            and hasattr(msg, "content")
            and msg.content
        ):
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            if content:
                return content

    return ""


if __name__ == "__main__":
    run_cadence()
