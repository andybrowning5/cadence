"""LangChain callbacks for real-time TUI updates."""

import json
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .live_display import LiveActivityDisplay


class TUICallbackHandler(BaseCallbackHandler):
    """Callback handler that updates LiveActivityDisplay in real-time."""

    def __init__(self, display: LiveActivityDisplay) -> None:
        super().__init__()
        self.display = display
        self.tool_run_ids: dict[UUID, str] = {}
        self.chain_depth = 0

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any],
        *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any,
    ) -> None:
        self.chain_depth += 1
        if self.chain_depth == 1:
            self.display.set_thought("thinking...")

    def on_chain_end(
        self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: Any,
    ) -> None:
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str,
        *, run_id: UUID, parent_run_id: UUID | None = None,
        tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None, **kwargs: Any,
    ) -> None:
        self.display.set_thought("")
        tool_name = serialized.get("name", "tool")

        args = {}
        if inputs:
            args = inputs
        elif input_str:
            try:
                args = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                if input_str and len(input_str) < 100:
                    args = {"input": input_str}

        activity_id = self.display.add_tool_call(tool_name, args)
        self.tool_run_ids[run_id] = activity_id

    def on_tool_end(
        self, output: Any, *, run_id: UUID, **kwargs: Any,
    ) -> None:
        activity_id = self.tool_run_ids.get(run_id)
        if not activity_id:
            return

        result_preview = ""
        if isinstance(output, str):
            result_preview = output[:60]
        elif isinstance(output, dict):
            for key in ["error", "message", "result", "output", "content"]:
                if key in output:
                    result_preview = str(output[key])[:60]
                    break
            if not result_preview:
                result_preview = "done"
        else:
            result_preview = str(output)[:60] if output else "done"

        success = "error" not in result_preview.lower()
        self.display.complete_activity(activity_id, result_preview, success)
        del self.tool_run_ids[run_id]

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any,
    ) -> None:
        activity_id = self.tool_run_ids.get(run_id)
        if activity_id:
            self.display.complete_activity(
                activity_id, f"error: {str(error)[:50]}", success=False
            )
            del self.tool_run_ids[run_id]

