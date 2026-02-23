"""LangGraph backend configuration for Gus.

Conversation history uses MemorySaver (resets on restart).
Long-term memory persists via Zep Cloud.
"""

import subprocess
from typing import Any

from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


class ExecutableFilesystemBackend(FilesystemBackend, SandboxBackendProtocol):
    """FilesystemBackend with shell execution support.

    Activity events (JSON lines with "type":"activity") from commands
    are forwarded to the agent's real stdout so the agentstore adapter
    sees them for real-time TUI display. All other output is captured
    and returned to the LLM.
    """

    def execute(self, command: str) -> ExecuteResponse:
        import json
        import sys
        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            output_lines = []
            for line in proc.stdout:
                line = line.rstrip("\n")
                # Forward activity events to the agent's real stdout
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "activity":
                        sys.stdout.write(line + "\n")
                        sys.stdout.flush()
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
                output_lines.append(line)
            stderr = proc.stderr.read()
            proc.wait(timeout=300)
            output = "\n".join(output_lines)
            if stderr:
                output = output + "\n" + stderr if output else stderr
            return ExecuteResponse(
                output=output,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            return ExecuteResponse(output="Command timed out", exit_code=1)

    @property
    def id(self) -> str:
        return f"fs-{id(self)}"


def get_store() -> InMemoryStore:
    """Get the store for LangGraph state."""
    return InMemoryStore()


def get_checkpointer() -> MemorySaver:
    """Get the checkpointer for conversation state."""
    return MemorySaver()


def make_backend(runtime: Any) -> CompositeBackend:
    """Create a CompositeBackend with persistent /memories/ route."""
    return CompositeBackend(
        default=ExecutableFilesystemBackend(),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )
