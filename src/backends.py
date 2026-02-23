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
    """FilesystemBackend with shell execution support."""

    def execute(self, command: str) -> ExecuteResponse:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=300,
            )
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
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
