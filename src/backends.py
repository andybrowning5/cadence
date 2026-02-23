"""LangGraph backend configuration for Gus.

Conversation history uses MemorySaver (resets on restart).
Long-term memory persists via Zep Cloud.
"""

from typing import Any

from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


def get_store() -> InMemoryStore:
    """Get the store for LangGraph state."""
    return InMemoryStore()


def get_checkpointer() -> MemorySaver:
    """Get the checkpointer for conversation state."""
    return MemorySaver()


def make_backend(runtime: Any) -> CompositeBackend:
    """Create a CompositeBackend with persistent /memories/ route."""
    return CompositeBackend(
        default=FilesystemBackend(),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )
