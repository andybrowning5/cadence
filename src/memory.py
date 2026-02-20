"""Local filesystem memory for Cadence.

Provides persistent conversational memory using JSON files.
Conversation history and facts persist across sessions via the sandbox filesystem.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("/home/user/data/memory")
CONVERSATION_FILE = MEMORY_DIR / "conversations.jsonl"
FACTS_FILE = MEMORY_DIR / "facts.json"

_active_user_id: str = "gus-default-user"
_active_thread_id: str = "default"


def set_ids(user_id: str, thread_id: str) -> None:
    """Set the active user and thread IDs for this session."""
    global _active_user_id, _active_thread_id
    _active_user_id = user_id
    _active_thread_id = thread_id


def init_memory(user_id: str = "gus-default-user", thread_id: str = "default") -> None:
    """Initialize the memory directory. Creates files if they don't exist."""
    set_ids(user_id, thread_id)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not FACTS_FILE.exists():
        FACTS_FILE.write_text(json.dumps({"facts": [], "entities": []}, indent=2))
    if not CONVERSATION_FILE.exists():
        CONVERSATION_FILE.touch()


def save_turn(thread_id: str, user_msg: str, assistant_msg: str) -> None:
    """Save a conversation turn to the conversation log."""
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_msg,
            "assistant": assistant_msg,
        }
        with CONVERSATION_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to save turn: %s", e)


def get_context(user_id: str | None = None) -> str:
    """Build memory context from conversation history and facts.

    Returns a formatted string with recent conversation summaries and known facts.
    """
    sections = []

    # Load facts
    try:
        if FACTS_FILE.exists():
            data = json.loads(FACTS_FILE.read_text())
            facts = data.get("facts", [])
            entities = data.get("entities", [])
            if facts:
                fact_lines = [f"- {f['fact']}" for f in facts[-20:]]
                sections.append("## Known facts\n" + "\n".join(fact_lines))
            if entities:
                entity_lines = [
                    f"- **{e['name']}** ({e.get('type', 'unknown')}): {e.get('summary', '')}"
                    for e in entities[-10:]
                ]
                sections.append("## Key entities\n" + "\n".join(entity_lines))
    except Exception as e:
        logger.warning("Failed to load facts: %s", e)

    # Load recent conversation history
    try:
        if CONVERSATION_FILE.exists():
            lines = CONVERSATION_FILE.read_text().strip().split("\n")
            recent = lines[-10:]  # last 10 turns
            if recent and recent[0]:
                history_lines = []
                for line in recent:
                    try:
                        entry = json.loads(line)
                        history_lines.append(
                            f"- User: {entry['user'][:100]}\n"
                            f"  Gus: {entry['assistant'][:100]}"
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue
                if history_lines:
                    sections.append("## Recent conversation history\n" + "\n".join(history_lines))
    except Exception as e:
        logger.warning("Failed to load conversation history: %s", e)

    return "\n\n".join(sections)


@tool
def remember(query: str) -> str:
    """Search your memory for facts, people, projects, deadlines, or any past context.

    You already receive automatic memory context with each message, but use this
    tool when you need to dig deeper on a specific topic — a person, a deadline,
    a project, or anything the user mentioned in a previous session.

    Args:
        query: What to search for (e.g., "Sarah", "report deadline", "Q4 project").

    Returns:
        Relevant facts and conversation excerpts from past sessions.
    """
    query_lower = query.lower()
    sections = []

    # Search facts
    try:
        if FACTS_FILE.exists():
            data = json.loads(FACTS_FILE.read_text())
            matching_facts = [
                f for f in data.get("facts", [])
                if query_lower in f.get("fact", "").lower()
            ]
            if matching_facts:
                lines = [f"- {f['fact']}" for f in matching_facts[:10]]
                sections.append("Facts:\n" + "\n".join(lines))

            matching_entities = [
                e for e in data.get("entities", [])
                if query_lower in e.get("name", "").lower()
                or query_lower in e.get("summary", "").lower()
            ]
            if matching_entities:
                lines = [
                    f"- {e['name']} ({e.get('type', '')}): {e.get('summary', '')}"
                    for e in matching_entities[:8]
                ]
                sections.append("Entities:\n" + "\n".join(lines))
    except Exception as e:
        logger.warning("remember facts search failed: %s", e)

    # Search conversation history
    try:
        if CONVERSATION_FILE.exists():
            lines = CONVERSATION_FILE.read_text().strip().split("\n")
            matches = []
            for line in lines:
                try:
                    entry = json.loads(line)
                    if (query_lower in entry.get("user", "").lower()
                            or query_lower in entry.get("assistant", "").lower()):
                        matches.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
            if matches:
                excerpts = []
                for m in matches[-5:]:
                    excerpts.append(
                        f"- [{m.get('timestamp', '?')[:10]}] "
                        f"User: {m['user'][:80]} → Gus: {m['assistant'][:80]}"
                    )
                sections.append("Past conversations:\n" + "\n".join(excerpts))
    except Exception as e:
        logger.warning("remember conversation search failed: %s", e)

    if not sections:
        return f"(nothing found for '{query}')"
    return "\n\n".join(sections)


# Keep old name as alias for compatibility
init_zep = init_memory
