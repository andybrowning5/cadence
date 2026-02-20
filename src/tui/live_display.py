"""Live activity display for real-time tool feedback."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import cycle
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from .themes import COLORS

SPINNER_FRAMES = ["✢", "✧", "✦", "✧"]


class ActivityStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ActivityItem:
    name: str
    description: str = ""
    status: ActivityStatus = ActivityStatus.PENDING
    result: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    is_subagent: bool = False


class LiveActivityDisplay:
    """Manages a live-updating activity panel."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.activities: list[ActivityItem] = []
        self.current_thought: str = ""
        self.live: Live | None = None
        self._start_time: datetime = datetime.now()
        self._spinner = cycle(SPINNER_FRAMES)
        self._spinner_frame = next(self._spinner)
        self._frame_count = 0
        self._running = False
        self._refresh_thread: threading.Thread | None = None

    def _render_status_line(self) -> Text:
        self._frame_count += 1
        if self._frame_count % 2 == 0:
            self._spinner_frame = next(self._spinner)

        text = Text()
        text.append("  ")
        text.append(self._spinner_frame, style=COLORS["primary"])
        text.append(" ")

        if self.current_thought:
            status = self.current_thought.lower().rstrip(".")
        else:
            running = [a for a in self.activities if a.status == ActivityStatus.RUNNING]
            if running:
                status = running[-1].name.lower()
            else:
                status = "thinking"

        text.append(status, style=COLORS["dim"])

        elapsed = _format_elapsed_compact(self._start_time)
        if elapsed:
            text.append(f" ({elapsed})", style=COLORS["dim"])

        return text

    def _render(self) -> Group:
        elements = [self._render_status_line()]
        running = [a for a in self.activities if a.status == ActivityStatus.RUNNING]
        for activity in running[-3:]:
            line = Text()
            line.append("    ○ ", style=COLORS["dim"])
            line.append(activity.name.lower(), style=COLORS["dim"])
            elements.append(line)
        return Group(*elements)

    def _refresh_loop(self) -> None:
        while self._running:
            time.sleep(0.25)
            if self.live and self._running:
                try:
                    self.live.update(self._render())
                except Exception:
                    pass

    def __enter__(self) -> "LiveActivityDisplay":
        self.activities.clear()
        self.current_thought = ""
        self._start_time = datetime.now()
        self._spinner = cycle(SPINNER_FRAMES)
        self._spinner_frame = next(self._spinner)
        self._frame_count = 0
        self._running = True

        self.live = Live(self._render(), console=self.console, transient=True)
        self.live.__enter__()

        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

        return self

    def __exit__(self, *args) -> None:
        self._running = False
        if self.live:
            self.live.__exit__(*args)
            self.live = None

    def _update(self) -> None:
        if self.live:
            self.live.update(self._render())

    def set_thought(self, thought: str) -> None:
        self.current_thought = thought
        self._update()

    def add_tool_call(self, name: str, args: dict[str, Any] | None = None) -> str:
        desc = ""
        if args:
            key_args = [f"{k}={_truncate(str(v), 20)}" for k, v in list(args.items())[:2]]
            if key_args:
                desc = f"({', '.join(key_args)})"

        activity = ActivityItem(name=name, description=desc, status=ActivityStatus.RUNNING)
        self.activities.append(activity)
        self._update()
        return str(len(self.activities) - 1)

    def add_subagent(self, name: str, task: str = "", depth: int = 0) -> str:
        activity = ActivityItem(
            name=name, description=task, status=ActivityStatus.RUNNING, is_subagent=True
        )
        self.activities.append(activity)
        self._update()
        return str(len(self.activities) - 1)

    def complete_activity(self, activity_id: str, result: str = "", success: bool = True) -> None:
        try:
            idx = int(activity_id)
            if 0 <= idx < len(self.activities):
                self.activities[idx].status = (
                    ActivityStatus.SUCCESS if success else ActivityStatus.ERROR
                )
                self.activities[idx].result = result
                self.activities[idx].completed_at = datetime.now()
                self._update()
        except (ValueError, IndexError):
            pass

    def get_summary(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "status": a.status.value,
                "is_subagent": a.is_subagent,
                "result": a.result,
            }
            for a in self.activities
        ]


def _truncate(text: str, max_len: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _format_elapsed_compact(started_at: datetime) -> str:
    elapsed = (datetime.now() - started_at).total_seconds()
    if elapsed < 1:
        return ""
    elif elapsed < 60:
        return f"{int(elapsed)}s"
    else:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs}s"
