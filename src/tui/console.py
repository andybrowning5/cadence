"""Cadence TUI - Modern, minimal terminal interface."""

import re
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from .themes import COLORS, CADENCE_THEME


class CadenceTUI:
    """Modern, minimal terminal interface for Cadence."""

    def __init__(self) -> None:
        self.console = Console(
            theme=CADENCE_THEME,
            force_terminal=True,
            force_interactive=True,
        )
        self.history = InMemoryHistory()
        self._prompt_style = PTStyle.from_dict({
            "prompt": COLORS["primary"],
            "arrow": "#4B5563",
        })

    def print_welcome(self) -> None:
        self.console.print()
        self.console.print(
            f"  [bold {COLORS['primary']}]cadence[/bold {COLORS['primary']}]"
        )
        self.console.print()

    def get_input(self) -> str:
        try:
            user_input = prompt(
                HTML('<style fg="#4B5563">></style> '),
                history=self.history,
                style=self._prompt_style,
            )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"

    def print_assistant_message(self, message: str) -> None:
        self.console.print()
        message = _make_urls_clickable(message)
        md = Markdown(message, hyperlinks=True)
        padded = Padding(md, (0, 0, 0, 2))
        self.console.print(padded)
        self.console.print()

    def show_activity_summary(self, activities: list[dict]) -> None:
        if not activities:
            return
        total = len(activities)
        error_count = sum(1 for a in activities if a.get("status") == "error")
        error_text = (
            f" [{COLORS['error']}]· {error_count} failed[/{COLORS['error']}]"
            if error_count > 0
            else ""
        )
        self.console.print(
            f"  [{COLORS['dim']}]⌄ {total} action{'s' if total > 1 else ''}"
            f"{error_text}[/{COLORS['dim']}]"
        )

    def print_error(self, message: str) -> None:
        self.console.print()
        self.console.print(
            f"  [{COLORS['error']}]✗[/{COLORS['error']}] "
            f"[{COLORS['dim']}]{message}[/{COLORS['dim']}]"
        )
        self.console.print()

    def print_info(self, message: str) -> None:
        self.console.print(f"  [{COLORS['dim']}]{message}[/{COLORS['dim']}]")

    def print_success(self, message: str) -> None:
        self.console.print(
            f"  [{COLORS['success']}]✓[/{COLORS['success']}] "
            f"[{COLORS['dim']}]{message}[/{COLORS['dim']}]"
        )

    def print_goodbye(self) -> None:
        self.console.print()
        self.console.print(f"  [{COLORS['dim']}]goodbye[/{COLORS['dim']}]")
        self.console.print()


def _make_urls_clickable(text: str) -> str:
    url_pattern = r'(?<!\]\()https?://[^\s\)\]>]+'

    def replace_url(match: re.Match) -> str:
        url = match.group(0)
        display = url if len(url) <= 60 else url[:57] + "..."
        return f"[{display}]({url})"

    return re.sub(url_pattern, replace_url, text)
