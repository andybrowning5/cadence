"""Color themes and styles for Cadence TUI."""

from rich.theme import Theme

COLORS = {
    "primary": "#F59E0B",      # Warm amber
    "secondary": "#8B5CF6",    # Purple
    "success": "#34D399",      # Mint green
    "warning": "#FBBF24",      # Warm yellow
    "error": "#F87171",        # Soft red
    "muted": "#6B7280",        # Gray
    "text": "#E5E7EB",         # Light gray text
    "dim": "#4B5563",          # Dimmed text
}

STYLES = {
    "header": "bold",
    "tool_name": f"bold {COLORS['warning']}",
    "tool_arg_key": COLORS["primary"],
    "tool_arg_value": COLORS["text"],
    "thinking": f"italic {COLORS['dim']}",
    "timestamp": COLORS["dim"],
}

CADENCE_THEME = Theme({
    "primary": COLORS["primary"],
    "secondary": COLORS["secondary"],
    "success": COLORS["success"],
    "warning": COLORS["warning"],
    "error": COLORS["error"],
    "muted": COLORS["muted"],
    "text": COLORS["text"],
    "dim": COLORS["dim"],
    "user": COLORS["primary"],
    "assistant": COLORS["success"],
    "tool": COLORS["warning"],
    "info": COLORS["primary"],
    "dimmed": COLORS["dim"],
    "highlight": f"bold {COLORS['text']}",
})
