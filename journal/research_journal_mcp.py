#!/usr/bin/env python3.10
"""MCP server for daily research journaling.

Provides tools to log what was studied/tried each day and review past entries.
Entries are stored as markdown files in the journal/ directory.
"""

import os
import re
from datetime import date, timedelta
from pathlib import Path
from mcp.server.fastmcp import FastMCP

JOURNAL_DIR = Path(__file__).parent

mcp = FastMCP("Research Journal")


def _entry_path(d: date) -> Path:
    return JOURNAL_DIR / f"{d.year}" / f"{d.month:02d}" / f"{d.isoformat()}.md"


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _list_entry_files() -> list[Path]:
    return sorted(JOURNAL_DIR.glob("????/??/????-??-??.md"))


@mcp.tool()
def log_entry(content: str, entry_date: str = "") -> str:
    """Log a research journal entry for a given date.

    Args:
        content: Markdown-formatted text describing what was studied, tried,
                 results, observations, and next steps.
        entry_date: Date in YYYY-MM-DD format. Defaults to today.
    """
    d = _parse_date(entry_date) if entry_date else date.today()
    path = _entry_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text()
        # Append under a separator
        new_content = existing.rstrip() + "\n\n---\n\n" + content + "\n"
    else:
        new_content = f"# Research Journal — {d.isoformat()}\n\n{content}\n"

    path.write_text(new_content)
    return f"Entry saved to {path.name}"


@mcp.tool()
def read_entry(entry_date: str = "") -> str:
    """Read the journal entry for a specific date.

    Args:
        entry_date: Date in YYYY-MM-DD format. Defaults to today.
    """
    d = _parse_date(entry_date) if entry_date else date.today()
    path = _entry_path(d)
    if not path.exists():
        return f"No entry for {d.isoformat()}"
    return path.read_text()


@mcp.tool()
def list_entries(last_n: int = 0) -> str:
    """List all journal entry dates.

    Args:
        last_n: If > 0, only show the last N entries. 0 means show all.
    """
    files = _list_entry_files()
    if not files:
        return "No journal entries yet."

    dates = [f.stem for f in files]
    if last_n > 0:
        dates = dates[-last_n:]

    return "Journal entries:\n" + "\n".join(f"- {d}" for d in dates)


@mcp.tool()
def read_recent(days: int = 7) -> str:
    """Read all journal entries from the last N days.

    Args:
        days: Number of days to look back (default 7).
    """
    today = date.today()
    parts = []
    for i in range(days):
        d = today - timedelta(days=i)
        path = _entry_path(d)
        if path.exists():
            parts.append(path.read_text().rstrip())

    if not parts:
        return f"No entries in the last {days} days."
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_entries(query: str) -> str:
    """Search all journal entries for a keyword or phrase.

    Args:
        query: Text to search for (case-insensitive).
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results = []
    for path in _list_entry_files():
        text = path.read_text()
        matches = pattern.findall(text)
        if matches:
            # Show first matching line for context
            for line in text.splitlines():
                if pattern.search(line):
                    results.append(f"**{path.stem}**: {line.strip()}")
                    break

    if not results:
        return f"No entries matching '{query}'."
    return f"Found {len(results)} entries:\n" + "\n".join(results)


if __name__ == "__main__":
    mcp.run()
