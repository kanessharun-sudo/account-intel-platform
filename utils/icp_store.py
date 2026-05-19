"""Persist the user's ICP definition to a local JSON file."""
from __future__ import annotations

import json
from pathlib import Path

ICP_PATH = Path(__file__).resolve().parent.parent / "storage" / "icp.json"

DEFAULT_ICP_TEMPLATE = """\
## What my company does
[Describe your product/service in 2-3 sentences.]

## Target industries
- e.g. B2B SaaS, fintech, healthcare

## Target company size
- e.g. 50-500 employees, Series A through C

## Target geographies
- e.g. North America, Western Europe

## Target job titles / personas
- e.g. Head of Revenue Operations, VP Marketing, Director of Sales Enablement

## Pain points we solve
- e.g. Sales reps spend too long researching accounts before outreach
- e.g. Marketing can't tell which leads are worth prioritizing

## Disqualifiers (auto-low score)
- e.g. Company has <10 employees
- e.g. Industry is consumer retail
"""


def load_icp() -> str:
    """Load the ICP text, returning the default template if none saved yet."""
    if not ICP_PATH.exists():
        return DEFAULT_ICP_TEMPLATE
    try:
        data = json.loads(ICP_PATH.read_text(encoding="utf-8"))
        return data.get("icp", DEFAULT_ICP_TEMPLATE)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_ICP_TEMPLATE


def save_icp(icp_text: str) -> None:
    """Persist the ICP text to disk."""
    ICP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ICP_PATH.write_text(
        json.dumps({"icp": icp_text}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
