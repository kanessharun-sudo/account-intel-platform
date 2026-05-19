"""CSV parsing and export utilities."""
from __future__ import annotations

import io
import json
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

from enrichment.schema import EnrichmentResult, Lead


# Acceptable header aliases (case-insensitive). Maps canonical -> aliases.
COLUMN_ALIASES = {
    "name": {"name", "full name", "contact", "lead name", "person"},
    "website": {"website", "url", "domain", "company website", "site"},
    "designation": {
        "designation", "title", "job title", "role", "position",
    },
    "company": {"company", "company name", "organization", "org", "account"},
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical field names to the actual column name in the df."""
    lc_map = {c.lower().strip(): c for c in df.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lc_map:
                resolved[canonical] = lc_map[alias]
                break
    return resolved


def _derive_company_from_website(website: str | None) -> str | None:
    """Fallback: extract a likely company name from the domain."""
    if not website or not isinstance(website, str):
        return None
    site = website.strip()
    if not site:
        return None
    if "://" not in site:
        site = "https://" + site
    try:
        host = urlparse(site).hostname or ""
    except Exception:
        return None
    host = host.lower().lstrip("www.")
    # Strip TLD: "acme.io" -> "acme", "sub.acme.co.uk" -> "acme"
    parts = host.split(".")
    if len(parts) >= 2:
        return parts[-2].capitalize()
    return host.capitalize() or None


def parse_csv(file_bytes: bytes) -> tuple[list[Lead], list[str]]:
    """Parse uploaded CSV bytes into Lead objects.

    Returns (leads, warnings). Skips rows with no name, but tolerates
    missing website/designation/company.
    """
    warnings: list[str] = []
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}") from e

    if df.empty:
        return [], ["CSV has no rows"]

    cols = _resolve_columns(df)
    if "name" not in cols:
        raise ValueError(
            "CSV must include a 'Name' column (aliases: full name, contact, "
            "lead name, person)"
        )

    leads: list[Lead] = []
    canonical_cols = set(cols.values())

    for i, row in df.iterrows():
        name = (row.get(cols["name"]) or "").strip()
        if not name:
            warnings.append(f"Row {i + 2}: skipped (no name)")
            continue

        website = (row.get(cols["website"], "") if "website" in cols else "").strip() or None
        designation = (
            row.get(cols["designation"], "") if "designation" in cols else ""
        ).strip() or None
        company = (
            row.get(cols["company"], "") if "company" in cols else ""
        ).strip() or None

        if not company and website:
            company = _derive_company_from_website(website)

        # Collect any extra (non-canonical) columns as context
        extra = {
            col: row[col]
            for col in df.columns
            if col not in canonical_cols and str(row[col]).strip()
        }

        try:
            leads.append(
                Lead(
                    name=name,
                    website=website,
                    designation=designation,
                    company=company,
                    extra=extra,
                )
            )
        except Exception as e:
            warnings.append(f"Row {i + 2}: skipped ({e})")

    return leads, warnings


def results_to_dataframe(results: Iterable[EnrichmentResult]) -> pd.DataFrame:
    """Flatten EnrichmentResult list into a display-friendly DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "Name": r.name,
                "Company": r.company or "",
                "Designation": r.designation or "",
                "Website": r.website or "",
                "Score": r.score,
                "Relevance": r.score_breakdown.relevance,
                "Urgency": r.score_breakdown.urgency,
                "Time-to-Close": r.score_breakdown.time_to_close,
                "Confidence": r.confidence,
                "Company Summary": r.company_summary,
                "Person Summary": r.person_summary,
                "Signals": " | ".join(r.signals),
                "Justification": r.justification,
                "Sources": " | ".join(r.sources),
                "Error": r.error or "",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def results_to_json(results: Iterable[EnrichmentResult]) -> str:
    """Serialize results as a JSON array string."""
    return json.dumps(
        [r.model_dump() for r in results],
        indent=2,
        ensure_ascii=False,
    )
