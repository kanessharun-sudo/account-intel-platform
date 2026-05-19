"""Claude agentic loop for lead enrichment.

Uses Anthropic's server-side tools:
- web_search_20260209  (GA, dynamic filtering)
- web_fetch_20260209   (GA, dynamic filtering)

Both execute on Anthropic's infrastructure — no scraping infra required.
We just call messages.create() and Claude handles the multi-step research
internally, returning a final assistant message.

NOTE: Tool version strings verified against docs.claude.com on
2026-05-19. If Anthropic rotates versions, update TOOL_DEFINITIONS below.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from anthropic import AsyncAnthropic, APIError

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import EnrichmentResult, Lead, ScoreBreakdown

log = logging.getLogger(__name__)

# Server-side tools provided by Anthropic. The version suffixes are
# capability-keyed (the 20260209 versions add dynamic filtering — Claude
# post-processes results in a sandbox before they hit context).
TOOL_DEFINITIONS = [
    {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 4,  # cap per request
    },
    {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
        "max_uses": 3,
        "citations": {"enabled": True},
    },
]

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096


def _extract_final_text(message) -> str:
    """Pull the assistant's final text content out of a Messages response.

    With server tools, all tool_use / tool_result blocks are handled
    server-side; we receive a single response object whose `content` is
    a list of blocks. We want the trailing text block(s).
    """
    parts = []
    for block in message.content:
        # The SDK returns typed objects; we check the .type attr defensively.
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(p for p in parts if p).strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from the model's text.

    The prompt instructs JSON-only, but we defensively strip code fences
    and locate the outermost {...} if the model adds preamble.
    """
    if not text:
        return None

    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1).strip() if fence else text.strip()

    # Try a direct parse first
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: find first '{' and matching last '}'
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


async def enrich_lead(
    client: AsyncAnthropic,
    lead: Lead,
    icp_text: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 180.0,
) -> EnrichmentResult:
    """Run the agentic enrichment loop for a single lead.

    Because web_search and web_fetch are server-side tools, we don't have
    to implement the tool_use -> tool_result loop ourselves. Anthropic's
    infrastructure executes the searches and feeds results back to the
    model before returning the final response. From our side, this is a
    single messages.create() call.
    """
    user_msg = build_user_prompt(lead, icp_text)

    try:
        message = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=[{"role": "user", "content": user_msg}],
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return _error_result(lead, "Enrichment timed out")
    except APIError as e:
        log.exception("Anthropic API error for lead %s", lead.name)
        return _error_result(lead, f"API error: {e}")
    except Exception as e:
        log.exception("Unexpected error for lead %s", lead.name)
        return _error_result(lead, f"Unexpected error: {e}")

    text = _extract_final_text(message)
    parsed = _extract_json_object(text)

    if not parsed:
        return _error_result(
            lead,
            "Model did not return parseable JSON",
            raw=text[:500] if text else None,
        )

    # Ensure identity fields are populated (model may omit them)
    parsed.setdefault("name", lead.name)
    parsed.setdefault("company", lead.company)
    parsed.setdefault("website", lead.website)
    parsed.setdefault("designation", lead.designation)

    try:
        return EnrichmentResult(**parsed)
    except Exception as e:
        log.warning("Schema validation failed for %s: %s", lead.name, e)
        return _error_result(
            lead,
            f"Schema validation failed: {e}",
            raw=json.dumps(parsed)[:500],
        )


def _error_result(lead: Lead, msg: str, raw: Optional[str] = None) -> EnrichmentResult:
    """Build a graceful failure result so the UI still has a row."""
    return EnrichmentResult(
        name=lead.name,
        company=lead.company,
        website=lead.website,
        designation=lead.designation,
        company_summary="(enrichment failed)",
        person_summary="(enrichment failed)",
        signals=[],
        score=0,
        score_breakdown=ScoreBreakdown(relevance=0, urgency=0, time_to_close=0),
        justification=f"Could not enrich: {msg}" + (f" | raw: {raw}" if raw else ""),
        sources=[],
        confidence="low",
        error=msg,
    )


async def enrich_leads(
    client: AsyncAnthropic,
    leads: list[Lead],
    icp_text: str,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    progress_callback=None,
) -> list[EnrichmentResult]:
    """Enrich a batch of leads with bounded concurrency.

    progress_callback: optional callable invoked as
        progress_callback(done_count, total, latest_result)
    after each lead completes (in completion order, not input order).
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(leads)
    done = 0
    results: list[Optional[EnrichmentResult]] = [None] * total

    async def _run(idx: int, lead: Lead):
        nonlocal done
        async with sem:
            res = await enrich_lead(client, lead, icp_text, model=model)
        results[idx] = res
        done += 1
        if progress_callback:
            try:
                progress_callback(done, total, res)
            except Exception:
                log.exception("progress_callback raised")

    await asyncio.gather(*(_run(i, lead) for i, lead in enumerate(leads)))
    # All slots are filled by gather completion
    return [r for r in results if r is not None]
