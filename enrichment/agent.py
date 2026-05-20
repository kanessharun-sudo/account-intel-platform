"""Claude agentic loop for lead enrichment.

Uses Anthropic's server-side tools:
- web_search_20260209  (GA, dynamic filtering)
- web_fetch_20260209   (GA, dynamic filtering)

Both execute on Anthropic's infrastructure — no scraping infra required.
We just call messages.create() and Claude handles the multi-step research
internally, returning a final assistant message.

NOTE: Tool version strings verified against docs.claude.com on
2026-05-19. If Anthropic rotates versions, update TOOL_DEFINITIONS below.

Reliability features (added 2026-05-19 based on production feedback):
- Exponential backoff retry on 429 (rate limit) and 529 (overloaded)
- One retry on JSON parse failure with a stricter reminder
- Longer per-lead timeout (240s)
- Per-lead 'started' progress events for better UI feedback
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Optional

from anthropic import AsyncAnthropic, APIError

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import EnrichmentResult, Lead, ScoreBreakdown

log = logging.getLogger(__name__)

# Server-side tools provided by Anthropic. The version suffixes are
# capability-keyed (the 20260209 versions add dynamic filtering — Claude
# post-processes results in a sandbox before they hit context).
#
# allowed_callers=["direct"] is REQUIRED for Haiku 4.5 (it doesn't support
# programmatic tool calling) and harmless for Sonnet/Opus, so we set it
# unconditionally for cross-model compatibility.
TOOL_DEFINITIONS = [
    {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 4,  # cap per request
        "allowed_callers": ["direct"],
    },
    {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
        "max_uses": 3,
        "citations": {"enabled": True},
        "allowed_callers": ["direct"],
    },
]

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 420.0  # 7 min — covers up to 2 long rate-limit waits + research

# Retry config for transient API failures (429 rate limit, 529 overloaded)
MAX_API_RETRIES = 3
# Backoff schedules differ by error type:
# - 429 rate-limit errors: the per-minute window is 60s, so retrying sooner
#   just hits the same wall. Wait long enough for the window to reset.
# - 529/503/502 (server-side overloads): exponential backoff is appropriate.
RATE_LIMIT_BACKOFF_SECONDS = [65.0, 95.0, 125.0]  # ~1 min, ~1.5 min, ~2 min
OVERLOAD_BACKOFF_SECONDS = [5.0, 15.0, 30.0]      # exponential


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


def _is_retryable_api_error(err: Exception) -> bool:
    """Decide whether an APIError is worth retrying (rate limit / overload)."""
    # The Anthropic SDK exposes status_code on APIError; fall back to text match
    status = getattr(err, "status_code", None)
    if status in (429, 529, 503, 502):
        return True
    msg = str(err).lower()
    return any(x in msg for x in ("rate limit", "overload", "529", "429", "503"))


def _is_rate_limit_error(err: Exception) -> bool:
    """True for 429 rate-limit errors specifically (need long waits)."""
    if getattr(err, "status_code", None) == 429:
        return True
    msg = str(err).lower()
    return "429" in msg or "rate_limit_error" in msg or "rate limit" in msg


def _backoff_for(err: Exception, attempt: int) -> float:
    """Pick a wait duration in seconds given the error type and attempt index."""
    schedule = (
        RATE_LIMIT_BACKOFF_SECONDS
        if _is_rate_limit_error(err)
        else OVERLOAD_BACKOFF_SECONDS
    )
    idx = min(attempt, len(schedule) - 1)
    base = schedule[idx]
    # ±20% jitter so concurrent retries don't synchronize
    return base * (0.8 + random.random() * 0.4)


async def _call_with_retries(
    client: AsyncAnthropic,
    *,
    model: str,
    system: str,
    tools: list,
    messages: list,
    max_tokens: int,
    timeout: float,
) -> object:
    """Call messages.create with backoff on transient failures.

    Uses long waits (~60-120s) for 429 rate-limit errors since the per-minute
    window must reset, and shorter exponential backoff for 5xx overloads.
    Returns the message on success; raises the last exception on final failure.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES + 1):  # initial try + retries
        try:
            return await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                ),
                timeout=timeout,
            )
        except APIError as e:
            last_exc = e
            if not _is_retryable_api_error(e) or attempt == MAX_API_RETRIES:
                raise
            wait = _backoff_for(e, attempt)
            err_kind = "rate-limit" if _is_rate_limit_error(e) else "overload"
            log.warning(
                "Retryable %s error (attempt %d/%d), waiting %.1fs: %s",
                err_kind, attempt + 1, MAX_API_RETRIES, wait,
                str(e)[:200],
            )
            await asyncio.sleep(wait)
        except asyncio.TimeoutError:
            # Timeouts are not retryable here — they consume the whole budget
            raise
    # Unreachable, but satisfies the type checker
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


async def enrich_lead(
    client: AsyncAnthropic,
    lead: Lead,
    icp_text: str,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
) -> EnrichmentResult:
    """Run the agentic enrichment loop for a single lead.

    Because web_search and web_fetch are server-side tools, we don't have
    to implement the tool_use -> tool_result loop ourselves. Anthropic's
    infrastructure executes the searches and feeds results back to the
    model before returning the final response. From our side, this is a
    single messages.create() call (plus retries on transient failure).
    """
    user_msg = build_user_prompt(lead, icp_text)

    # First attempt: full research + JSON output
    try:
        message = await _call_with_retries(
            client,
            model=model,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=MAX_TOKENS,
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

    # JSON parse retry: ask the model once more to output JSON only,
    # passing back what it said so it can self-correct.
    if not parsed:
        log.info("First-pass JSON parse failed for %s, retrying once.", lead.name)
        retry_user_msg = (
            "Your previous response could not be parsed as JSON. "
            "Re-emit your final answer as a SINGLE JSON object matching the "
            "schema, with no surrounding prose, no markdown fences, no "
            "commentary — just the JSON. "
            f"\n\nOriginal request was:\n\n{user_msg}"
        )
        try:
            # No tools on the retry — research is done, we just want the JSON
            retry_msg = await _call_with_retries(
                client,
                model=model,
                system=SYSTEM_PROMPT,
                tools=[],
                messages=[
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": text or "(no output)"},
                    {"role": "user", "content": retry_user_msg},
                ],
                max_tokens=MAX_TOKENS,
                timeout=60.0,  # no research, so much shorter
            )
            retry_text = _extract_final_text(retry_msg)
            parsed = _extract_json_object(retry_text)
        except Exception as e:
            log.warning("JSON retry failed for %s: %s", lead.name, e)

    if not parsed:
        return _error_result(
            lead,
            "Model did not return parseable JSON (after retry)",
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
    concurrency: int = 3,  # lowered default from 4 — gentler on rate limits
    progress_callback=None,
) -> list[EnrichmentResult]:
    """Enrich a batch of leads with bounded concurrency.

    progress_callback: optional callable invoked as
        progress_callback(event_type, done_count, total, payload)
      where event_type is 'started' or 'completed', and payload is the
      Lead (for 'started') or the EnrichmentResult (for 'completed').

    Backwards compatibility: if progress_callback only accepts 3 args
    (done, total, latest_result), we fall back to calling it that way
    on 'completed' events only.
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(leads)
    done = 0
    started = 0
    results: list[Optional[EnrichmentResult]] = [None] * total

    # Detect callback signature for backwards compatibility
    cb_supports_events = False
    if progress_callback is not None:
        try:
            import inspect
            sig = inspect.signature(progress_callback)
            cb_supports_events = len(sig.parameters) >= 4
        except (TypeError, ValueError):
            cb_supports_events = False

    def _notify(event: str, payload):
        nonlocal done, started
        if not progress_callback:
            return
        try:
            if cb_supports_events:
                count = done if event == "completed" else started
                progress_callback(event, count, total, payload)
            else:
                # Legacy 3-arg callback: only fire on completed
                if event == "completed":
                    progress_callback(done, total, payload)
        except Exception:
            log.exception("progress_callback raised")

    async def _run(idx: int, lead: Lead):
        nonlocal done, started
        async with sem:
            started += 1
            _notify("started", lead)
            res = await enrich_lead(client, lead, icp_text, model=model)
        results[idx] = res
        done += 1
        _notify("completed", res)

    await asyncio.gather(*(_run(i, lead) for i, lead in enumerate(leads)))
    return [r for r in results if r is not None]
