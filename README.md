# Account Intelligence & Lead Scoring Platform

A functional prototype that takes a CSV of leads, uses Claude with native
`web_search` + `web_fetch` server tools to research each company and
contact, and scores them against your Ideal Customer Profile (ICP).

## What it does

1. You define your ICP once (industries, company size, target titles,
   pain points you solve, disqualifiers). It's persisted to
   `storage/icp.json`.
2. You upload a CSV with `Name`, `Website`, `Designation` (and any extras).
3. For each lead, Claude:
   - Searches the web for the company and the person
   - Fetches the most relevant pages directly
   - Synthesizes a 2-4 sentence company summary and a person priorities/pain
     summary
   - Scores on **Relevance**, **Urgency**, and **Time-to-Close** (each 0-100)
     plus a weighted overall score, with a written justification
4. You browse the dashboard (color-coded by score, filterable, drill-down
   per lead) and export to CSV or JSON.

All research is server-side. There's no scraping infrastructure to manage
— Anthropic executes `web_search` and `web_fetch` and returns the final
synthesized result.

## Stack

- **Language:** Python 3.10+
- **UI:** Streamlit
- **LLM:** Anthropic API, default model `claude-sonnet-4-6`
- **Tools:** `web_search_20260209` + `web_fetch_20260209` (both GA, dynamic
  filtering enabled — verified against `docs.claude.com` on 2026-05-19)
- **Concurrency:** `asyncio` with a configurable semaphore (default 4
  parallel leads)

## File layout

```
account-intel-platform/
├── app.py                       # Streamlit UI
├── enrichment/
│   ├── agent.py                 # Claude agentic loop
│   ├── prompts.py               # System + per-lead prompts
│   └── schema.py                # Pydantic models for structured output
├── utils/
│   ├── csv_io.py                # CSV parse + result-to-DataFrame/JSON
│   └── icp_store.py             # ICP persistence
├── storage/
│   └── icp.json                 # (created on first save)
├── sample_leads.csv             # 5 sample leads for testing
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# then edit .env and put your real ANTHROPIC_API_KEY

# 3. Run
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`.

## Usage

1. **Edit your ICP** in the text area (Markdown is fine). Click **Save ICP**
   to persist for future sessions.
2. **Upload a CSV.** It must have a `Name` column. `Website`,
   `Designation`, and `Company` are recognized too, plus common aliases
   (Title, URL, etc.). Extra columns are passed to Claude as context.
3. Click **Enrich leads.** The progress bar updates per lead as the
   agentic loop completes for each one.
4. **Browse the dashboard** — filter by score / confidence, drill into any
   lead to see full summaries, signals, and sources.
5. **Export** to CSV or JSON.

## CSV format

Required:

| Column | Aliases recognized |
| ------ | ------------------ |
| Name   | full name, contact, lead name, person |

Optional:

| Column      | Aliases |
| ----------- | ------- |
| Website     | url, domain, company website, site |
| Designation | title, job title, role, position |
| Company     | company name, organization, org, account |

If `Company` is missing but `Website` is present, the company name is
derived from the domain (e.g. `acme.io` → `Acme`).

Headers are case-insensitive. Any extra columns are forwarded to Claude
as `additional_csv_fields` in the lead context.

## Design notes & caveats

- **Why no manual `tool_use` loop?** `web_search` and `web_fetch` are
  server-executed tools. Anthropic runs them on its infrastructure and
  returns the final assistant message — we don't have to handle the
  `tool_use → tool_result` round-trip ourselves.
- **Per-lead tool budget:** capped at 4 searches + 3 fetches in the tool
  definitions. This is enough for company + person research without
  runaway cost.
- **Structured output:** the prompt instructs JSON-only output, and we
  defensively strip fences / locate the outermost `{...}` if the model
  adds preamble. Anthropic also supports `strict: true` structured
  outputs — a future enhancement would be to migrate to that for
  stronger guarantees.
- **Failures degrade gracefully:** if a lead can't be parsed or the model
  errors, the row appears in the dashboard with score 0, `confidence:
  low`, and the failure reason in `Error`.
- **Cost:** each lead is roughly 1 API call with up to 7 server-tool
  invocations. With Sonnet 4.6 and a typical ICP, budget for tens of
  cents per lead. Web search is billed at $10 per 1,000 searches (per
  Anthropic's pricing page) plus normal token costs. Verify current
  pricing at https://www.anthropic.com/pricing before running large
  batches.
- **Rate limits:** the default concurrency of 4 is conservative. If you
  hit 429s, lower it in the sidebar.

## Things I deliberately did not build (out of scope for a prototype)

- Persistent run cache (re-running the same CSV re-enriches every lead).
  Easy to add: hash `(name, company, website)` and cache results in
  `storage/runs/`.
- Auth, multi-tenancy, multi-user ICPs.
- A separate "qualification call" agent that does deeper, multi-turn
  research on high-score leads.
- CRM integration (HubSpot/Salesforce export).
