"""Account Intelligence & Lead Scoring Platform — Streamlit UI.

Run with:
    streamlit run app.py

Requires ANTHROPIC_API_KEY in the environment (or in a .env file).
"""
from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from queue import Queue, Empty

import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from enrichment.agent import enrich_leads, DEFAULT_MODEL
from enrichment.schema import EnrichmentResult
from utils.csv_io import parse_csv, results_to_dataframe, results_to_json
from utils.icp_store import load_icp, save_icp

load_dotenv()

st.set_page_config(
    page_title="Account Intelligence & Lead Scoring",
    page_icon="🎯",
    layout="wide",
)

# ---------- Session state ----------
if "leads" not in st.session_state:
    st.session_state.leads = []
if "results" not in st.session_state:
    st.session_state.results = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "icp_text" not in st.session_state:
    st.session_state.icp_text = load_icp()
if "is_running" not in st.session_state:
    st.session_state.is_running = False


# ---------- Header ----------
st.title("🎯 Account Intelligence & Lead Scoring")
st.caption(
    "Upload leads → Claude researches each company & person via "
    "`web_search` + `web_fetch` → scores them against your ICP."
)

# ---------- Sidebar: config ----------
with st.sidebar:
    st.header("Configuration")

    api_key_present = bool(os.getenv("ANTHROPIC_API_KEY"))
    if api_key_present:
        st.success("✓ ANTHROPIC_API_KEY found in env")
    else:
        st.error("⚠ ANTHROPIC_API_KEY not set. Add it to a `.env` file.")

    model = st.selectbox(
        "Model",
        options=["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
        index=0,
        help="Sonnet 4.6 is the default — best cost/capability balance for "
        "the agentic research loop. Opus 4.7 is more thorough but slower "
        "and more expensive. Haiku 4.5 is fastest/cheapest but may miss nuance.",
    )

    concurrency = st.slider(
        "Concurrent leads",
        min_value=1,
        max_value=10,
        value=4,
        help="How many leads to process in parallel. Higher = faster but "
        "more rate-limit risk.",
    )

    st.divider()
    st.caption(
        "Server tools used: `web_search_20260209` (max 4 uses/lead), "
        "`web_fetch_20260209` (max 3 uses/lead). Both run on Anthropic's "
        "infrastructure — no scraping setup required."
    )


# ---------- ICP editor ----------
st.subheader("1. Your Ideal Customer Profile (ICP)")
st.caption(
    "This is the ready-reckoner Claude uses to score each lead. "
    "Edit and click 'Save ICP' to persist between sessions."
)

icp_text = st.text_area(
    "ICP definition (Markdown supported)",
    value=st.session_state.icp_text,
    height=300,
    label_visibility="collapsed",
)

col_save, col_reset, _ = st.columns([1, 1, 4])
with col_save:
    if st.button("💾 Save ICP", use_container_width=True):
        save_icp(icp_text)
        st.session_state.icp_text = icp_text
        st.success("ICP saved.")
with col_reset:
    if st.button("↺ Reload saved", use_container_width=True):
        st.session_state.icp_text = load_icp()
        st.rerun()


# ---------- CSV upload ----------
st.subheader("2. Upload your leads CSV")
st.caption(
    "Required column: **Name**. Optional: **Website**, **Designation**, "
    "**Company**. Extra columns are passed along as context."
)

uploaded = st.file_uploader(
    "Drop CSV here",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded is not None:
    try:
        leads, warns = parse_csv(uploaded.read())
        st.session_state.leads = leads
        st.session_state.warnings = warns
        st.success(f"Parsed {len(leads)} lead(s).")
        if warns:
            with st.expander(f"⚠ {len(warns)} warning(s)"):
                for w in warns:
                    st.text(w)
    except ValueError as e:
        st.error(str(e))

if st.session_state.leads:
    with st.expander(f"Preview ({len(st.session_state.leads)} leads)"):
        preview_df = pd.DataFrame(
            [
                {
                    "Name": l.name,
                    "Company": l.company or "",
                    "Designation": l.designation or "",
                    "Website": l.website or "",
                }
                for l in st.session_state.leads[:50]
            ]
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


# ---------- Run enrichment ----------
st.subheader("3. Run enrichment")

run_disabled = (
    not api_key_present
    or not st.session_state.leads
    or st.session_state.is_running
)

run_col, status_col = st.columns([1, 4])
with run_col:
    run_clicked = st.button(
        "🚀 Enrich leads",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
    )

# Enrichment runs in a background thread so Streamlit can show live progress
# without blocking the main script. We use a thread-safe queue to pipe
# per-lead results back into the UI.
def _run_in_thread(
    leads, icp, model, concurrency, result_queue: Queue
):
    """Run the async enrichment loop in a fresh event loop on this thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = AsyncAnthropic()  # picks up ANTHROPIC_API_KEY from env

    def _cb(done, total, latest: EnrichmentResult):
        result_queue.put(("progress", done, total, latest))

    try:
        results = loop.run_until_complete(
            enrich_leads(
                client=client,
                leads=leads,
                icp_text=icp,
                model=model,
                concurrency=concurrency,
                progress_callback=_cb,
            )
        )
        result_queue.put(("done", results))
    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        loop.close()


if run_clicked:
    st.session_state.is_running = True
    st.session_state.results = []

    queue: Queue = Queue()
    thread = threading.Thread(
        target=_run_in_thread,
        args=(
            list(st.session_state.leads),
            icp_text,
            model,
            concurrency,
            queue,
        ),
        daemon=True,
    )
    thread.start()

    progress = st.progress(0.0, text="Starting enrichment...")
    live_results: list[EnrichmentResult] = []
    total = len(st.session_state.leads)

    # Poll the queue while the thread runs
    while True:
        try:
            msg = queue.get(timeout=0.5)
        except Empty:
            if not thread.is_alive():
                # Thread died without sending a 'done' or 'error' — shouldn't
                # happen, but break to avoid hang.
                break
            continue

        kind = msg[0]
        if kind == "progress":
            _, done, total_, latest = msg
            live_results.append(latest)
            progress.progress(
                done / total_,
                text=f"Enriched {done}/{total_}: {latest.name} → score {latest.score}",
            )
        elif kind == "done":
            st.session_state.results = msg[1]
            progress.progress(1.0, text=f"✓ Done — {total} leads enriched.")
            break
        elif kind == "error":
            st.error(f"Enrichment failed: {msg[1]}")
            break

    thread.join(timeout=5)
    st.session_state.is_running = False
    st.rerun()


# ---------- Dashboard ----------
if st.session_state.results:
    st.subheader("4. Dashboard")

    results = st.session_state.results
    df = results_to_dataframe(results)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total leads", len(results))
    if not df.empty:
        m2.metric("Avg score", f"{df['Score'].mean():.0f}")
        m3.metric("High-fit (≥75)", int((df["Score"] >= 75).sum()))
        m4.metric("Failures", int((df["Error"] != "").sum()))

    # Filters
    fc1, fc2, _ = st.columns([1, 1, 3])
    with fc1:
        min_score = st.slider("Min score", 0, 100, 0)
    with fc2:
        conf_filter = st.multiselect(
            "Confidence",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"],
        )

    view = df[(df["Score"] >= min_score) & (df["Confidence"].isin(conf_filter))]

    # Color-band the Score column
    def _score_color(v):
        try:
            v = int(v)
        except (ValueError, TypeError):
            return ""
        if v >= 75:
            return "background-color: #c8e6c9"  # green
        if v >= 50:
            return "background-color: #fff9c4"  # yellow
        if v > 0:
            return "background-color: #ffcdd2"  # red
        return "background-color: #eeeeee"

    styled = view.style.map(_score_color, subset=["Score"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%d"),
            "Relevance": st.column_config.NumberColumn(format="%d"),
            "Urgency": st.column_config.NumberColumn(format="%d"),
            "Time-to-Close": st.column_config.NumberColumn(format="%d"),
            "Company Summary": st.column_config.TextColumn(width="medium"),
            "Person Summary": st.column_config.TextColumn(width="medium"),
            "Justification": st.column_config.TextColumn(width="large"),
            "Sources": st.column_config.TextColumn(width="medium"),
        },
    )

    # Drill-down
    if not view.empty:
        with st.expander("🔍 Drill into a single lead"):
            names = view["Name"].tolist()
            pick = st.selectbox("Pick a lead", names)
            row = view[view["Name"] == pick].iloc[0]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"### {row['Name']}")
                st.markdown(f"**{row['Designation']}** at **{row['Company']}**")
                if row["Website"]:
                    st.markdown(f"🌐 {row['Website']}")
                st.markdown(f"**Score: {row['Score']}** ({row['Confidence']} confidence)")
                st.progress(int(row["Score"]) / 100)
                st.caption(
                    f"Relevance {row['Relevance']} · "
                    f"Urgency {row['Urgency']} · "
                    f"Time-to-Close {row['Time-to-Close']}"
                )
            with cc2:
                st.markdown("**Company**")
                st.write(row["Company Summary"])
                st.markdown("**Person**")
                st.write(row["Person Summary"])

            st.markdown("**Justification**")
            st.info(row["Justification"])

            if row["Signals"]:
                st.markdown("**Signals**")
                for s in row["Signals"].split(" | "):
                    st.markdown(f"- {s}")
            if row["Sources"]:
                st.markdown("**Sources**")
                for s in row["Sources"].split(" | "):
                    st.markdown(f"- {s}")

    # Export
    st.subheader("5. Export")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_str = results_to_json(results)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇ Download enriched CSV",
            data=csv_bytes,
            file_name=f"enriched_leads_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "⬇ Download JSON",
            data=json_str,
            file_name=f"enriched_leads_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )
