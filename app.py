"""
Khazanah Annual Review 2026 — RAG Query Interface
Streamlit frontend that talks to the FastAPI backend.
"""

import streamlit as st
import httpx
import pandas as pd
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Khazanah Annual Review 2026",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://127.0.0.1:8000"

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E5F9E 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #CBD5E1; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .confidence-high   { background:#D1FAE5; color:#065F46; padding:3px 10px;
                         border-radius:12px; font-size:0.8rem; font-weight:600; }
    .confidence-medium { background:#FEF3C7; color:#92400E; padding:3px 10px;
                         border-radius:12px; font-size:0.8rem; font-weight:600; }
    .confidence-low    { background:#FEE2E2; color:#991B1B; padding:3px 10px;
                         border-radius:12px; font-size:0.8rem; font-weight:600; }
    .confidence-no_context { background:#F1F5F9; color:#475569; padding:3px 10px;
                              border-radius:12px; font-size:0.8rem; font-weight:600; }

    .source-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2E5F9E;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .answer-box {
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        line-height: 1.7;
    }
    .warning-box {
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #9A3412;
    }
    .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def check_health() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def query_api(query: str, top_k: int, full_context: bool) -> dict | None:
    try:
        r = httpx.post(
            f"{API_BASE}/query",
            json={"query": query, "top_k": top_k, "full_context_mode": full_context},
            timeout=60
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_extracted_data() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/extract", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def confidence_badge(level: str) -> str:
    labels = {
        "high":       ("HIGH CONFIDENCE",   "confidence-high"),
        "medium":     ("MEDIUM CONFIDENCE", "confidence-medium"),
        "low":        ("LOW CONFIDENCE",    "confidence-low"),
        "no_context": ("NOT IN DOCUMENT",   "confidence-no_context"),
    }
    label, css = labels.get(level, ("UNKNOWN", "confidence-low"))
    return f'<span class="{css}">{label}</span>'


def confidence_warning(level: str) -> str | None:
    warnings = {
        "low": (
            "⚠️ **Low confidence** — The retrieved slides may not directly answer "
            "this question. Verify against the source document before sharing."
        ),
        "no_context": (
            "🚫 **Not found in document** — This information does not appear in the "
            "Annual Review. Do not present this as a Khazanah-sourced answer."
        ),
    }
    return warnings.get(level)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/"
        "Coat_of_arms_of_Malaysia.svg/200px-Coat_of_arms_of_Malaysia.svg.png",
        width=60
    )
    st.markdown("### Khazanah RAG System")
    st.markdown("---")

    # Health status
    health = check_health()
    if health and health.get("pdf_ingested"):
        st.success(f"✅ System healthy\n\n{health['chunk_count']} slides loaded")
    else:
        st.error("❌ Backend not reachable or PDF not ingested")

    st.markdown("---")

    # Query settings
    st.markdown("**Query Settings**")
    top_k = st.slider(
        "Slides to retrieve (top-k)",
        min_value=1, max_value=10, value=5,
        help="How many slides to search before generating an answer"
    )
    full_context_mode = st.toggle(
        "Full Context Mode",
        value=False,
        help=(
            "Send the entire document to the AI instead of just retrieved slides. "
            "More accurate but uses more API tokens."
        )
    )

    if full_context_mode:
        st.info("📄 Full context mode ON — entire document sent to AI")

    st.markdown("---")

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    sample_questions = [
        "What was Khazanah's net asset value in 2025?",
        "What were the investment returns by asset class?",
        "What is MAHB privatisation deal value?",
        "Summarise Khazanah's sustainability initiatives",
        "What are Khazanah's 2025 financial highlights?",
        "Which sectors did Khazanah invest in?",
        "What was Malaysia's GDP growth in 2025?",
        "What is Dana Impak and what does it do?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["prefill_query"] = q

    st.markdown("---")
    st.markdown(
        "<small>Built for Khazanah Data Associate Assessment · "
        "GPT-4o-mini · ChromaDB · FastAPI</small>",
        unsafe_allow_html=True
    )


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🇲🇾 Khazanah Annual Review 2026</h1>
  <p>AI-powered Q&A · Ask questions about Khazanah's portfolio, performance, and strategy</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_chat, tab_data, tab_about = st.tabs([
    "💬 Ask Questions",
    "📊 Extracted Data",
    "ℹ️ About this System"
])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    f'<div class="answer-box">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                # Show confidence badge
                if "confidence" in msg:
                    st.markdown(
                        confidence_badge(msg["confidence"]),
                        unsafe_allow_html=True
                    )
                # Show warning if needed
                warning = confidence_warning(msg.get("confidence", ""))
                if warning:
                    st.markdown(
                        f'<div class="warning-box">{warning}</div>',
                        unsafe_allow_html=True
                    )
                # Show sources
                if msg.get("sources"):
                    with st.expander(
                        f"📄 Sources ({len(msg['sources'])} slides referenced)",
                        expanded=False
                    ):
                        for src in msg["sources"]:
                            score_pct = int(src["relevance_score"] * 100)
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>Slide {src["page_number"]}:</strong> '
                                f'{src["slide_title"]}<br>'
                                f'<small>Relevance: {score_pct}% · '
                                f'Type: {src["slide_type"]} · '
                                f'Extracted via: {src["extraction_method"]}</small><br>'
                                f'<em>{src["text_preview"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.markdown(msg["content"])

    # Handle prefilled query from sidebar buttons
    prefill = st.session_state.pop("prefill_query", None)

    # Chat input
    user_input = st.chat_input(
        "Ask anything about Khazanah's Annual Review 2026...",
    )

    # Use prefill if no direct input
    query = user_input or prefill

    if query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching Annual Review..."):
                result = query_api(query, top_k, full_context_mode)

            if result and "error" not in result:
                answer = result.get("answer", "No answer returned.")
                confidence = result.get("confidence", "low")
                sources = result.get("sources", [])
                full_ctx_used = result.get("full_context_mode_used", False)

                # Display answer
                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True
                )

                # Confidence badge
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.markdown(
                        confidence_badge(confidence),
                        unsafe_allow_html=True
                    )
                with col2:
                    if full_ctx_used:
                        st.caption("📄 Full context mode used")

                # Warning for low/no confidence
                warning = confidence_warning(confidence)
                if warning:
                    st.markdown(
                        f'<div class="warning-box">{warning}</div>',
                        unsafe_allow_html=True
                    )

                # Sources expander
                if sources:
                    with st.expander(
                        f"📄 Sources ({len(sources)} slides referenced)",
                        expanded=False
                    ):
                        for src in sources:
                            score_pct = int(src["relevance_score"] * 100)
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>Slide {src["page_number"]}:</strong> '
                                f'{src["slide_title"]}<br>'
                                f'<small>Relevance: {score_pct}% · '
                                f'Type: {src["slide_type"]} · '
                                f'Method: {src["extraction_method"]}</small><br>'
                                f'<em>{src["text_preview"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "sources": sources
                })

            else:
                error_msg = result.get("error", "Unknown error") if result else "Backend unreachable"
                st.error(f"❌ Error: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {error_msg}",
                    "confidence": "no_context",
                    "sources": []
                })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()


# ── Tab 2: Extracted Data ─────────────────────────────────────────────────────

with tab_data:
    st.markdown("### 📊 Structured Data Extracted from Annual Review")
    st.caption(
        "This data was extracted during ingestion using GPT-4o-mini vision. "
        "Values from charts are approximate — verify against source slides."
    )

    data = get_extracted_data()

    if data:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Financial Metrics Extracted",
                len(data.get("key_financial_metrics", []))
            )
        with col2:
            st.metric(
                "Portfolio Companies Identified",
                len(data.get("portfolio_companies", []))
            )

        st.markdown("---")

        # Financial metrics table
        metrics = data.get("key_financial_metrics", [])
        if metrics:
            st.markdown("#### 💰 Key Financial Metrics")
            df_metrics = pd.DataFrame(metrics)

            # Reorder and rename columns for display
            display_cols = ["metric_name", "value", "unit", "year", "context"]
            display_cols = [c for c in display_cols if c in df_metrics.columns]
            df_metrics = df_metrics[display_cols]
            df_metrics.columns = [
                c.replace("_", " ").title() for c in display_cols
            ]

            st.dataframe(
                df_metrics,
                use_container_width=True,
                hide_index=True,
                height=400
            )

            # Download button
            csv = df_metrics.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Metrics CSV",
                csv,
                "khazanah_metrics_2026.csv",
                "text/csv"
            )

        # Portfolio companies table
        companies = data.get("portfolio_companies", [])
        if companies:
            st.markdown("#### 🏢 Portfolio Companies")
            df_companies = pd.DataFrame(companies)
            display_cols = [c for c in ["name", "sector", "ownership_stake", "notes", "page_number"] if c in df_companies.columns]
            df_companies = df_companies[display_cols]
            df_companies.columns = [c.replace("_", " ").title() for c in display_cols]

            st.dataframe(
                df_companies,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "Portfolio companies were not separately extracted during ingestion. "
                "Ask the chat: 'What companies does Khazanah hold?' to get this from RAG."
            )
    else:
        st.error("Could not load extracted data. Make sure the backend is running.")


# ── Tab 3: About ──────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("### ℹ️ About This System")

    st.markdown("""
    This tool allows you to interactively query **Khazanah Nasional Berhad's Annual Review 2026**
    using AI. It was built as part of a technical assessment for the Data Associate / AVP role.

    ---

    #### 🏗️ How it works

    1. **Ingestion Pipeline** — The Annual Review PDF is parsed in two passes:
       - *Pass 1 (PyMuPDF):* Extracts the native text layer from each slide
       - *Pass 2 (GPT-4o-mini Vision):* Renders each page as an image and extracts
         structured data including charts, tables, and infographics

    2. **Embeddings** — Each slide's content is converted into a vector representation
       using OpenAI's `text-embedding-3-small` model and stored in ChromaDB

    3. **RAG Query Engine** — When you ask a question:
       - Your question is converted to a vector
       - The most relevant slides are retrieved by similarity
       - GPT-4o-mini generates an answer using only those slides as context
       - Source citations and confidence scores are returned with every answer

    4. **Full Context Mode** — Optionally, the entire document (all 23 slides) can be
       sent directly to the AI, bypassing retrieval for maximum accuracy

    ---

    #### ⚠️ Known Limitations

    - **Chart accuracy:** Values extracted from bar charts and graphs are approximate.
      Vision models achieve ~35-50% accuracy on chart value extraction.
      Always verify chart-derived numbers against the original PDF.

    - **Presentation deck:** The source document is a 23-slide presentation, not a
      full text report. Some detail present in a full annual report may not be here.

    - **Confidence scores:** High confidence means the retrieved slides are semantically
      similar to your query — it does not guarantee the answer is correct.
      Always cite the source document, not this tool, in formal communications.

    ---

    #### 🛠️ Tech Stack

    | Component | Tool |
    |-----------|------|
    | PDF Parsing | PyMuPDF + GPT-4o-mini Vision |
    | Embeddings | OpenAI text-embedding-3-small |
    | Vector Store | ChromaDB (local, persistent) |
    | LLM | GPT-4o-mini |
    | Backend | FastAPI + Uvicorn |
    | Frontend | Streamlit |
    """)