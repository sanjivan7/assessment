# Assessment

**Built for:** KNB Take-Home Assessment
**Author:** Sanjivan Balajawahar
**Submission date:** 10-Apr-2026

---

## What This Tool Does

This tool lets you ask plain English questions about Khazanah Nasional Berhad's
Annual Review 2026 and get accurate, sourced answers in seconds.

Instead of manually searching through a 23-slide presentation to find a specific
financial figure or investment detail, you type your question and the system finds
the most relevant slides, reads them, and writes a direct answer — telling you
exactly which slides it used.

**Example questions it can answer:**
- "What was Khazanah's net asset value in 2025?"
- "What were the investment returns by asset class?"
- "What is the MAHB privatisation deal value and who were the partners?"
- "Summarise Khazanah's Dana Impak initiatives"
- "What is Malaysia's GDP growth forecast for 2026?"

If the answer is not in the document, the system says so clearly rather than
guessing — this is by design.

---

## Architecture Overview

```

PDF (23 slides)
      │
      ├── Pass 1: PyMuPDF ──────────► Native text extraction (titles, bullet points, captions)
      └── Pass 2: GPT-4o-mini Vision (Image-based extraction) ► (Charts, tables, infographics)
                        │
                        ▼
              Combined chunk per slide
              (text + vision data merged)
                        │
                        ▼
            OpenAI text-embedding-3-small
          (converts each slide to a vector)
                        │
                        ▼
               ChromaDB — local vector store
              23 chunks stored with metadata
                        │
            ┌───────────┴────────────┐
            │                        │
         RAG mode                Full context mode
      → embed query            all 23 slides which is sent
     → top-k retrieval          directly to GPT-4o-mini (LLM)
     → find similar slides      
     → GPT-4o-mini answers      
     → return + cite slides     
            │                      
            ▼
      FastAPI Backend
  /health  /query  /extract
            │                      
            ▼
    Streamlit Frontend
 Chat UI + Data Viewer + About

**In plain English:** 
The document is broken into 23 pieces (one per slide).
Each piece is converted into a mathematical fingerprint (embedding) that captures
its meaning. When you ask a question, your question gets the same treatment, and
the system finds the slides whose fingerprints most closely match yours. Those
slides are handed to GPT-4o-mini, which reads them and writes your answer.

---

## Design Decisions and Trade-offs

### Why GPT-4o-mini for vision extraction?
The Annual Review is a presentation deck, not a text document. Most financial
data — portfolio returns, asset values, chart trends — is embedded in images,
charts, and tables that standard text extraction cannot read. GPT-4o-mini's
vision capability reads each slide as an image and extracts structured data.
Total cost for 23 slides: approximately $0.03.

**Trade-off:** Vision models achieve roughly 35–50% accuracy on chart value
extraction. Numbers extracted from bar charts should be treated as approximate.
Text-based tables and explicit numerical callouts are much more reliable (~90%+).

### Why text-embedding-3-small over free local models?
The leading free local model (all-MiniLM-L6-v2) has a hard 256-token limit.
For a slide with a detailed chart description, this silently truncates the most
important financial figures from the embedding — causing retrieval failures with
no error message. text-embedding-3-small supports 8,191 tokens and scores
~20% better on retrieval benchmarks. Total cost for this corpus: $0.0003
(three hundredths of a cent).

### Why ChromaDB over a simpler numpy approach?
At 23 chunks, ChromaDB performs brute-force cosine similarity — mathematically
identical to numpy. ChromaDB was chosen because it stores documents, embeddings,
and metadata together in a clean API, and persists to disk so the ingestion step
only needs to run once. The abstraction cost is justified by the cleaner code.

### Why Streamlit over Next.js?
The assessment allows either. For a 48-hour solo sprint, Streamlit's native
`st.chat_message` and `st.chat_input` components produce a working chat interface
in ~30 lines versus 6–8 hours for an equivalent Next.js implementation. The
FastAPI backend is fully decoupled — swapping Streamlit for Next.js requires
only a new frontend, no backend changes.

### Why RAG + full-context mode?
The entire corpus is ~6,000 tokens — less than 5% of GPT-4o-mini's 128K context
window. This means full-context mode (sending all 23 slides directly) is feasible
and sometimes more accurate than retrieval, particularly for synthesis questions.
RAG mode is kept as the default because it provides natural slide-level citation
that full-context mode does not.

### Why no LangChain or LlamaIndex?
LangChain introduces ~2.7x token overhead per query versus raw implementation
and produces difficult-to-debug abstractions. For a 23-slide corpus with a
straightforward retrieval pattern, the raw implementation is faster to build,
easier to debug, and fully transparent.

---

## Known Limitations

**1. Source document is a presentation deck, not a full annual report.**
The 23-slide deck contains high-level summaries. Granular data present in a
full text annual report (footnotes, detailed financial statements, complete
portfolio tables) is not present. The system can only answer what is in the slides.

**2. Chart value extraction is approximate.**
GPT-4o-mini reads charts as images. Bar chart values, trend lines, and
infographic numbers are estimated visually and may be off by small amounts.
Always verify chart-derived figures against the original PDF slide.

**3. Retrieval misses on specific entity queries.**
The system occasionally retrieves semantically similar but factually adjacent
slides. Example: asking for MAHB privatisation partners retrieved the right
financial figure but missed the partner logos visible on slide 13. Mitigation:
use Full Context Mode for questions about specific named entities or transactions.

**4. Source citations in refusal messages.**
When the system cannot find an answer, it still lists slide numbers in the
"Sources" field. These slides were retrieved as the closest matches but did not
contain the answer — they should not be interpreted as containing the information.

**5. No authentication.**
This is a local POC. The API has no authentication layer. Not suitable for
production deployment without adding auth.

**6. Single document.**
The system is built for the 2026 Annual Review only. Multi-year comparison
(2025 vs 2024 vs 2023) would require re-ingesting multiple PDFs with year
metadata tagging — a straightforward extension but not implemented here.

---

## What I Would Do With More Time

- **RAGAS evaluation:** Build a test set of 20–30 known Q&A pairs from the
  document and measure faithfulness, answer relevance, and context precision
  systematically rather than manually
- **Hybrid retrieval:** Add BM25 keyword search alongside vector similarity.
  For precise entity queries ("MAHB partners"), keyword matching outperforms
  pure semantic search
- **Multi-year comparison:** Ingest 2024 and 2023 Annual Reviews with year
  metadata, enabling questions like "How did NAV change from 2023 to 2025?"
- **Next.js frontend:** Replace Streamlit with a proper Next.js interface using
  the Vercel AI SDK for streaming responses and a more polished analyst UX
- **Caching:** Redis cache for repeated queries — the same financial figure
  question asked twice should not trigger two LLM calls
- **Re-ingestion on document update:** A `/ingest` API endpoint (currently
  script-only) with proper job queuing so non-technical users can trigger
  pipeline updates

---

## Setup Instructions

### Prerequisites
- Python 3.11
- Git
- An OpenAI API key (free tier is sufficient — total cost ~$0.15)

### Step 1 — Clone and set up environment
```bash
git clone <your-repo-url>
cd Sanjivan_Khazanah
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Step 2 — Add API keys
Create a `.env` file in the project root:
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-gemini-key-here

### Step 3 — Add the PDF
Download Khazanah's Annual Review 2026 from:
`https://www.khazanah.com.my/media-downloads/khazanah-annual-review/`

Place it in the `docs/` folder and name it `KhazanahAnnualReview2026.pdf`

### Step 4 — Run ingestion (one-time, ~5 minutes)
```bash
python scripts/ingest.py
```

### Step 5 — Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Step 6 — Start the frontend (new terminal)
```bash
venv\Scripts\activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**API documentation:** `http://localhost:8000/docs`

---

## Section 6 — Design Questions

### Q1: If this tool were deployed internally at Khazanah for daily use by analysts, what would you change?

**Authentication and access control first.** The current API has no auth layer —
anyone on the network can query it. Production deployment needs SSO integration
(Azure AD or Okta), role-based access, and audit logging of every query and
answer for compliance.

**Infrastructure changes:**
- Replace ChromaDB local with Qdrant Cloud or Weaviate for concurrent access.
  ChromaDB's local persistent client is single-process — multiple analysts
  hitting it simultaneously will cause locking errors
- Move ingestion to a scheduled pipeline (Airflow or a simple cron job) that
  re-processes the Annual Review when a new version is published
- Add Redis caching for repeated queries — analysts often ask the same financial
  figure questions repeatedly; these should be served from cache, not LLM calls
- Replace Streamlit with a Next.js frontend deployed on the internal network
  with proper session management and conversation history per user

**Observability:**
- Structured logging of every query, retrieved chunks, confidence score, and
  answer — this data is essential for identifying systematic retrieval failures
- Latency tracking — the current implementation has no SLA. Production needs
  p95 response time monitoring
- A "thumbs down" feedback button that logs negative responses for manual review

**Model tier:**
- Upgrade from GPT-4o-mini to GPT-4o for production answer quality. The cost
  difference at internal usage volumes (~500 queries/day) is approximately
  $15/month — negligible for an institution of Khazanah's scale

### Q2: A user reports that the tool confidently returned an incorrect answer and shared it in a presentation. How would you prevent this?

This scenario most likely occurred because the confidence scoring was high
(slides retrieved were semantically similar) but the actual answer was in a
chart that the vision model extracted imprecisely.

**Three layers of prevention:**

*Layer 1 — Retrieval gate (already implemented):*
Refuse to answer when no chunk scores above the similarity threshold. This
catches out-of-scope queries cleanly.

*Layer 2 — UI design:*
The confidence badge and warning messages are the most important safety feature.
A "LOW CONFIDENCE" warning in amber with explicit text "verify before sharing"
creates a paper trail — if a user ignores the warning and shares incorrect data,
the system documented its uncertainty. The current implementation shows this
warning, but it should be made more prominent for analyst-facing deployment.

*Layer 3 — Source transparency:*
Every answer must show which slides it used and what those slides actually say.
If an analyst can see "this answer came from the bar chart on slide 8," they
can open slide 8 and verify in 10 seconds. The current source expander
implements this — but in production, it should link directly to the PDF page.

**What I would add:**
- A mandatory disclaimer on every answer: "AI-generated summary. Verify
  against source document before use in presentations or reports."
- Answer versioning — if the PDF is updated and re-ingested, answers generated
  from the old version should be marked as potentially stale
- A formal escalation path: "Report incorrect answer" button that logs the
  query, answer, and user report to a review queue

### Q3: You need to push an update to the RAG pipeline, but the tool is actively being used by 20 analysts. How do you roll out safely?

**Never update the live collection in place.** The core mistake would be
dropping and recreating the ChromaDB collection while analysts are mid-query.

**The safe process:**

*Step 1 — Build in parallel:*
Run the updated ingestion pipeline against a new ChromaDB collection named
`khazanah_kar_2026_v2`. The live collection `khazanah_kar_2026` continues
serving all queries uninterrupted.

*Step 2 — Validate before switching:*
Run the regression test suite (20–30 known Q&A pairs) against the new
collection. Compare answer quality, confidence scores, and retrieval patterns
against the baseline. Only proceed if quality is equal or better.

*Step 3 — Shadow mode:*
Route 10% of queries to the new collection while still serving the response
from the old one. Log both responses. Review differences manually over 24 hours.
If no regressions, increase to 50%, then 100%.

*Step 4 — Atomic cutover:*
Update the collection name in config and do a zero-downtime restart of the
FastAPI server. Since the new collection is already warm and indexed, there
is no gap in service.

*Step 5 — Keep the old collection for 48 hours:*
Don't delete `khazanah_kar_2026` immediately. If analysts report regressions
after the cutover, roll back by changing one config value and restarting.

**For a pipeline change that affects embeddings specifically** (e.g. switching
embedding models), the old and new collections are incompatible — query
embeddings must match ingestion embeddings. This requires a full parallel
rebuild and cannot be partially rolled out. Schedule this during off-hours with
a maintenance notification.
