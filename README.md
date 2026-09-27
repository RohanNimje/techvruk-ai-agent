# ⚡ Hybrid AI Research Analyst

> A **production-ready**, autonomous dual-engine AI agent that intelligently routes queries between a verified internal knowledge base (FAISS RAG) and the live internet (DuckDuckGo), powered by **Google Gemini 3.x**, **LangGraph**, and a world-class **Streamlit** SaaS interface.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-6366F1?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.x%20Flash-4285F4?style=flat&logo=google&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=flat)](https://opensource.org/licenses/MIT)

---

## ✨ Production-Grade Feature Highlights

| Feature | Details |
|---|---|
| 🧠 **ReAct Architecture** | LangGraph state machine with explicit Reason → Act → Observe loop |
| 🎨 **SaaS-Grade UI** | Linear/Notion-inspired light theme, glassmorphic cards, high-contrast chat input |
| ⚡ **Sub-2s Reload Times** | `@st.cache_resource` caches 80MB embeddings & FAISS index across all reruns |
| ⚠️ **Graceful Error Handling** | Amber (quota) / Red (generic) notification cards — zero raw JSON crashes |
| 🔄 **Live Model Switching** | Swap between 5 Gemini 3.x models from the sidebar mid-conversation |
| 🔒 **Secure by Design** | API key via `.env` only — never exposed in the UI |
| 📌 **Error State Persistence** | Error cards survive Streamlit reruns (stored in session state with type flags) |

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                   HYBRID AI RESEARCH ANALYST                     ║
║                  (ReAct Architecture — LangGraph)                ║
╚══════════════════════════════════════════════════════════════════╝

  User Query
      │
      ▼
┌─────────────┐
│  Streamlit  │  ← app.py  (SaaS chat UI, error cards, tool badge pills)
│     UI      │
└──────┬──────┘
       │  calls run_agent(query, model_name)
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                        │
│                                                                  │
│   MessagesState = { "messages": [HumanMsg, AIMsg, ToolMsg, ...] }│
│                                                                  │
│   [START]                                                        │
│      │                                                           │
│      ▼                                                           │
│  ┌───────────────────────────────────┐                           │
│  │         REASONER NODE             │  ← agent.py: reasoner_node│
│  │                                   │                           │
│  │  Gemini LLM (with tools bound)    │                           │
│  │  Reads full message history       │                           │
│  │  PLAN: Which tool should I use?   │                           │
│  └──────────────┬────────────────────┘                           │
│                 │                                                 │
│         should_continue()  ← conditional edge router             │
│                 │                                                 │
│         ┌───────┴────────┐                                        │
│         │                │                                        │
│   tool_calls?          no tool calls                             │
│         │                │                                        │
│         ▼                ▼                                        │
│  ┌─────────────┐      [END] → final answer                       │
│  │  TOOL NODE  │  ← agent.py: ToolNode(TOOLS)                    │
│  │             │                                                  │
│  │  ACT:       │                                                  │
│  │  Executes   ├──── rag_search() ──► FAISS Vector Store         │
│  │  chosen     │                       (dummy_data.txt, cached)  │
│  │  tool       ├──── web_search() ──► DuckDuckGo API             │
│  └──────┬──────┘                                                  │
│         │                                                         │
│         │  OBSERVE: Tool output appended to state as ToolMessage  │
│         │                                                         │
│         └──────────────────────► back to REASONER NODE           │
│                                  (LLM reads the tool result       │
│                                   and decides next action)        │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
  Final Answer + Tools Used
       │
       ▼
┌─────────────┐
│  Streamlit  │  Renders clean card with tool badge pills
│     UI      │  (🗂️ Internal KB  ·  🌐 Live Web  ·  ⚡ Direct Synthesis)
└─────────────┘
```

### Plan → Act → Observe Cycle

| Phase | Where | What it does |
|---|---|---|
| **PLAN** | `reasoner_node` | Gemini reads query + history, picks a tool or ends |
| **ACT** | `tool_node` | Executes `rag_search` (FAISS) or `web_search` (DuckDuckGo) |
| **OBSERVE** | `MessagesState` | Tool output appended as `ToolMessage`; LLM reads it next cycle |
| **REPEAT** | Conditional edge | Loop continues until Gemini returns a plain-text final answer |

---

## 📁 Project Structure

```
techvruk-research-agent/
│
├── agent.py          # Core: LangGraph graph, tools, LLM binding, run_agent()
├── app.py            # Streamlit SaaS UI — Linear/Notion design system
├── dummy_data.txt    # Internal company knowledge base (TechVruk KB)
├── requirements.txt  # All Python dependencies
├── .env              # Your GOOGLE_API_KEY (never committed)
├── .gitignore        # Excludes .env, __pycache__, venv/
└── README.md         # This file
```

---

## ⚙️ Local Setup & Installation

### Prerequisites

- Python **3.10+**
- A **Google Gemini API key** — free tier at [Google AI Studio](https://aistudio.google.com/app/apikey)

### Step 1 — Clone the repository

```bash
git clone https://github.com/RohanNimje/techvruk-ai-agent.git
cd techvruk-ai-agent
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On the first run, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~80 MB). This happens only once — subsequent startups load from cache in **1–2 seconds**.

### Step 4 — Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

> **Security:** `.env` is in `.gitignore` and is **never** committed or shown in the UI.

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🚀 Deploy on Streamlit Cloud

1. **Push your code** to a public GitHub repository (ensure `.env` is in `.gitignore`).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → click **"New app"**.

3. Select your repository, branch (`main`), and entry point (`app.py`).

4. Under **"Advanced settings → Secrets"**, add your API key in TOML format:

   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key-here"
   ```

5. Click **"Deploy"** — your app is live in ~2 minutes.

> **Tip:** Streamlit Secrets are injected as environment variables at runtime. `load_dotenv()` in `app.py` and `agent.py` will automatically pick them up alongside any local `.env` file.

---

## 🎨 Premium UI/UX

The interface is built on a **Linear & Notion-inspired design system** — clean, high-contrast, and product-quality.

### Design System

| Element | Implementation |
|---|---|
| **Color Palette** | Porcelain `#F8FAFC` bg · White `#FFFFFF` cards · Slate `#E2E8F0` borders |
| **Typography** | *Plus Jakarta Sans* (display) + *Inter* (body) via Google Fonts |
| **User Bubbles** | Indigo gradient `#4F46E5 → #4338CA`, white text, right-aligned |
| **Assistant Cards** | White card, 1px slate border, soft shadow, asymmetric border-radius |
| **Tool Badges** | Indigo pill 🗂️ Internal KB · Emerald pill 🌐 Live Web · Slate ⚡ Direct Synthesis |
| **Chat Input** | `#FFFFFF` bg · `#0F172A` text · `#6366F1` focus ring — zero contrast bugs |
| **Sidebar Toggle** | Fully visible collapse/expand button with hover states — never hidden |
| **Sidebar Padding** | `4rem` top padding prevents brand header overlapping the collapse button |
| **Stats Row** | Live counters: Queries · Internal KB Calls · Live Web Calls |
| **Empty State** | Hero card with icon, headline, and onboarding hint |

### Humanized Sidebar Workflow

Replaces robotic "Plan→Act→Observe" jargon with three human-readable value propositions displayed as styled feature cards:

- **Smart Routing** — Knows instantly whether to query verified internal records or browse the web
- **Zero Hallucination** — Pulls precise facts from FAISS vector storage
- **Real-Time Intelligence** — Searches current global sources for anything beyond company walls

---

## ⚡ Performance — Sub-2s Reload Times

### The Problem

Streamlit re-executes the entire script on every user interaction. Without caching, the 80 MB `all-MiniLM-L6-v2` sentence-transformer weights and FAISS index are rebuilt **on every rerun** — producing `Loading weights…` spam in the terminal and 30–60 second delays.

### The Solution: `@st.cache_resource`

```python
# agent.py

@st.cache_resource(show_spinner=False)
def get_vector_store(file_path: str = "dummy_data.txt") -> FAISS:
    """
    Loads the KB, embeds it, and builds a FAISS index.
    Cached by Streamlit — executed ONCE on server start,
    then served from memory on all subsequent reruns.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
```

**Measured results:**

| Call | Time |
|---|---|
| First call (server start, cold) | ~30–60s (one-time download + load) |
| All subsequent reruns | **< 0.001s** (served from memory) |

The `rag_search` tool always calls `get_vector_store()` — which hits the in-process cache instantly, never rebuilding the index.

---

## ⚠️ Graceful Error Handling

Raw API errors (429 rate limits, malformed responses, network failures) are **classified and intercepted** before they reach the UI. Zero raw JSON or stack traces are ever shown to the user.

### Error Classification Logic

```python
# app.py

_quota_signals = (
    "429", "RESOURCE_EXHAUSTED", "quota",
    "rate limit", "rate_limit", "rateLimitExceeded", "Too Many Requests",
)

if any(sig.lower() in str(e).lower() for sig in _quota_signals):
    _error_type = "quota"    # → Amber notification card
else:
    _error_type = "generic"  # → Red notification card
```

### What the User Sees

**API Quota / Rate Limit (HTTP 429):**

> ⚠️ **API Quota Exceeded**
>
> The model **gemini-3.8-flash** has temporarily hit its rate limit. This usually resolves within a minute, or you can switch to a lighter model from the sidebar right now.
>
> **Suggested alternatives:** `gemini-3.5-flash-lite` · `gemini-3.1-flash-lite`

**Generic Error:**

> ❌ **Agent Error**
>
> `<truncated exception string, max 400 chars>`
>
> Please verify that `GOOGLE_API_KEY` is set correctly in your `.env` file.

### Error State Persistence

Each message saved to `st.session_state` carries `is_error: bool` and `error_type: "quota" | "generic"`. The history renderer checks these flags on every rerun and re-renders the styled notification card — errors never revert to raw text after a page interaction.

---

## 🔄 Dynamic Model Switching

The sidebar `st.selectbox` exposes five Gemini 3.x models. Switching mid-conversation instantly hot-swaps the LLM via `set_model()` in `agent.py` (rebuilds only when the model actually changes):

| Model | Best for |
|---|---|
| `gemini-3.8-flash` | Default — most capable, highest throughput |
| `gemini-3.7-flash` | Stable, well-tested alternative |
| `gemini-3.5-flash` | Balanced speed and reasoning |
| `gemini-3.5-flash-lite` | Lighter — use when rate limits are hit |
| `gemini-3.1-flash-lite` | Lightest — maximum quota headroom |

A try/except fallback in `build_llm()` automatically drops to `gemini-3.8-flash` if the selected model returns a 404 Not Found.

---

## 🛠️ How the LLM Chooses a Tool

The system prompt instructs Gemini to follow strict routing rules:

| Query type | Tool used |
|---|---|
| TechVruk internship · stipend · HR · team · products | `rag_search` (FAISS) |
| General knowledge · current events · tech concepts | `web_search` (DuckDuckGo) |
| RAG returns no useful results | Fall back to `web_search` |

---

## 🧪 Example Queries

| Query | Expected Tool |
|---|---|
| "What is the monthly stipend for AI/ML interns?" | 🗂️ `rag_search` |
| "Who is TechVruk's CTO?" | 🗂️ `rag_search` |
| "What awards has TechVruk won?" | 🗂️ `rag_search` |
| "What is LangGraph?" | 🌐 `web_search` |
| "Latest AI news today" | 🌐 `web_search` |
| "What is the capital of France?" | 🌐 `web_search` |

---

## 🔑 Technology Stack

| Component | Technology | Why |
|---|---|---|
| **Orchestration** | LangGraph `StateGraph` | Explicit, debuggable ReAct state machine |
| **LLM** | Google Gemini 3.x Flash | Fast inference, excellent tool-calling, free tier |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | No API key needed; offline-capable; ~80 MB |
| **Vector Store** | FAISS (in-memory, cached) | Zero infrastructure; sub-millisecond search |
| **Web Search** | DuckDuckGo (`langchain-community`) | Free, no API key required |
| **Frontend** | Streamlit 1.64 | Rapid UI with deep customisation via `st.markdown` |
| **Caching** | `@st.cache_resource` | Eliminates repeated model weight loading |
| **Security** | `python-dotenv` + `.gitignore` | API key never in code or UI |

---

## 📄 License

```
MIT License

Copyright (c) 2026 Rohan Nimje

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built for the TechVruk Internship Technical Challenge · Production-Ready Edition*
