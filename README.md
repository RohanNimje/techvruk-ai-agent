# 🔬 Hybrid AI Research Analyst

> An intelligent ReAct-based AI agent that autonomously decides whether to search internal company knowledge (via FAISS RAG) or the live internet (via DuckDuckGo), powered by **Google Gemini**, **LangGraph**, and **Streamlit**.

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
│  Streamlit  │  ← app.py  (Chat UI, renders messages, shows tool badges)
│     UI      │
└──────┬──────┘
       │  calls run_agent(query)
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
│  │  chosen     │                       (dummy_data.txt)           │
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
│  Streamlit  │  Renders answer bubble with tool badge pills
│     UI      │  (🗂️ Internal KB  or  🌐 Web Search)
└─────────────┘
```

### Plan → Act → Observe Explained

| Phase       | Where it happens      | What it does                                                        |
|-------------|-----------------------|---------------------------------------------------------------------|
| **PLAN**    | `reasoner_node`       | Gemini reads the query + message history, decides which tool to call|
| **ACT**     | `tool_node`           | Executes `rag_search` (FAISS) or `web_search` (DuckDuckGo)         |
| **OBSERVE** | State (`messages`)    | Tool output is appended as a `ToolMessage`; LLM reads it next cycle |
| **REPEAT**  | Conditional edge      | If LLM needs more info, loop continues; else graph ends             |

---

## 📁 Project Structure

```
techvruk-research-agent/
│
├── agent.py          # Core: LangGraph graph, tools, LLM binding, run_agent()
├── app.py            # Streamlit chat UI
├── requirements.txt  # All Python dependencies
├── dummy_data.txt    # Internal company knowledge base (used by FAISS RAG)
└── README.md         # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A **Google Gemini API key** (free tier available at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Step 1 — Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd techvruk-research-agent

# Or just navigate to the folder:
cd path/to/techvruk-research-agent
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

> **Note:** On first run, the `sentence-transformers` library will download the  
> `all-MiniLM-L6-v2` model (~80MB). This happens only once.

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

### Step 5 — Enter your API key

Paste your **Google Gemini API key** in the left sidebar field.  
Alternatively, set it as an environment variable before running:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-api-key-here"

# macOS / Linux
export GOOGLE_API_KEY="your-api-key-here"
```

---

## 🛠️ How the LLM Chooses a Tool

The system prompt instructs Gemini to follow these routing rules:

| Query type                                        | Tool used        |
|---------------------------------------------------|------------------|
| TechVruk internship / stipend / HR / team / products | `rag_search`  |
| General knowledge / current events / tech concepts  | `web_search`  |
| RAG returns no results → fall back                | `web_search`  |

---

## 🧪 Example Queries to Test

| Query                                       | Expected Tool    |
|---------------------------------------------|------------------|
| "What is the monthly stipend for interns?"  | 🗂️ `rag_search`  |
| "Who is TechVruk's CTO?"                    | 🗂️ `rag_search`  |
| "What awards has TechVruk won?"             | 🗂️ `rag_search`  |
| "What is LangGraph used for?"               | 🌐 `web_search`  |
| "Latest AI news this week"                  | 🌐 `web_search`  |
| "What is the capital of France?"            | 🌐 `web_search`  |

---

## 🔑 Key Technology Choices

| Component           | Technology                    | Why                                                   |
|---------------------|-------------------------------|-------------------------------------------------------|
| Orchestration       | LangGraph `StateGraph`        | Simple, explicit state machine; easy to trace + debug |
| LLM                 | Google Gemini 1.5 Flash       | Fast, free-tier, excellent tool-calling support       |
| Embeddings          | `all-MiniLM-L6-v2` (local)   | No API needed; works offline; fast inference          |
| Vector Store        | FAISS (in-memory)             | Zero infrastructure; perfect for demo scale           |
| Web Search          | DuckDuckGo (no API key)       | Free, no rate limits for demo purposes                |
| UI                  | Streamlit                     | Fast to build; renders beautifully for demos          |

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

*Built for the TechVruk Internship Technical Challenge.*
