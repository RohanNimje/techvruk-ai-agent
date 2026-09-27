# =============================================================================
# agent.py — Hybrid AI Research Analyst (Core Agent Logic)
# =============================================================================
# ARCHITECTURE: ReAct (Reasoning + Acting) using LangGraph
#
# The flow is a simple loop:
#
#   [START]
#      │
#      ▼
#   ┌──────────────┐      tool calls present?       ┌──────────────┐
#   │   REASONER   │ ─────────── YES ──────────────► │  TOOL NODE   │
#   │  (LLM node)  │                                 │ (executes    │
#   └──────────────┘ ◄──── loops back ───────────── │  RAG or Web) │
#          │                                         └──────────────┘
#      no tool calls
#          │
#          ▼
#       [END] → final answer returned
#
# =============================================================================

import os

# Load .env file into os.environ as early as possible.
# This means `python agent.py` works standalone without manually exporting env vars.
from dotenv import load_dotenv
load_dotenv()  # reads GOOGLE_API_KEY (and others) from .env

from typing import Annotated, Sequence

# --- LangChain / LangGraph imports ---
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph's prebuilt ToolNode handles calling whichever tool the LLM chose
from langgraph.prebuilt import ToolNode

# MessagesState is a built-in state schema that stores a list of messages.
# The "messages" key uses `add_messages` as its reducer — this means every
# new message is APPENDED to the list rather than overwriting it.
# This is how conversational memory is maintained across graph nodes.
from langgraph.graph import END, START, MessagesState, StateGraph

# --- Streamlit for resource caching ---
import streamlit as st

# --- RAG-specific imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Web Search import ---
from langchain_community.tools import DuckDuckGoSearchRun

# =============================================================================
# STEP 1: BUILD & CACHE THE FAISS VECTOR STORE (Resource Caching)
# =============================================================================
# We load dummy_data.txt, split it into chunks, embed each chunk using a
# lightweight sentence-transformer model, and store in an in-memory FAISS index.
#
# Decorated with @st.cache_resource so HuggingFace embeddings and FAISS index
# are loaded and cached ONCE upon server start across all Streamlit reruns.

@st.cache_resource(show_spinner=False)
def get_vector_store(file_path: str = "dummy_data.txt") -> FAISS:
    """
    Reads the local text file, chunks it, and builds a FAISS vector store.
    Cached with @st.cache_resource so HuggingFace embeddings and the FAISS index
    are loaded only ONCE upon server start, eliminating repeated weight loading.
    """
    print(f"Building FAISS vector store from {file_path} (one-time initialization)...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store ready and cached.")
    return vector_store

# Alias build_vector_store for backwards compatibility
build_vector_store = get_vector_store

# Warm-up / initialize the cached vector store
_vector_store = get_vector_store()


# =============================================================================
# STEP 2: DEFINE THE TOOLS
# =============================================================================
# Tools are plain Python functions decorated with @tool.
# LangChain reads the function's docstring to tell the LLM what each tool does.
# The LLM uses this description to autonomously decide which tool to call.

@tool
def rag_search(query: str) -> str:
    """
    Search the internal TechVruk company knowledge base for information about
    the company's products, internship program, team, HR policies, culture,
    tech stack, and other internal topics. Use this tool FIRST for any question
    that sounds like it could be answered from internal company documents.
    """
    try:
        # similarity_search returns the top-k most relevant document chunks
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=3)

        if not results:
            return "No relevant information found in the internal knowledge base."

        # Combine the content of the top matching chunks into a single string
        combined = "\n\n---\n\n".join([doc.page_content for doc in results])
        return f"[Internal KB Results]\n\n{combined}"

    except Exception as e:
        # Return a graceful fallback so a FAISS / embedding failure doesn't
        # crash the entire LangGraph run — the LLM can still try web_search.
        print(f"rag_search error: {e}")
        return (
            "[Internal KB Unavailable] The knowledge base search encountered an error. "
            "Please try rephrasing your question or use the web search tool instead."
        )


@tool
def web_search(query: str) -> str:
    """
    Search the live internet using DuckDuckGo for real-time, up-to-date
    information such as current news, recent events, general world knowledge,
    public company data, or anything not found in internal company documents.
    """
    try:
        # DuckDuckGoSearchRun is a LangChain wrapper around DuckDuckGo's search API.
        # It returns a summary string of the top search results.
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        if not result or not result.strip():
            return "[Web Search] No results returned for this query. Please try a different search term."
        return f"[Web Search Results]\n\n{result}"

    except Exception as e:
        err_str = str(e)
        print(f"web_search error: {e}")
        # Distinguish rate-limit errors so the LLM can give a helpful reply.
        _rate_signals = ("202", "429", "rate", "ratelimit", "too many", "blocked")
        if any(sig.lower() in err_str.lower() for sig in _rate_signals):
            return (
                "[Web Search Unavailable] DuckDuckGo has temporarily rate-limited this request. "
                "Please wait a moment and try again."
            )
        return (
            f"[Web Search Unavailable] The web search encountered an error: {err_str[:200]}. "
            "Please try again shortly."
        )


# =============================================================================
# STEP 3: CONFIGURE THE LLM WITH TOOL BINDING
# =============================================================================
# We initialize the Gemini LLM and then "bind" our tools to it.
# Binding tells the LLM about the tools' names, descriptions, and input schemas.
# When the LLM decides to use a tool, it returns a structured ToolCall object
# (not a plain text response). LangGraph's ToolNode then executes that call.

# Collect tools in a list so we can bind them and also pass them to ToolNode
TOOLS = [rag_search, web_search]


def build_llm(model_name: str = "gemini-3.8-flash"):
    """
    Initialize and return a Gemini LLM with tools bound to it.

    Args:
        model_name: The Gemini model to use (e.g. 'gemini-3.8-flash', 'gemini-3.5-flash').

    Raises immediately on any failure — no automatic fallback. This lets
    app.py's top-level exception handler classify the error (quota vs. generic)
    and render the styled error card so the user can manually switch models
    from the sidebar.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it before running the agent."
        )

    # No try/except here — let all failures propagate directly to app.py's
    # handler, which will show the appropriate styled error card.
    print(f"Initializing LLM with model: {model_name}")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,    # Lower temp → more factual, consistent answers
    )
    # .bind_tools() attaches the tool schemas to the LLM so it knows
    # which tools are available and what arguments each tool expects.
    llm_with_tools = llm.bind_tools(TOOLS)
    return llm_with_tools, model_name


# =============================================================================
# STEP 4: DEFINE THE GRAPH NODES
# =============================================================================
# A LangGraph "node" is simply a function that:
#   - Takes the current State as input
#   - Returns a PARTIAL update to the State (only the keys it wants to change)
#
# The State here is MessagesState, which has one key: "messages" (a list).

# Build the LLM once at module load time using the default model.
# app.py can override this by calling set_model() before invoking run_agent().
_llm_with_tools, _active_model = build_llm("gemini-3.8-flash")


def set_model(model_name: str):
    """
    Hot-swaps the LLM to a different Gemini model at runtime.
    Called by app.py whenever the user changes the model selector in the sidebar.
    Updates the module-level _llm_with_tools used inside reasoner_node.
    """
    global _llm_with_tools, _active_model
    # Only rebuild if the model has actually changed — avoids unnecessary API calls
    if model_name != _active_model:
        _llm_with_tools, _active_model = build_llm(model_name)
        print(f"Model switched to: {_active_model}")

# System prompt that guides the LLM's decision-making logic
SYSTEM_PROMPT = SystemMessage(content="""You are a Hybrid AI Research Analyst for TechVruk.

You have access to two tools:
1. `rag_search` — searches TechVruk's internal company knowledge base
2. `web_search` — searches the live internet via DuckDuckGo

Decision Rules (follow strictly):
- If the question is about TechVruk's internship, team, products, HR, culture, or tech stack → use `rag_search` FIRST.
- If the question is about current events, general world knowledge, or topics unlikely to be in company docs → use `web_search`.
- If `rag_search` returns no useful results, fall back to `web_search`.
- Always cite which source your answer came from (Internal KB or Web).
- Be concise, professional, and helpful.
""")


def reasoner_node(state: MessagesState) -> dict:
    """
    THE BRAIN OF THE AGENT.

    This node invokes the LLM (with tools bound) on the current message history.
    The LLM reads all previous messages and decides:
      A) Return a final answer (plain text) → graph routes to END
      B) Call a tool (structured ToolCall) → graph routes to tool_node

    The returned dict {"messages": [response]} is MERGED into the state's
    message list by LangGraph's `add_messages` reducer (not overwritten).
    """
    from langchain_core.messages import AIMessage

    # Prepend the system prompt to provide context on every LLM call
    messages = [SYSTEM_PROMPT] + state["messages"]

    try:
        # Invoke the LLM — it either returns text or a tool call
        response = _llm_with_tools.invoke(messages)
    except Exception as e:
        # Surface LLM errors as a plain AIMessage so the graph reaches END
        # cleanly instead of crashing — app.py's top-level handler will then
        # re-raise this to display the styled error card in the UI.
        print(f"reasoner_node LLM error: {e}")
        raise  # re-raise so app.py's except block classifies & renders it

    # Return only the new message; LangGraph appends it to state["messages"]
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """
    THE ROUTER — this is a LangGraph "conditional edge" function.

    After the reasoner_node runs, LangGraph calls this function to decide
    the next step. It inspects the LAST message in state:

    - If the last message contains tool_calls → route to "tool_node"
    - Otherwise (plain text response) → route to END (graph finishes)

    This creates the ReAct loop: Reason → Act → Observe → Reason → ...
    """
    last_message = state["messages"][-1]

    # AIMessage populates .tool_calls if the LLM chose to call a tool
    if last_message.tool_calls:
        return "tool_node"   # Continue the loop → execute the tool

    return END               # No tool call → we have the final answer


# =============================================================================
# STEP 5: BUILD THE LANGGRAPH STATE MACHINE
# =============================================================================

def build_graph() -> StateGraph:
    """
    Assembles and compiles the LangGraph state machine.

    Node summary:
      - "reasoner"  → LLM with tools (decides what to do)
      - "tool_node" → Executes whichever tool the LLM chose

    Edge summary:
      - START → reasoner         (always start here)
      - reasoner → tool_node     (if LLM made a tool call)
      - reasoner → END           (if LLM gave a final answer)
      - tool_node → reasoner     (always loop back after tool execution)
    """

    # StateGraph(MessagesState) creates a graph that manages a state dict
    # with a single "messages" key. The add_messages reducer on that key
    # ensures messages accumulate across node calls (conversation memory).
    graph = StateGraph(MessagesState)

    # --- Register Nodes ---
    # "reasoner" node: our custom function that calls the LLM
    graph.add_node("reasoner", reasoner_node)

    # "tool_node" node: LangGraph's prebuilt ToolNode.
    # It reads the tool_calls from the last AIMessage, executes the matching
    # tool function, and appends the tool's output as a ToolMessage to state.
    graph.add_node("tool_node", ToolNode(TOOLS))

    # --- Register Edges ---
    # The graph always begins at the "reasoner" node
    graph.add_edge(START, "reasoner")

    # After "reasoner" runs, call `should_continue` to pick the next node.
    # The dict maps return values of `should_continue` to node names.
    graph.add_conditional_edges(
        "reasoner",          # source node
        should_continue,     # routing function
        {
            "tool_node": "tool_node",   # route to tool execution
            END: END,                   # route to graph completion
        },
    )

    # After the tool executes, ALWAYS go back to the reasoner so the LLM
    # can observe the tool output and decide its next action (ReAct loop)
    graph.add_edge("tool_node", "reasoner")

    # compile() validates the graph structure and returns a runnable object
    compiled = graph.compile()
    return compiled


# Build and export the compiled graph (imported by app.py)
research_agent = build_graph()
print("LangGraph ReAct agent compiled and ready.")


# =============================================================================
# STEP 6: HELPER — Normalise LLM message content to a plain string
# =============================================================================
# Newer Gemini models (2.x / 3.x series) can return .content as a LIST of
# content-block dicts rather than a plain string. Each block looks like:
#   {"type": "text", "text": "...", "extras": {"signature": "..."}}
# or as LangChain content objects with a .text attribute.
# _extract_text() normalises ALL of these shapes into one clean str.

def _extract_text(content) -> str:
    """
    Convert an AIMessage .content value into a clean plain-text string,
    regardless of whether Gemini returned a str, a list of dicts, or a
    list of LangChain content-block objects.

    Strips 'extras', 'signature', and any other metadata fields.
    """
    # Case 1: already a plain string — just strip whitespace and return
    if isinstance(content, str):
        return content.strip()

    # Case 2: list of content blocks (Gemini 2.x / 3.x multipart format)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Gemini dict block: {"type": "text", "text": "...", "extras": {...}}
                # We only want the "text" key — ignore type, extras, signature, etc.
                text = block.get("text", "")
                if text:
                    parts.append(str(text).strip())
            elif hasattr(block, "text"):
                # LangChain content object with a .text attribute
                text = getattr(block, "text", "")
                if text:
                    parts.append(str(text).strip())
            else:
                # Unknown block shape — cast to string as a last resort
                parts.append(str(block).strip())
        return "\n\n".join(filter(None, parts))

    # Case 3: some other type — convert to string defensively
    return str(content).strip()


# =============================================================================
# STEP 7: Run the agent and return structured results
# =============================================================================

def run_agent(user_query: str, model_name: str = "gemini-3.8-flash") -> dict:
    """
    Invokes the compiled LangGraph agent with a user query.

    Args:
        user_query: The plain-text question from the user.
        model_name: Gemini model to use. If it differs from the currently active
                    model, set_model() is called to hot-swap the LLM before running.

    Returns a dict with:
      - "answer"      : the final text response from the LLM
      - "tools_used"  : list of tool names that were called during the run
      - "active_model": the model that actually ran (may differ if fallback triggered)
      - "all_messages": the full message history for debugging
    """
    # Hot-swap the LLM if the user selected a different model in the UI
    set_model(model_name)

    # Wrap the user's plain text into a HumanMessage (LangChain format)
    initial_state = {"messages": [HumanMessage(content=user_query)]}

    # .invoke() runs the graph synchronously until it reaches END.
    # The final state contains ALL messages accumulated across every node call.
    final_state = research_agent.invoke(initial_state)

    all_messages = final_state["messages"]

    # --- Extract which tools were used ---
    tools_used = []
    for msg in all_messages:
        # AIMessages that contain tool_calls have a .tool_calls attribute
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc["name"]
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

    # The last message is always the final AIMessage with the text answer.
    # _extract_text() normalises the content to a clean string regardless of
    # whether Gemini returned a plain str or a list of content-block dicts.
    final_answer = _extract_text(all_messages[-1].content)

    return {
        "answer": final_answer,
        "tools_used": tools_used,
        "active_model": _active_model,
        "all_messages": all_messages,
    }


# Quick local test (runs only when this file is executed directly)
if __name__ == "__main__":
    query = "What is the stipend for the TechVruk AI/ML internship?"
    print(f"\nQuery: {query}\n")
    result = run_agent(query)
    print(f"Answer:\n{result['answer']}")
    print(f"\nTools Used: {result['tools_used']}")
