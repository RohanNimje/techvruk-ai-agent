# =============================================================================
# app.py — Hybrid AI Research Analyst (SaaS Product Interface)
# =============================================================================
# Built with: Streamlit · LangGraph · Google Gemini · FAISS · DuckDuckGo
# Design System: Clean, high-contrast porcelain & slate palette inspired by Linear & Notion
# =============================================================================

import os
import time
from dotenv import load_dotenv

# Load environment variables from .env before importing agent
load_dotenv()

import streamlit as st
from agent import run_agent

# Use markdown-it for pixel-perfect markdown rendering in assistant cards
try:
    from markdown_it import MarkdownIt
    _md_parser = MarkdownIt()
    def render_markdown(text: str) -> str:
        return _md_parser.render(text)
except Exception:
    def render_markdown(text: str) -> str:
        return f"<p>{text}</p>"

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TechVruk Research Analyst",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN SYSTEM & CUSTOM CSS (Linear & Notion Inspired)
# =============================================================================

st.markdown("""
<style>
    /* ── Typography ────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-display: 'Plus Jakarta Sans', var(--font-main);
        --bg-app: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-focus: #6366F1;
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --indigo-primary: #6366F1;
        --indigo-hover: #4F46E5;
        --indigo-tint: #EEF2FF;
        --indigo-border: #C7D2FE;
        --emerald-tint: #ECFDF5;
        --emerald-border: #A7F3D0;
        --emerald-text: #047857;
    }

    /* ── Global Styles ─────────────────────────────────────────────────── */
    html, body, .stApp {
        font-family: var(--font-main) !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* ── Hide Streamlit Chrome & Cloud Toolbar ─────────────────────────── */
    #MainMenu { visibility: hidden !important; display: none !important; }
    footer    { visibility: hidden !important; display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stHeader"]  { display: none !important; }
    header[data-testid="stHeader"] .stAppHeader { display: none !important; }

    /* ── Sidebar Collapse Button (inside expanded sidebar) ─────────────── */
    /* Streamlit renders this as a button at the top of the sidebar panel.  */
    /* We style it cleanly but NEVER hide it.                               */
    [data-testid="stSidebarCollapseButton"] > button {
        color: #64748B !important;
        background: transparent !important;
        border: none !important;
        padding: 4px !important;
        border-radius: 6px !important;
        transition: color 0.15s ease, background 0.15s ease !important;
    }
    [data-testid="stSidebarCollapseButton"] > button:hover {
        color: #0F172A !important;
        background: #F1F5F9 !important;
    }
    [data-testid="stSidebarCollapseButton"] > button svg {
        width: 18px !important;
        height: 18px !important;
    }

    /* ── Sidebar Expand Button (when sidebar is collapsed) ─────────────── */
    /* Streamlit renders this as a floating chevron on the left edge.       */
    /* Must ALWAYS remain visible and functional.                           */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0 8px 8px 0 !important;
        box-shadow: 2px 0 6px rgba(15, 23, 42, 0.06) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        z-index: 999999 !important;
    }
    [data-testid="collapsedControl"]:hover {
        border-color: #6366F1 !important;
        box-shadow: 2px 0 8px rgba(99, 102, 241, 0.12) !important;
    }
    [data-testid="collapsedControl"] button {
        display: flex !important;
        visibility: visible !important;
        color: #334155 !important;
    }
    [data-testid="collapsedControl"] button:hover {
        color: #6366F1 !important;
    }

    /* ── Layout Padding ────────────────────────────────────────────────── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6.5rem !important;
        max-width: 820px !important;
    }

    /* ── Sidebar: Crisp White & High Contrast ──────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 2px 0 8px rgba(15, 23, 42, 0.02) !important;
    }

    /* 4rem top padding so the brand header never overlaps the collapse button */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 4rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-family: var(--font-display) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #334155 !important;
    }

    [data-testid="stSidebar"] hr, hr {
        border: none !important;
        border-top: 1px solid #E2E8F0 !important;
        margin: 1.1rem 0 !important;
    }

    /* ── Sidebar Selectbox (Model Selector) ────────────────────────────── */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #475569 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 2px 6px !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #CBD5E1 !important;
        background-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:focus-within > div {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        background-color: #FFFFFF !important;
    }

    /* ── Sidebar Buttons (Interactive Rounded Pills) ──────────────────── */
    [data-testid="stSidebar"] .stButton > button {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 9999px !important;
        color: #334155 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.45rem 0.9rem !important;
        margin-bottom: 0.3rem !important;
        transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #FFFFFF !important;
        border-color: #6366F1 !important;
        color: #4338CA !important;
        box-shadow: 0 2px 5px rgba(99, 102, 241, 0.08) !important;
        transform: translateY(-1px) !important;
    }

    /* Clear conversation button */
    [data-testid="stSidebar"] div.clear-btn-wrap .stButton > button,
    [data-testid="stSidebar"] div:has(> button:contains("Clear conversation")) button {
        border-radius: 8px !important;
        color: #64748B !important;
        font-size: 0.78rem !important;
        background: transparent !important;
        border: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] div.clear-btn-wrap .stButton > button:hover {
        background: #FFF1F2 !important;
        border-color: #FECDD3 !important;
        color: #E11D48 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Feature Cards (How It Works) ──────────────────────────────────── */
    .how-it-works-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .how-it-works-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03);
    }
    .feature-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 2px;
    }
    .feature-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .feature-desc {
        font-size: 0.74rem;
        color: #64748B;
        line-height: 1.45;
    }

    /* ── Metric Cards ─────────────────────────────────────────────────── */
    [data-testid="metric-container"], [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover, [data-testid="stMetric"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 3px 6px rgba(15, 23, 42, 0.05) !important;
    }
    [data-testid="metric-container"] label, [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"], [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        font-family: var(--font-display) !important;
    }

    /* ── Chat Input Container & Fix for Contrast Bug ───────────────────── */
    [data-testid="stBottom"], .stBottom {
        background-color: transparent !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background-color: #F8FAFC !important;
        border-top: 1px solid transparent !important;
        padding-top: 0.75rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* High contrast chat input box */
    [data-testid="stChatInput"], .stChatInput {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05), 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 3px !important;
    }

    [data-testid="stChatInput"]:focus-within, .stChatInput:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18), 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }

    [data-testid="stChatInput"] textarea, 
    .stChatInputContainer textarea,
    [data-testid="stChatInputTextArea"] {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        font-size: 0.92rem !important;
        font-weight: 450 !important;
        font-family: var(--font-main) !important;
        line-height: 1.55 !important;
        padding: 10px 14px !important;
        caret-color: #6366F1 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInputContainer textarea::placeholder {
        color: #94A3B8 !important;
        font-weight: 400 !important;
    }

    [data-testid="stChatInputSubmitButton"] button {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9px !important;
        margin-right: 4px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stChatInputSubmitButton"] button:hover {
        background-color: #4F46E5 !important;
        transform: scale(1.03) !important;
    }
    [data-testid="stChatInputSubmitButton"] button svg {
        fill: #FFFFFF !important;
    }

    /* ── Chat Message Bubbles ─────────────────────────────────────────── */
    .message-row {
        display: flex;
        flex-direction: column;
        margin: 14px 0;
        width: 100%;
    }

    .user-bubble {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        border: 1px solid #4338CA;
        border-radius: 16px 16px 4px 16px;
        color: #FFFFFF !important;
        font-weight: 450;
        line-height: 1.6;
        font-size: 0.92rem;
        padding: 12px 18px;
        max-width: 82%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.18);
        word-break: break-word;
        white-space: pre-wrap;
    }

    .assistant-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px 16px 16px 16px;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.03);
        max-width: 95%;
        margin-right: auto;
        padding: 18px 20px;
        word-break: break-word;
    }

    .assistant-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
        flex-wrap: wrap;
        gap: 6px;
    }

    .analyst-identity {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .analyst-avatar {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: #EEF2FF;
        color: #6366F1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
    }

    .analyst-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* ── Tool Badges ──────────────────────────────────────────────────── */
    .tool-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-rag {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4338CA;
    }
    .badge-web {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #047857;
    }
    .badge-direct {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #64748B;
    }

    .meta-tag {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 500;
        background: #F8FAFC;
        border: 1px solid #F1F5F9;
        padding: 2px 7px;
        border-radius: 5px;
    }

    /* ── Assistant Typography & Formatting ────────────────────────────── */
    .assistant-body {
        font-size: 0.92rem;
        line-height: 1.68;
        color: #1E293B;
    }
    .assistant-body p {
        color: #1E293B !important;
        margin-bottom: 0.65rem;
    }
    .assistant-body p:last-child {
        margin-bottom: 0;
    }
    .assistant-body h1, .assistant-body h2, .assistant-body h3, .assistant-body h4 {
        color: #0F172A !important;
        font-weight: 700;
        font-family: var(--font-display) !important;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    .assistant-body strong {
        color: #0F172A !important;
        font-weight: 600;
    }
    .assistant-body ul, .assistant-body ol {
        margin-left: 1.25rem;
        margin-bottom: 0.65rem;
        color: #334155;
    }
    .assistant-body li {
        margin-bottom: 0.25rem;
    }
    .assistant-body code {
        background: #F1F5F9;
        color: #0F172A;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85em;
        border: 1px solid #E2E8F0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .assistant-body pre {
        background: #0F172A;
        color: #F8FAFC;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.85em;
        margin: 0.75rem 0;
        overflow-x: auto;
    }
    .assistant-body table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.75rem 0;
        font-size: 0.88rem;
    }
    .assistant-body th, .assistant-body td {
        border: 1px solid #E2E8F0;
        padding: 8px 12px;
        text-align: left;
    }
    .assistant-body th {
        background: #F8FAFC;
        color: #0F172A;
        font-weight: 600;
    }

    /* ── Main Header ──────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1.25rem 0;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #4F46E5;
        margin-bottom: 0.65rem;
    }
    .main-header h1 {
        font-family: var(--font-display) !important;
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        line-height: 1.2;
    }
    .main-header p {
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 450;
        line-height: 1.5;
        max-width: 580px;
        margin: 0 auto;
    }

    /* ── Model Pill ───────────────────────────────────────────────────── */
    .model-pill-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.25rem;
    }
    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        color: #334155;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .pulse-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
    }

    /* ── Empty State Hero Card ────────────────────────────────────────── */
    .empty-hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
    }
    .hero-icon {
        width: 48px;
        height: 48px;
        background: #EEF2FF;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: #6366F1;
        margin-bottom: 14px;
    }
    .hero-title {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        max-width: 460px;
        margin: 0 auto 18px auto;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

defaults = {
    "messages": [],       # Full chat history
    "total_queries": 0,   # Total questions asked
    "rag_hits": 0,        # Internal KB tool executions
    "web_hits": 0,        # Web search tool executions
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # ── Brand Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.4rem;">
        <div style="width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 15px; box-shadow: 0 2px 6px rgba(99, 102, 241, 0.25);">
            ⚡
        </div>
        <div>
            <div style="font-size: 0.96rem; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">TechVruk Analyst</div>
            <div style="font-size: 0.72rem; color: #64748B; font-weight: 500;">Autonomous Dual Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model Selector ────────────────────────────────────────────────────────
    selected_model = st.selectbox(
        "Active Intelligence Model",
        options=[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ],
        index=0,
        help="Select the Gemini foundation model used for reasoning and tool orchestration.",
        key="model_selector",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── How It Works (Authentic, Peer-Level Tone) ──────────────────────────────
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>System Workflow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px;">
        <div class="how-it-works-card">
            <div class="feature-title">
                <span class="feature-dot" style="background: #6366F1;"></span>
                Smart Routing
            </div>
            <div class="feature-desc">
                Knows instantly whether to check verified internal company records or browse the live web.
            </div>
        </div>
        <div class="how-it-works-card">
            <div class="feature-title">
                <span class="feature-dot" style="background: #10B981;"></span>
                Zero Hallucination
            </div>
            <div class="feature-desc">
                Pulls precise facts from FAISS vector storage.
            </div>
        </div>
        <div class="how-it-works-card">
            <div class="feature-title">
                <span class="feature-dot" style="background: #F59E0B;"></span>
                Real-Time Intelligence
            </div>
            <div class="feature-desc">
                Searches current global sources when questions go beyond company walls.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Suggested Queries (Rounded Interactive Pills) ─────────────────────────
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Suggested Queries</div>", unsafe_allow_html=True)
    
    suggested_queries = [
        "What is the stipend for AI/ML track?",
        "Who is TechVruk's CTO?",
        "What awards has TechVruk won?",
        "What is LangGraph?",
        "Latest AI developments today",
    ]
    for q in suggested_queries:
        if st.button(q, key=f"pill_{q[:16]}", use_container_width=True):
            st.session_state["prefill_query"] = q

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Clear Chat ────────────────────────────────────────────────────────────
    st.markdown('<div class="clear-btn-wrap">', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state["messages"] = []
        st.session_state["total_queries"] = 0
        st.session_state["rag_hits"] = 0
        st.session_state["web_hits"] = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN AREA — Header & Metrics
# =============================================================================

st.markdown("""
<div class="main-header">
    <div class="header-badge">Autonomous Hybrid Intelligence</div>
    <h1>Hybrid AI Research Analyst</h1>
    <p>Autonomous dual-engine intelligence: Verified internal enterprise knowledge meets live web search.</p>
</div>
""", unsafe_allow_html=True)

# Active Model Status Pill
st.markdown(
    f"""
    <div class="model-pill-container">
        <span class="model-pill">
            <span class="pulse-dot"></span>
            Orchestrator: <strong style="color:#0F172A">{selected_model}</strong>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Stats Row ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("💬 Queries", st.session_state["total_queries"])
col2.metric("🗂️ Internal KB Calls", st.session_state["rag_hits"])
col3.metric("🌐 Live Web Calls", st.session_state["web_hits"])

st.markdown("<hr style='margin: 1.2rem 0;'>", unsafe_allow_html=True)

# =============================================================================
# RENDER CHAT HISTORY
# =============================================================================

if len(st.session_state["messages"]) == 0:
    st.markdown("""
    <div class="empty-hero-card">
        <div class="hero-icon">⚡</div>
        <div class="hero-title">Ready for Your First Inquiry</div>
        <div class="hero-subtitle">
            Ask about TechVruk company structure, roles, awards, and stipends, or explore live global topics across the web.
        </div>
        <div style="font-size:0.75rem; color:#94A3B8; font-weight:500;">
            Select a suggested query from the sidebar or type below to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="message-row"><div class="user-bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # If this message was an error, re-render the styled notification
            if msg.get("is_error"):
                err_cls = "err-quota" if msg.get("error_type") == "quota" else "err-generic"
                st.markdown(
                    f'<div class="{err_cls}">{render_markdown(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Build tool badges
                badges_html = ""
                tools = msg.get("tools_used", [])
                for t in tools:
                    if t == "rag_search":
                        badges_html += '<span class="tool-badge badge-rag">🗂 Internal KB</span>'
                    elif t == "web_search":
                        badges_html += '<span class="tool-badge badge-web">🌐 Live Web</span>'

                if not badges_html:
                    badges_html = '<span class="tool-badge badge-direct">⚡ Direct Synthesis</span>'

                model_label = msg.get("active_model", "")
                meta_tag = f'<span class="meta-tag">{model_label}</span>' if model_label else ""

                body_html = render_markdown(msg["content"])

                card_html = (
                    f'<div class="message-row">'
                    f'<div class="assistant-card">'
                    f'<div class="assistant-header">'
                    f'<div class="analyst-identity">'
                    f'<div class="analyst-avatar">⚡</div>'
                    f'<span class="analyst-name">Research Analyst</span>'
                    f'{badges_html}'
                    f'</div>'
                    f'{meta_tag}'
                    f'</div>'
                    f'<div class="assistant-body">{body_html}</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

# =============================================================================
# CHAT INPUT & EXECUTION
# =============================================================================

# Retrieve and consume prefill from sidebar buttons if clicked
prefill = st.session_state.pop("prefill_query", "")

user_input = st.chat_input(placeholder="Ask about TechVruk or anything across the web…")

# If sidebar query pill was clicked and no direct input was typed
if prefill and not user_input:
    user_input = prefill

if user_input:
    # 1. Append user message to history and render immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(
        f'<div class="message-row"><div class="user-bubble">{user_input}</div></div>',
        unsafe_allow_html=True,
    )

    # 2. Invoke the agent with spinner feedback
    _agent_error: str | None = None   # None = success; str = error message to display
    _error_type: str = "generic"       # "quota" | "generic"

    with st.spinner("Synthesizing research…"):
        start_time = time.time()
        try:
            result = run_agent(user_input, model_name=selected_model)
            elapsed = round(time.time() - start_time, 2)

            answer       = result["answer"]
            tools_used   = result["tools_used"]
            active_model = result.get("active_model", selected_model)

        except Exception as e:
            elapsed      = round(time.time() - start_time, 2)
            tools_used   = []
            active_model = selected_model
            err_str      = str(e)

            # ── Classify: quota / rate-limit vs generic error ─────────────
            _quota_signals = (
                "429", "RESOURCE_EXHAUSTED", "quota",
                "rate limit", "rate_limit", "rateLimitExceeded", "Too Many Requests",
            )
            if any(sig.lower() in err_str.lower() for sig in _quota_signals):
                _error_type  = "quota"
                _agent_error = (
                    f"⚠️ **API Quota Exceeded**\n\n"
                    f"The model **{selected_model}** has temporarily hit its rate limit. "
                    f"This usually resolves within a minute, or you can switch to a lighter "
                    f"model from the sidebar right now.\n\n"
                    f"**Suggested alternatives:** "
                    f"`gemini-3.5-flash-lite` \u00b7 `gemini-3.1-flash-lite`"
                )
            else:
                _error_type  = "generic"
                _agent_error = (
                    f"❌ **Agent Error**\n\n"
                    f"`{err_str[:400]}`\n\n"
                    f"Please verify that `GOOGLE_API_KEY` is set correctly in your `.env` "
                    f"file and that `dummy_data.txt` exists in the project root."
                )

            answer = _agent_error   # saved to history as plain text

    # 3. Update session statistics (always, even on error)
    st.session_state["total_queries"] += 1
    if "rag_search" in tools_used:
        st.session_state["rag_hits"] += 1
    if "web_search" in tools_used:
        st.session_state["web_hits"] += 1

    # 4. Render: styled error notification OR normal assistant card
    if _agent_error:
        # ── Professional notification — never raw JSON ────────────────────
        if _error_type == "quota":
            st.markdown("""
<style>
.err-quota {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.9rem;
    color: #78350F;
    line-height: 1.65;
}
.err-quota strong { color: #92400E; }
.err-quota code {
    background: #FEF3C7;
    border: 1px solid #FCD34D;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.84em;
    color: #92400E;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
</style>""", unsafe_allow_html=True)
            st.markdown(
                f'<div class="err-quota">{render_markdown(_agent_error)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
<style>
.err-generic {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    border-left: 4px solid #F43F5E;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.9rem;
    color: #881337;
    line-height: 1.65;
}
.err-generic strong { color: #9F1239; }
.err-generic code {
    background: #FFE4E6;
    border: 1px solid #FECDD3;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.84em;
    color: #9F1239;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    word-break: break-all;
}
</style>""", unsafe_allow_html=True)
            st.markdown(
                f'<div class="err-generic">{render_markdown(_agent_error)}</div>',
                unsafe_allow_html=True,
            )

    else:
        # ── Normal assistant card ─────────────────────────────────────────
        badges_html = ""
        for t in tools_used:
            if t == "rag_search":
                badges_html += '<span class="tool-badge badge-rag">🗂 Internal KB</span>'
            elif t == "web_search":
                badges_html += '<span class="tool-badge badge-web">🌐 Live Web</span>'

        if not badges_html:
            badges_html = '<span class="tool-badge badge-direct">⚡ Direct Synthesis</span>'

        meta_tag  = f'<span class="meta-tag">{active_model} \u00b7 {elapsed}s</span>'
        body_html = render_markdown(answer)

        card_html = (
            f'<div class="message-row">'
            f'<div class="assistant-card">'
            f'<div class="assistant-header">'
            f'<div class="analyst-identity">'
            f'<div class="analyst-avatar">⚡</div>'
            f'<span class="analyst-name">Research Analyst</span>'
            f'{badges_html}'
            f'</div>'
            f'{meta_tag}'
            f'</div>'
            f'<div class="assistant-body">{body_html}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # 5. Persist response/error to session history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "tools_used": tools_used,
        "active_model": active_model,
        "is_error": bool(_agent_error),
        "error_type": _error_type if _agent_error else None,
    })

    # 6. Rerun to refresh the metric counters at the top
    st.rerun()
