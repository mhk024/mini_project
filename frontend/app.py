"""
frontend/app.py
===============
Multi-Mode AI Research Assistant — Streamlit UI

Modes:
  🎓 Student / Faculty  → Chat + simplified answers + key points + summary
  🔬 Author / Researcher → Analysis dashboard (quality, plagiarism, trends, suggestions)

Both modes share:
  • ML Pipeline Details toggle (original query → enhanced → retrieved → reranked → answer)
  • Auth (login / signup)
  • Document management (upload, switch, history)
"""

import re
import html as html_lib
import streamlit as st
import requests
import time
import uuid
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────────────────────────
# 🔗  API URLs
# ─────────────────────────────────────────────────────────────

BASE = os.getenv("API_URL", "").rstrip('/')
API_URL       = f"{BASE}/ask"
UPLOAD_URL    = f"{BASE}/upload"
LOGIN_URL     = f"{BASE}/login"
SIGNUP_URL    = f"{BASE}/signup"
HISTORY_URL   = f"{BASE}/history"
FILES_URL     = f"{BASE}/files"
SET_FILE_URL  = f"{BASE}/set_file"
ANA_QUALITY   = f"{BASE}/analyze/quality"
ANA_PLAGIARISM= f"{BASE}/analyze/plagiarism"
ANA_TRENDS    = f"{BASE}/analyze/trends"
ANA_SUGGEST   = f"{BASE}/analyze/suggestions"
EVAL_URL      = f"{BASE}/evaluate"
STATE_URL     = f"{BASE}/state"
DASH_URL      = f"{BASE}/dashboard"
DELETE_HISTORY_URL = f"{BASE}/history"

# Academic Intelligence API URLs
ACAD_SIMILAR   = f"{BASE}/academic/similar-papers"
ACAD_QUALITY   = f"{BASE}/academic/quality-index"
ACAD_TRENDS    = f"{BASE}/academic/trends"
ACAD_SUGGEST   = f"{BASE}/academic/suggestions"
ACAD_NOVELTY   = f"{BASE}/academic/novelty-check"
ACAD_FULL      = f"{BASE}/academic/full-analysis"
HEALTH_URL     = f"{BASE}/health"

st.set_page_config(
    page_title="ResearchBot — Multi-Mode Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def check_backend_health():
    try:
        r = requests.get(HEALTH_URL, timeout=120)
        return r.status_code == 200
    except:
        return False

# ─────────────────────────────────────────────────────────────
# 🎨  PREMIUM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', 'Inter', sans-serif; }

/* ── Gradient title ── */
h1 {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899, #6366f1);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
@keyframes gradientFlow {
    0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}
}

/* ── Modern Glassmorphism Card ── */
.ai-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.125);
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.6s ease-out;
}
.ai-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Pulse Animation for Active Analysis ── */
.pulse-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 2s infinite;
    margin-right: 8px;
}
@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* ── Researcher Dashboard Grid Enhancements ── */
.research-stat-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* ── Mode pill tabs ── */
.mode-bar { display:flex; gap:10px; margin:15px 0; }
.mode-pill {
    flex:1; padding:12px 0; border-radius:14px; text-align:center;
    font-size:14px; font-weight:600; cursor:pointer;
    border:1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}
.mode-pill.student { background: rgba(99, 102, 241, 0.1); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.3); }
.mode-pill.researcher { background: rgba(245, 158, 11, 0.1); color: #fcd34d; border-color: rgba(245, 158, 11, 0.3); }

/* ── Status steps ── */
.status-step { display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 14px; }
.status-step.done { color: #10b981; }
.status-step.active { color: #6366f1; font-weight: 600; }
.status-step.pending { color: #64748b; }

/* ── Scrollable paper list ── */
.papers-scroll {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.1);
}
.paper-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    transition: transform 0.2s, border-color 0.2s;
}
.paper-card:hover {
    transform: translateX(4px);
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(255, 255, 255, 0.06);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 🧠  SESSION STATE
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in":      False,
    "username":       "",
    "messages":       [],
    "past_history":   {},
    "session_id":     uuid.uuid4().hex,
    "loaded_file":    "",
    "mode":           "student",          # "student" | "researcher" | "eval"
    "show_ml":        False,
    "pipeline_data":  {},                 # {msg_idx: pipeline_dict}
    # Researcher analysis results (cached per doc)
    "res_quality":    None,
    "res_plagiarism": None,
    "res_trends":     None,
    "res_suggestions":None,
    # Academic Intelligence results (cached per doc)
    "academic_similar":    None,
    "academic_quality":    None,
    "academic_trends":     None,
    "academic_suggestions":None,
    "academic_novelty":    None,
    # Evaluation results (cached per doc)
    "eval_result":    None,
    "eval_job_id":    None,
    "dashboard_data": None,
    "confirm_delete_sid": None,
    "confirm_delete_title": None,
    "mode_store": {
        "student": {"messages": [], "pipeline_data": {}, "last_input": ""},
        "researcher": {
            "messages": [], "pipeline_data": {}, "last_input": "",
            "res_quality": None, "res_plagiarism": None, "res_trends": None, "res_suggestions": None,
        },
        "eval": {"eval_result": None, "eval_job_id": None, "last_input": "", "runs": []},
        "eval_engine": {"eval_result": None, "questions": ["What is the primary objective of this research?", "What is the core methodology used?", "What are the key findings?", "What are the main limitations identified?", "How does this work contribute to the field?"], "last_input": ""},
    },
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# URL-based login persistence
qp = st.query_params
if "user" in qp and not st.session_state.logged_in:
    st.session_state.logged_in = True
    st.session_state.username  = qp["user"]


# ─────────────────────────────────────────────────────────────
# 🔧  HELPER RENDERERS
# ─────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities from a string."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", str(text))
    clean = html_lib.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _normalize_message(msg):
    """Ensure every message dict has 'role' and 'content' keys."""
    if not isinstance(msg, dict):
        return {"role": "assistant", "content": str(msg)}
    return {
        "role": msg.get("role", "assistant"),
        "content": msg.get("content", ""),
    }


def _sbadge(n, cls):
    return f'<span class="sbadge {cls}">{n}</span>'


def score_badge(score):
    cls = "sc-hi" if score >= 7 else ("sc-md" if score >= 4 else "sc-lo")
    return f'<span class="score-badge {cls}">{score:.1f}/10</span>'


def risk_badge(risk: str):
    cls_map = {"Low": "risk-low", "Medium": "risk-med", "High": "risk-hi"}
    cls = cls_map.get(risk, "risk-lo")
    return f'<span class="{cls}">{risk}</span>'


def score_bar(label: str, value: float, color: str):
    pct = int(value * 10)  # value is 0-10 → 0-100%
    return f"""
    <div class="score-bar-wrap">
        <div class="score-label"><span>{label}</span><span style="color:{color};font-weight:600">{value:.1f}/10</span></div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>"""


def render_doc_card(doc: dict, idx: int):
    score   = doc.get("score", 0)
    source  = doc.get("source", "Unknown")
    content = doc.get("content", "")
    preview = (content[:160] + "…") if len(content) > 160 else content
    badge   = score_badge(score * 10) if score <= 1 else score_badge(score)
    st.markdown(f"""
    <div class="doc-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="doc-source">📎 {source}</span>{badge}
        </div>
        <div class="doc-preview">{preview}</div>
    </div>""", unsafe_allow_html=True)
    if len(content) > 160:
        with st.expander(f"📖 Full content — Doc {idx+1}", expanded=False):
            st.markdown(content)


def render_pipeline(data: dict):
    """Full 5-step RAG pipeline visualisation."""
    st.markdown(f"""
    <div class="ai-card step-1" style="animation-delay:0s">
        <div class="card-header">{_sbadge(1,'s1')} 🔍 Original Query</div>
        <div class="card-content">{data.get('original_query','—')}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    enh = data.get("enhanced_query", data.get("original_query", "—"))
    st.markdown(f"""
    <div class="ai-card step-2" style="animation-delay:0.08s">
        <div class="card-header">{_sbadge(2,'s2')} ✨ Enhanced Query
            <small style="color:#94a3b8;font-weight:400;text-transform:none;letter-spacing:0">(ML-rewritten)</small>
        </div>
        <div class="card-content">{enh}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    rdocs = data.get("retrieved_documents", [])
    st.markdown(f"""
    <div class="ai-card step-3" style="animation-delay:0.16s">
        <div class="card-header">{_sbadge(3,'s3')} 📄 Retrieved Documents
            <small style="color:#94a3b8;font-weight:400;text-transform:none;letter-spacing:0">({len(rdocs)} via vector similarity)</small>
        </div>
    </div>""", unsafe_allow_html=True)
    for i, d in enumerate(rdocs[:5]):
        render_doc_card(d, i)

    st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    rkdocs = data.get("documents", [])
    st.markdown(f"""
    <div class="ai-card step-4" style="animation-delay:0.24s">
        <div class="card-header">{_sbadge(4,'s4')} ⭐ Re-ranked Documents
            <small style="color:#94a3b8;font-weight:400;text-transform:none;letter-spacing:0">(Cross-Encoder scored)</small>
        </div>
    </div>""", unsafe_allow_html=True)
    for i, d in enumerate(rkdocs[:5]):
        render_doc_card(d, i)

    st.markdown('<div class="pipe-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-card step-5" style="animation-delay:0.32s">
        <div class="card-header">{_sbadge(5,'s5')} 🤖 Generated Answer</div>
        <div class="card-content">{data.get('answer','—')}</div>
    </div>""", unsafe_allow_html=True)


def loading_pipeline_animation(container, steps_list, active_idx):
    html = ""
    for j, (icon, txt) in enumerate(steps_list):
        if j < active_idx:
            html += f'<div class="prog-step prog-done">✅ {txt}</div>'
        elif j == active_idx:
            html += f'<div class="prog-step prog-active">{icon} {txt}</div>'
        else:
            html += f'<div class="prog-step prog-wait">⏳ {txt}</div>'
    container.markdown(f"""
    <div class="ai-card" style="opacity:1">
        <div class="card-header">⚙️ Processing RAG Pipeline…</div>{html}
    </div>""", unsafe_allow_html=True)

def _snapshot_mode_local(mode: str):
    if mode == "student":
        st.session_state.mode_store["student"] = {
            "messages": st.session_state.messages,
            "pipeline_data": st.session_state.pipeline_data,
            "last_input": st.session_state.mode_store["student"].get("last_input", ""),
        }
    elif mode == "researcher":
        st.session_state.mode_store["researcher"] = {
            "messages": st.session_state.messages,
            "pipeline_data": st.session_state.pipeline_data,
            "last_input": st.session_state.mode_store["researcher"].get("last_input", ""),
            "res_quality": st.session_state.res_quality,
            "res_plagiarism": st.session_state.res_plagiarism,
            "res_trends": st.session_state.res_trends,
            "res_suggestions": st.session_state.res_suggestions,
            "academic_similar": st.session_state.academic_similar,
            "academic_quality": st.session_state.academic_quality,
            "academic_trends": st.session_state.academic_trends,
            "academic_suggestions": st.session_state.academic_suggestions,
            "academic_novelty": st.session_state.academic_novelty,
        }
    elif mode == "eval":
        st.session_state.mode_store["eval"] = {
            "eval_result": st.session_state.eval_result,
            "eval_job_id": st.session_state.eval_job_id,
            "last_input": st.session_state.mode_store["eval"].get("last_input", ""),
            "runs": st.session_state.mode_store["eval"].get("runs", []),
        }
    else:
        st.session_state.mode_store["eval_engine"] = {
            "eval_result": st.session_state.mode_store["eval_engine"].get("eval_result"),
            "questions": st.session_state.mode_store["eval_engine"].get("questions"),
            "last_input": st.session_state.mode_store["eval_engine"].get("last_input", ""),
        }

def _restore_mode_local(mode: str):
    data = st.session_state.mode_store.get(mode, {})
    if mode == "student":
        st.session_state.messages = [_normalize_message(m) for m in data.get("messages", [])]
        st.session_state.pipeline_data = data.get("pipeline_data", {})
    elif mode == "researcher":
        st.session_state.messages = [_normalize_message(m) for m in data.get("messages", [])]
        st.session_state.pipeline_data = data.get("pipeline_data", {})
        st.session_state.res_quality = data.get("res_quality")
        st.session_state.res_plagiarism = data.get("res_plagiarism")
        st.session_state.res_trends = data.get("res_trends")
        st.session_state.res_suggestions = data.get("res_suggestions")
        st.session_state.academic_similar = data.get("academic_similar")
        st.session_state.academic_quality = data.get("academic_quality")
        st.session_state.academic_trends = data.get("academic_trends")
        st.session_state.academic_suggestions = data.get("academic_suggestions")
        st.session_state.academic_novelty = data.get("academic_novelty")
    elif mode == "eval":
        st.session_state.eval_result = data.get("eval_result")
        st.session_state.eval_job_id = data.get("eval_job_id")
    else:
        # eval_engine uses its own internal state in mode_store mostly
        pass

def _sync_mode_to_backend(mode: str):
    if not st.session_state.username:
        return
    data = st.session_state.mode_store.get(mode, {})
    try:
        requests.post(f"{STATE_URL}/{st.session_state.username}", json={"mode": mode, "data": data}, timeout=120)
    except Exception:
        pass

def _load_full_state_from_backend():
    if not st.session_state.username:
        return
    try:
        r = requests.get(f"{STATE_URL}/{st.session_state.username}", timeout=120)
        if r.status_code != 200:
            return
        payload = r.json().get("state", {})
        student_chat = payload.get("Student", {}).get("chat", [])
        student_messages = []
        for item in student_chat:
            q = item.get("query")
            a = item.get("response")
            if q:
                student_messages.append({"role": "user", "content": q})
            if a:
                student_messages.append({"role": "assistant", "content": a})
        st.session_state.mode_store["student"] = {
            "messages": student_messages,
            "pipeline_data": {},
            "last_input": payload.get("Student", {}).get("last_input", ""),
        }
        research_chat = payload.get("Research", {}).get("chat", [])
        research_messages = []
        for item in research_chat:
            q = item.get("query")
            a = item.get("response")
            if q:
                research_messages.append({"role": "user", "content": q})
            if a:
                research_messages.append({"role": "assistant", "content": a})
        st.session_state.mode_store["researcher"] = {
            "messages": research_messages,
            "pipeline_data": {},
            "last_input": payload.get("Research", {}).get("last_input", ""),
            "res_quality": payload.get("Research", {}).get("insights", {}).get("quality"),
            "res_plagiarism": payload.get("Research", {}).get("insights", {}).get("plagiarism"),
            "res_trends": payload.get("Research", {}).get("insights", {}).get("trends"),
            "res_suggestions": payload.get("Research", {}).get("insights", {}).get("suggestions"),
        }
        st.session_state.mode_store["eval"] = {
            "eval_result": payload.get("Evaluate", {}).get("last_results"),
            "eval_job_id": None,
            "last_input": payload.get("Evaluate", {}).get("last_input", ""),
            "runs": payload.get("Evaluate", {}).get("runs", []),
        }
    except Exception:
        pass

def _snapshot_mode_local(mode: str):
    if mode == "student":
        st.session_state.mode_store["student"] = {
            "messages": st.session_state.messages,
            "pipeline_data": st.session_state.pipeline_data,
            "last_input": st.session_state.mode_store["student"].get("last_input", ""),
        }
    elif mode == "researcher":
        st.session_state.mode_store["researcher"] = {
            "messages": st.session_state.messages,
            "pipeline_data": st.session_state.pipeline_data,
            "last_input": st.session_state.mode_store["researcher"].get("last_input", ""),
            "res_quality": st.session_state.res_quality,
            "res_plagiarism": st.session_state.res_plagiarism,
            "res_trends": st.session_state.res_trends,
            "res_suggestions": st.session_state.res_suggestions,
        }
    else:
        st.session_state.mode_store["eval"] = {
            "eval_result": st.session_state.eval_result,
            "eval_job_id": st.session_state.eval_job_id,
            "last_input": st.session_state.mode_store["eval"].get("last_input", ""),
            "runs": st.session_state.mode_store["eval"].get("runs", []),
        }

def _restore_mode_local(mode: str):
    data = st.session_state.mode_store.get(mode, {})
    if mode == "student":
        st.session_state.messages = data.get("messages", [])
        st.session_state.pipeline_data = data.get("pipeline_data", {})
    elif mode == "researcher":
        st.session_state.messages = data.get("messages", [])
        st.session_state.pipeline_data = data.get("pipeline_data", {})
        st.session_state.res_quality = data.get("res_quality")
        st.session_state.res_plagiarism = data.get("res_plagiarism")
        st.session_state.res_trends = data.get("res_trends")
        st.session_state.res_suggestions = data.get("res_suggestions")
    else:
        st.session_state.eval_result = data.get("eval_result")
        st.session_state.eval_job_id = data.get("eval_job_id")

def _sync_mode_to_backend(mode: str):
    if not st.session_state.username:
        return
    data = st.session_state.mode_store.get(mode, {})
    try:
        requests.post(f"{STATE_URL}/{st.session_state.username}", json={"mode": mode, "data": data}, timeout=120)
    except Exception:
        pass

def _load_full_state_from_backend():
    if not st.session_state.username:
        return
    try:
        r = requests.get(f"{STATE_URL}/{st.session_state.username}", timeout=120)
        if r.status_code != 200:
            return
        payload = r.json().get("state", {})
        st.session_state.mode_store["student"] = {
            "messages": payload.get("Student", {}).get("chat", []),
            "pipeline_data": payload.get("Student", {}).get("last_pipeline", {}),
            "last_input": payload.get("Student", {}).get("last_input", ""),
        }
        st.session_state.mode_store["researcher"] = {
            "messages": payload.get("Research", {}).get("chat", []),
            "pipeline_data": payload.get("Research", {}).get("last_pipeline", {}),
            "last_input": payload.get("Research", {}).get("last_input", ""),
            "res_quality": payload.get("Research", {}).get("insights", {}).get("quality"),
            "res_plagiarism": payload.get("Research", {}).get("insights", {}).get("plagiarism"),
            "res_trends": payload.get("Research", {}).get("insights", {}).get("trends"),
            "res_suggestions": payload.get("Research", {}).get("insights", {}).get("suggestions"),
        }
        st.session_state.mode_store["eval"] = {
            "eval_result": payload.get("Evaluate", {}).get("last_results"),
            "eval_job_id": None,
            "last_input": payload.get("Evaluate", {}).get("last_input", ""),
            "runs": payload.get("Evaluate", {}).get("runs", []),
        }
    except Exception:
        pass


def _delete_chat_session(session_id: str):
    """Delete a specific chat session from backend and local state."""
    username = st.session_state.username
    if not username or not session_id:
        return False
    try:
        r = requests.delete(f"{DELETE_HISTORY_URL}/{username}/{session_id}", timeout=120)
        if r.status_code == 200:
            # Remove from local past_history
            if session_id in st.session_state.past_history:
                del st.session_state.past_history[session_id]
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────
# 🔐  AUTH HELPERS
# ─────────────────────────────────────────────────────────────
def login_user(username, password):
    with st.spinner("🔐 Authenticating…"):
        res = requests.post(LOGIN_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username  = username
        st.query_params["user"]    = username
        try:
            h = requests.get(f"{HISTORY_URL}/{username}")
            if h.status_code == 200:
                st.session_state.past_history = h.json().get("history", {})
        except:
            pass
        st.session_state.messages   = []
        st.session_state.session_id = uuid.uuid4().hex
        _load_full_state_from_backend()
        _restore_mode_local(st.session_state.mode)
        st.session_state._state_bootstrapped = True
        st.toast(f"Welcome back, {username}!", icon="👋")
        time.sleep(0.4)
        st.rerun()
    else:
        st.error("❌ Invalid credentials")


def signup_user(username, password):
    with st.spinner("📝 Creating account…"):
        res = requests.post(SIGNUP_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.toast("Account created! You can now login.", icon="🎉")
        st.balloons()
    else:
        st.error(res.json().get("detail", "Signup failed"))

if st.session_state.logged_in and not st.session_state.get("_state_bootstrapped"):
    _load_full_state_from_backend()
    _restore_mode_local(st.session_state.mode)
    st.session_state._state_bootstrapped = True


# ─────────────────────────────────────────────────────────────
# 🔑  LOGIN PAGE
# ─────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align:center;font-size:2.4rem'>🧠 ResearchAI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#94a3b8;font-size:16px;margin-bottom:28px'>Multi-Mode AI Research Assistant — Student & Researcher</p>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["🔐 Login", "📝 Signup"])
        with t1:
            with st.container(border=True):
                u = st.text_input("Username", key="lu", placeholder="Your username")
                p = st.text_input("Password", type="password", key="lp", placeholder="Your password")
                st.markdown("")
                if st.button("🔓 Login", use_container_width=True, type="primary"):
                    login_user(u, p) if (u and p) else st.warning("Enter both fields")
        with t2:
            with st.container(border=True):
                u = st.text_input("New Username", key="su", placeholder="Choose a username")
                p = st.text_input("New Password", type="password", key="sp", placeholder="Choose a password")
                st.markdown("")
                if st.button("🚀 Create Account", use_container_width=True, type="primary"):
                    signup_user(u, p) if (u and p) else st.warning("Enter both fields")
    st.stop()


# ─────────────────────────────────────────────────────────────
# 🧭  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Health Check Indicator
    is_healthy = check_backend_health()
    health_color = "#10b981" if is_healthy else "#f87171"
    health_txt = "Backend Online" if is_healthy else "Backend Offline"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:10px;background:rgba(255,255,255,0.03);border-radius:12px;border:1px solid rgba(255,255,255,0.05);margin-bottom:15px">
        <span class="pulse-indicator" style="background:{health_color};box-shadow:0 0 0 0 {health_color}44"></span>
        <span style="font-size:13px;font-weight:600;color:#94a3b8">{health_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 👤 **{st.session_state.username}**")

    # ── Mode selector ──
    st.markdown("#### 🎛 Mode")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        if st.button("🎓 Student", use_container_width=True,
                     type="primary" if st.session_state.mode == "student" else "secondary"):
            _snapshot_mode_local(st.session_state.mode)
            _sync_mode_to_backend(st.session_state.mode)
            st.session_state.mode = "student"
            _restore_mode_local("student")
            try:
                requests.post(f"{STATE_URL}/{st.session_state.username}/switch", json={"mode": "student"}, timeout=120)
            except Exception:
                pass
            st.rerun()
    with mc2:
        if st.button("🔬 Research", use_container_width=True,
                     type="primary" if st.session_state.mode == "researcher" else "secondary"):
            _snapshot_mode_local(st.session_state.mode)
            _sync_mode_to_backend(st.session_state.mode)
            st.session_state.mode = "researcher"
            _restore_mode_local("researcher")
            try:
                requests.post(f"{STATE_URL}/{st.session_state.username}/switch", json={"mode": "researcher"}, timeout=120)
            except Exception:
                pass
            st.rerun()
    with mc3:
        if st.button("📊 Evaluate", use_container_width=True,
                     type="primary" if st.session_state.mode == "eval" else "secondary"):
            _snapshot_mode_local(st.session_state.mode)
            _sync_mode_to_backend(st.session_state.mode)
            st.session_state.mode = "eval"
            _restore_mode_local("eval")
            try:
                requests.post(f"{STATE_URL}/{st.session_state.username}/switch", json={"mode": "eval"}, timeout=120)
            except Exception:
                pass
            st.rerun()
    with mc4:
        if st.button("🎓 Engine", use_container_width=True,
                     type="primary" if st.session_state.mode == "eval_engine" else "secondary",
                     help="Academic Evaluation Engine"):
            _snapshot_mode_local(st.session_state.mode)
            _sync_mode_to_backend(st.session_state.mode)
            st.session_state.mode = "eval_engine"
            _restore_mode_local("eval_engine")
            try:
                requests.post(f"{STATE_URL}/{st.session_state.username}/switch", json={"mode": "eval_engine"}, timeout=120)
            except Exception:
                pass
            st.rerun()

    _lbl = {"student": "🎓 Student / Faculty", "researcher": "🔬 Author / Researcher", "eval": "📊 ML Evaluation", "eval_engine": "🎓 Academic Eval Engine"}
    st.caption(f"Active: **{_lbl.get(st.session_state.mode, '')}**")
    try:
        d = requests.get(f"{DASH_URL}/{st.session_state.username}", timeout=120)
        if d.status_code == 200:
            st.session_state.dashboard_data = d.json()
    except Exception:
        pass

    dash = st.session_state.dashboard_data or {}
    if dash:
        st.markdown("#### 📈 Performance")
        for mk, label in [("Student", "Student"), ("Research", "Research"), ("Evaluate", "Evaluate"), ("EvalEngine", "Engine")]:
            m = dash.get(mk, {})
            st.caption(
                f"{label}: q={m.get('total_queries',0)} | avg={m.get('avg_response_time_ms',0):.1f}ms | "
                f"cache={m.get('cache_hit_rate',0)*100:.1f}%"
            )

    st.divider()

    # ── ML details toggle ──
    st.session_state.show_ml = st.toggle(
        "🔬 Show ML Pipeline Details",
        value=st.session_state.show_ml,
        help="Show full RAG pipeline: query enhancement → retrieval → re-ranking → generation"
    )
    st.caption("📊 Pipeline ON" if st.session_state.show_ml else "💬 Pipeline OFF")

    st.divider()

    # ── Quick actions ──
    ca, cb = st.columns(2)
    with ca:
        if st.button("🗑 Clear Current Mode", use_container_width=True):
            try:
                requests.post(
                    f"{STATE_URL}/{st.session_state.username}/clear_mode/{st.session_state.mode}",
                    timeout=120
                )
            except Exception:
                pass
            if st.session_state.mode in ["student", "researcher"]:
                st.session_state.messages = []
                st.session_state.pipeline_data = {}
            if st.session_state.mode == "researcher":
                st.session_state.res_quality = None
                st.session_state.res_plagiarism = None
                st.session_state.res_trends = None
                st.session_state.res_suggestions = None
            if st.session_state.mode == "eval":
                st.session_state.eval_result = None
                st.session_state.eval_job_id = None
            _snapshot_mode_local(st.session_state.mode)
            st.toast("Current mode cleared", icon="🧹"); time.sleep(0.4); st.rerun()
    with cb:
        if st.button("🗑 Clear All Modes", use_container_width=True):
            try:
                requests.post(f"{STATE_URL}/{st.session_state.username}/clear_all", timeout=120)
            except Exception:
                pass
            st.session_state.mode_store = DEFAULTS["mode_store"].copy()
            st.session_state.messages = []
            st.session_state.pipeline_data = {}
            st.session_state.res_quality = None
            st.session_state.res_plagiarism = None
            st.session_state.res_trends = None
            st.session_state.res_suggestions = None
            st.session_state.eval_result = None
            st.session_state.eval_job_id = None
            st.toast("All modes cleared", icon="🧹"); time.sleep(0.4); st.rerun()

    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.clear(); st.query_params.clear()
            st.toast("Logged out", icon="👋"); time.sleep(0.4); st.rerun()

    st.divider()

    # ── Document management ──
    st.subheader("📂 Documents")
    try:
        files = requests.get(FILES_URL, timeout=120).json().get("files", [])
    except:
        files = []

    selected = st.selectbox("Select file", [""] + files)
    if selected:
        if st.button("📥 Load File", use_container_width=True, type="primary"):
            st.session_state.loaded_file = ""
            # Reset all cached results
            for k in ["res_quality","res_plagiarism","res_trends","res_suggestions","eval_result",
                      "academic_similar","academic_quality","academic_trends","academic_suggestions","academic_novelty"]:
                st.session_state[k] = None
            with st.spinner(f"Loading `{selected}`…"):
                r = requests.post(SET_FILE_URL, json={"filename": selected}, timeout=600)
            if r.status_code == 200:
                st.session_state.loaded_file = selected
                st.toast(f"Loaded {selected}", icon="✅"); time.sleep(0.4); st.rerun()
            else:
                msg = r.text
                try: msg = r.json().get("detail", msg)
                except: pass
                st.error(f"Failed: {msg}")

    if st.session_state.loaded_file:
        st.success(f"📄 **Active:** {st.session_state.loaded_file}")
    else:
        st.info("No document active.")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages      = []
        st.session_state.pipeline_data = {}
        st.session_state.session_id    = uuid.uuid4().hex
        st.rerun()

    # ── Chat history (student mode) ──
    st.subheader("📝 Recent Chats")
    loaded_doc = st.session_state.loaded_file
    past = st.session_state.past_history

    # Confirmation dialog for deleting a chat
    confirm_sid = st.session_state.get("confirm_delete_sid")
    if confirm_sid:
        confirm_title = st.session_state.get("confirm_delete_title", "this chat")
        st.markdown(
            f"<div style='background:rgba(248,113,113,0.12);border:1px solid rgba(248,113,113,0.3);"
            f"border-radius:10px;padding:12px;margin:8px 0;color:#e2e8f0;font-size:14px'>"
            f"🗑 <strong>Delete this chat?</strong><br><span style='color:#94a3b8'>{confirm_title}</span></div>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Delete", key="confirm_del", use_container_width=True, type="primary"):
                deleted_current = (confirm_sid == st.session_state.session_id)
                if _delete_chat_session(confirm_sid):
                    st.session_state.confirm_delete_sid = None
                    st.session_state.confirm_delete_title = None
                    st.toast("Chat deleted", icon="🗑")
                    # If we deleted the currently open chat, switch to newest remaining or fresh chat
                    if deleted_current:
                        remaining = [
                            (s, d) for s, d in st.session_state.past_history.items()
                            if d.get("messages", []) and d["messages"][-1].get("filename") == loaded_doc
                        ]
                        if remaining:
                            # Switch to the most recently updated remaining chat
                            newest_sid, newest_data = remaining[-1]
                            st.session_state.session_id = newest_sid
                            st.session_state.messages = [_normalize_message(m) for m in newest_data.get("messages", [])]
                            st.session_state.pipeline_data = {}
                        else:
                            # No remaining chats for this doc → start fresh
                            st.session_state.messages = []
                            st.session_state.pipeline_data = {}
                            st.session_state.session_id = uuid.uuid4().hex
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Failed to delete chat")
        with c2:
            if st.button("❌ Cancel", key="cancel_del", use_container_width=True, type="secondary"):
                st.session_state.confirm_delete_sid = None
                st.session_state.confirm_delete_title = None
                st.rerun()
        st.divider()

    if loaded_doc and past and isinstance(past, dict):
        shown = False
        for sid, sdata in reversed(list(past.items())):
            msgs = sdata.get("messages", [])
            fname = msgs[-1].get("filename") if msgs else None
            if fname == loaded_doc:
                shown = True
                title    = sdata.get("title", "Chat")
                is_act   = (sid == st.session_state.session_id)
                btype    = "primary" if is_act else "secondary"
                # Use columns to place chat button and delete icon side by side
                cbtn, cdel = st.columns([5, 1])
                with cbtn:
                    if st.button(f"💬 {title}", key=f"h_{sid}", use_container_width=True, type=btype):
                        st.session_state.session_id    = sid
                        st.session_state.messages      = [_normalize_message(m) for m in msgs]
                        st.session_state.pipeline_data = {}
                        st.rerun()
                with cdel:
                    if st.button("🗑", key=f"del_{sid}", use_container_width=True, type="secondary"):
                        st.session_state.confirm_delete_sid = sid
                        st.session_state.confirm_delete_title = title
                        st.rerun()
        if not shown:
            st.info(f"No history for '{loaded_doc}'.")
    else:
        st.info("Select a document first." if not loaded_doc else "No history yet.")

    st.divider()

    # ── Upload ──
    st.subheader("⬆ Upload File")
    ufile = st.file_uploader("PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
    if ufile:
        if st.button("☁️ Upload to KB", use_container_width=True, type="primary"):
            st.session_state.loaded_file = ""
            for k in ["res_quality","res_plagiarism","res_trends","res_suggestions",
                      "academic_similar","academic_quality","academic_trends","academic_suggestions","academic_novelty"]:
                st.session_state[k] = None
            with st.status("Uploading…", expanded=True) as status:
                st.write("Sending to server…")
                fp = {"file": (ufile.name, ufile.getvalue(), ufile.type)}
                r  = requests.post(UPLOAD_URL, files=fp, timeout=300)
                if r.status_code == 200:
                    status.update(label="Done!", state="complete", expanded=False)
                    st.session_state.loaded_file = ufile.name
                    st.toast(f"'{ufile.name}' uploaded!", icon="🚀")
                    st.balloons(); time.sleep(1); st.rerun()
                else:
                    msg = r.text
                    try: msg = r.json().get("detail", msg)
                    except: pass
                    status.update(label="Failed", state="error", expanded=True)
                    st.error(f"Error: {msg}")


# ─────────────────────────────────────────────────────────────
# 🏠  MAIN AREA — header
# ─────────────────────────────────────────────────────────────
_TITLES = {
    "student":    ("🎓 Student / Faculty Mode",   "Ask questions about your research paper and get simplified, structured answers."),
    "researcher": ("🔬 Author / Researcher Mode",   "Analyze your research paper for quality, originality, trends, and improvements."),
    "eval":       ("📊 ML Evaluation Mode",          "Benchmark the RAG pipeline across standard research questions with full ML metrics."),
}
_t, _c = _TITLES.get(st.session_state.mode, _TITLES["student"])
st.title(_t)
st.caption(_c)

if st.session_state.loaded_file:
    st.markdown(
        f"<div style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);"
        f"border-radius:10px;padding:8px 16px;margin-bottom:12px;color:#a5b4fc;font-size:14px'>"
        f"📄 &nbsp;<strong>Active document:</strong> {st.session_state.loaded_file}</div>",
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Please load or upload a document from the sidebar.")

if st.session_state.show_ml:
    st.markdown(
        "<div style='background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(56,189,248,0.08));"
        "border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:10px 18px;"
        "margin-bottom:14px;display:flex;align-items:center;gap:10px'>"
        "<span class='pulse-indicator'></span>"
        "<span style='color:#c4b5fd;font-weight:600'>ML Pipeline Mode ON</span>"
        "<span style='color:#64748b;font-size:13px'>— Full pipeline shown after each response.</span></div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# 🎓  STUDENT MODE
# ─────────────────────────────────────────────────────────────
if st.session_state.mode == "student":

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#64748b">
            <div style="font-size:44px;margin-bottom:14px">🎓</div>
            <h3 style="color:#94a3b8;-webkit-text-fill-color:#94a3b8;font-weight:500">Ready to help you understand</h3>
            <p style="font-size:15px">Upload a research paper and ask any question — you'll get a clear answer, simplified explanation, key points, and a summary.</p>
        </div>""", unsafe_allow_html=True)

    # Display history
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

        if msg.get("role", "assistant") == "assistant":
            pid = str(idx)
            pdata = st.session_state.pipeline_data.get(pid, {})

            # Student extras
            if pdata:
                tabs = st.tabs(["💡 Simplified", "📝 Key Points", "📄 Summary", "🔬 ML Pipeline" if st.session_state.show_ml else ""])

                with tabs[0]:  # Simplified explanation
                    simp = pdata.get("simplified_explanation", "")
                    if simp:
                        st.markdown(f"""
                        <div class="ai-card step-simp" style="opacity:1;animation:none">
                            <div class="card-header">💡 Simplified Explanation</div>
                            <div class="card-content">{simp}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info("No simplified explanation available.")

                with tabs[1]:  # Key points
                    kps = pdata.get("key_points", [])
                    if kps:
                        items_html = "".join(
                            f'<div class="kp-item"><span class="kp-bullet">▸</span><span>{kp}</span></div>'
                            for kp in kps
                        )
                        st.markdown(f"""
                        <div class="ai-card step-kp" style="opacity:1;animation:none">
                            <div class="card-header">📝 Key Points</div>
                            {items_html}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info("No key points extracted.")

                with tabs[2]:  # Summary
                    summ = pdata.get("summary", {})
                    if summ:
                        st.markdown(f"""
                        <div class="ai-card step-sum" style="opacity:1;animation:none">
                            <div class="card-header">📄 Short Summary</div>
                            <div class="card-content">{summ.get('short','—')}</div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("")
                        if summ.get("detailed"):
                            with st.expander("📋 Detailed Summary", expanded=False):
                                st.markdown(summ.get("detailed", ""))

                if st.session_state.show_ml and len(tabs) > 3:
                    with tabs[3]:  # Pipeline
                        render_pipeline(pdata)

    # Chat input
    if prompt := st.chat_input("Ask something about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder  = st.empty()
            full_resp    = ""
            pipeline_res = None

            try:
                with st.status("🧠 Processing RAG Pipeline...", expanded=st.session_state.show_ml) as status:
                    st.write("🔍 Analyzing query...")
                    time.sleep(0.1)
                    st.write("✨ Enhancing with ML context...")
                    
                    r = requests.post(API_URL, json={
                        "username":   st.session_state.username,
                        "question":   prompt,
                        "filename":   st.session_state.loaded_file,
                        "session_id": st.session_state.session_id,
                        "mode":       "student",
                    }, timeout=300)

                    if r.status_code == 200:
                        st.write("📄 Documents retrieved and ranked.")
                        st.write("🤖 Generating answer...")
                        data         = r.json()
                        answer       = data.get("answer", "")
                        pipeline_res = data
                        full_resp = answer
                        status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                        placeholder.markdown(full_resp)
                        cache_txt = "Hit" if data.get("cache_hit") else "Miss"
                        st.caption(f"Response time: {data.get('execution_time_ms', 0)} ms | Cache: {cache_txt}")
                    else:
                        status.update(label="❌ Backend Error", state="error")
                        st.error(f"Backend error: {r.status_code}")

            except Exception as e:
                st.error(f"⚠️ Network error: {e}")

        if full_resp:
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
            st.session_state.mode_store["student"]["last_input"] = prompt
            if pipeline_res:
                idx = str(len(st.session_state.messages) - 1)
                st.session_state.pipeline_data[idx] = pipeline_res

            sid = st.session_state.session_id
            if sid not in st.session_state.past_history:
                st.session_state.past_history[sid] = {
                    "title": (prompt[:30] + "…") if len(prompt) > 30 else prompt,
                    "messages": st.session_state.messages.copy()
                }
            else:
                st.session_state.past_history[sid]["messages"] = st.session_state.messages.copy()
            _snapshot_mode_local("student")
            _sync_mode_to_backend("student")


# ─────────────────────────────────────────────────────────────
# 🔬  RESEARCHER MODE
# ─────────────────────────────────────────────────────────────
elif st.session_state.mode == "researcher":
    st.markdown("### 📊 Research Analysis Dashboard")
    st.markdown("Run each analysis independently. Results are cached until you load a different document.")

    if not st.session_state.loaded_file:
        st.info("👆 Load a document from the sidebar to start analysis.")
        st.stop()

    # ── 4 analysis cards in 2×2 grid ──
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)

    # ── 1. Quality Index ─────────────────────────────────────
    with row1_c1:
        st.markdown("""
        <div class="ai-card" style="opacity:1;animation:none;min-height:200px">
            <div class="card-header">📊 Quality Index</div>
        </div>""", unsafe_allow_html=True)

        q_data = st.session_state.res_quality
        if q_data:
            # Support both old (quality{}) and new (scores{}) schema shapes
            q = q_data.get("scores") or q_data.get("quality", {})
            bar_colors = {
                "Clarity":       ("clarity",       "#818cf8"),
                "Novelty":       ("novelty",        "#f472b6"),
                "Readability":   ("readability",    "#34d399"),
                "Tech Depth":    ("technical_depth","#38bdf8"),
                "Methodology":   ("methodology",    "#fbbf24"),
            }
            bars_html = "".join(
                score_bar(lbl, q.get(key, 0), color)
                for lbl, (key, color) in bar_colors.items()
                if q.get(key) is not None
            )
            st.markdown(bars_html, unsafe_allow_html=True)

            # Metric row — show up to 4
            shown_keys = [(lbl, k) for lbl, (k, _) in bar_colors.items() if q.get(k) is not None][:4]
            cols = st.columns(len(shown_keys))
            for col, (lbl, key) in zip(cols, shown_keys):
                with col:
                    st.metric(lbl, f"{q.get(key, 0):.1f}")
        else:
            st.markdown("<p style='color:#64748b;font-size:14px'>Scores clarity, novelty, and readability of your paper using AI.</p>", unsafe_allow_html=True)

        if st.button("▶ Run Quality Analysis", key="btn_quality", use_container_width=True, type="primary"):
            with st.status("📊 Analyzing Quality...", expanded=True) as status:
                st.write("🧠 Scoring paper structure...")
                r = requests.post(ANA_QUALITY, timeout=120)
                if r.status_code == 200:
                    st.session_state.res_quality = r.json()
                    status.update(label="✅ Quality Analysis Done", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {r.text[:200]}")

    # ── 2. Plagiarism & Overlap Detection ──────────────────────────────
    with row1_c2:
        st.markdown("""
        <div class="ai-card" style="opacity:1;animation:none;min-height:200px">
            <div class="card-header">🔍 Plagiarism & Overlap Check</div>
        </div>""", unsafe_allow_html=True)

        p_data = st.session_state.res_plagiarism
        if p_data:
            plag = p_data.get("plagiarism", {})
            risk = plag.get("plagiarism_risk", "Unknown")
            novelty = plag.get("novelty_score", "N/A")
            overlap = plag.get("overlap_analysis", "")
            
            rb = risk_badge(risk)
            st.markdown(f"""
            <div style="margin:10px 0">
                <span style="color:#94a3b8;font-size:14px">Risk Level: </span>{rb}
                <span style="color:#94a3b8;font-size:14px;margin-left:15px">Novelty Score: <span style="color:#34d399;font-weight:700">{novelty}</span></span>
            </div>
            """, unsafe_allow_html=True)

            if overlap:
                with st.expander("🔎 Overlap Analysis", expanded=True):
                    st.markdown(f"<div style='color:#cbd5e1;font-size:14px;line-height:1.6'>{overlap}</div>", unsafe_allow_html=True)

            similar_summaries = plag.get("similar_papers_summary", [])
            if similar_summaries:
                with st.expander("📚 Similar Papers Summary", expanded=False):
                    for idx, summ in enumerate(similar_summaries):
                        st.markdown(f"<div style='margin-bottom:8px;font-size:13px;color:#e2e8f0'><b>{idx+1}.</b> {summ}</div>", unsafe_allow_html=True)

            missing_refs = plag.get("missing_references", [])
            if missing_refs:
                with st.expander("⚠️ Missing References", expanded=False):
                    for idx, ref in enumerate(missing_refs):
                        st.markdown(f"<div style='color:#f87171;font-size:13px'>- {ref}</div>", unsafe_allow_html=True)
                        
            improvements = plag.get("improvements", [])
            if improvements:
                with st.expander("💡 Improvements", expanded=False):
                    for idx, imp in enumerate(improvements):
                        st.markdown(f"<div style='color:#fbbf24;font-size:13px'>- {imp}</div>", unsafe_allow_html=True)
                        
        else:
            st.markdown("<p style='color:#64748b;font-size:14px'>Checks overlap, novelty, and missing citations against real-world similar published papers.</p>", unsafe_allow_html=True)

        if st.button("▶ Run Plagiarism Check", key="btn_plag", use_container_width=True, type="primary"):
            with st.status("🔍 Checking Plagiarism...", expanded=True) as status:
                st.write("🌐 Searching external databases...")
                r = requests.post(ANA_PLAGIARISM, timeout=180)
                if r.status_code == 200:
                    st.session_state.res_plagiarism = r.json()
                    status.update(label="✅ Plagiarism Check Done", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Check Failed", state="error")
                    st.error(f"Error: {r.text[:200]}")

    # ── 3. Trend Analysis ────────────────────────────────────
    with row2_c1:
        st.markdown("""
        <div class="ai-card" style="opacity:1;animation:none;min-height:200px">
            <div class="card-header">📈 Trend Analysis</div>
        </div>""", unsafe_allow_html=True)

        t_data = st.session_state.res_trends
        if t_data:
            trend_text = t_data.get("trend_analysis", "No analysis available.")
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.15);
                        border-radius:12px;padding:16px 18px;color:#e2e8f0;font-size:14px;line-height:1.75">
                {trend_text.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#64748b;font-size:14px'>Evaluates how relevant your paper is to current research trends (2024-2025) and suggests missing modern topics.</p>", unsafe_allow_html=True)

        if st.button("▶ Run Trend Analysis", key="btn_trends", use_container_width=True, type="primary"):
            with st.status("📈 Analyzing Research Trends...", expanded=True) as status:
                st.write("🌐 Fetching market trends...")
                r = requests.post(ANA_TRENDS, timeout=120)
                if r.status_code == 200:
                    st.session_state.res_trends = r.json()
                    status.update(label="✅ Trend Analysis Done", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {r.text[:200]}")

    # ── 4. Improvement Suggestions ───────────────────────────
    with row2_c2:
        st.markdown("""
        <div class="ai-card" style="opacity:1;animation:none;min-height:200px">
            <div class="card-header">💡 Improvement Suggestions</div>
        </div>""", unsafe_allow_html=True)

        s_data = st.session_state.res_suggestions
        if s_data:
            suggs = s_data.get("suggestions", [])
            if suggs:
                items_html = "".join(
                    f'<div class="sug-item"><span class="sug-bullet">#{i+1}</span><span>{sug}</span></div>'
                    for i, sug in enumerate(suggs)
                )
                st.markdown(items_html, unsafe_allow_html=True)
            else:
                st.info("No suggestions generated.")
        else:
            st.markdown("<p style='color:#64748b;font-size:14px'>Provides section-wise actionable feedback: abstract clarity, citations, redundancy, methodology, conclusions, and more.</p>", unsafe_allow_html=True)

        if st.button("▶ Generate Suggestions", key="btn_sugg", use_container_width=True, type="primary"):
            with st.status("💡 Generating Suggestions...", expanded=True) as status:
                st.write("🧠 Thinking of improvements...")
                r = requests.post(ANA_SUGGEST, timeout=120)
                if r.status_code == 200:
                    st.session_state.res_suggestions = r.json()
                    status.update(label="✅ Suggestions Generated", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Generation Failed", state="error")
                    st.error(f"Error: {r.text[:200]}")

    # ── 📋 Unified Paper Analysis (new schema) ────────────────
    st.divider()
    st.markdown("### 📋 Unified Paper Analysis")
    st.caption("Single-call evaluation: quality scores, contributions, suggestions, similar papers, and plagiarism.")

    ANA_PAPER = f"{BASE}/analyze/paper"

    col_run, col_ref = st.columns([1, 3])
    with col_ref:
        ref_text = st.text_area(
            "Reference text for plagiarism (optional)",
            height=68,
            placeholder="Paste a reference paper excerpt here (optional)…",
            key="unified_ref_text",
            label_visibility="collapsed",
        )
    with col_run:
        if st.button("▶ Run Unified Analysis", key="btn_unified", use_container_width=True, type="primary"):
            with st.status("🔬 Running Full Evaluation...", expanded=True) as status:
                st.write("📄 Processing multiple modules...")
                r = requests.post(
                    ANA_PAPER,
                    json={"reference_text": ref_text or ""},
                    timeout=180,
                )
                if r.status_code == 200:
                    st.session_state["unified_result"] = r.json()
                    status.update(label="✅ Unified Analysis Done", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {r.text[:200]}")

    unified = st.session_state.get("unified_result")
    if unified and not unified.get("error"):
        scores = unified.get("scores", {})
        score_colors = {
            "Clarity":        ("clarity",        "#818cf8"),
            "Novelty":        ("novelty",         "#f472b6"),
            "Technical Depth":("technical_depth", "#38bdf8"),
            "Methodology":    ("methodology",     "#fbbf24"),
            "Readability":    ("readability",      "#34d399"),
        }
        bars_html = "".join(
            score_bar(lbl, scores.get(key, 0), color)
            for lbl, (key, color) in score_colors.items()
            if scores.get(key) is not None
        )
        st.markdown(bars_html, unsafe_allow_html=True)

        u_c1, u_c2, u_c3 = st.columns(3)
        sections = [
            (u_c1, "✅ Contributions",  unified.get("contributions", []),  "contrib"),
            (u_c2, "⚠️ Weaknesses",     unified.get("weaknesses", []),     "weakness"),
            (u_c3, "💡 Suggestions",    unified.get("suggestions", []),    "suggestion"),
        ]
        for col, hdr, items, cls in sections:
            with col:
                st.markdown(f"<div class='result-section-header'>{hdr}</div>", unsafe_allow_html=True)
                items_html = "".join(
                    f'<div class="{cls}-item"><span class="{cls}-bullet">{'▸' if cls=='contrib' else ('✕' if cls=='weakness' else '→')}</span><span>{strip_html(str(item))}</span></div>'
                    for item in items
                ) or "<span style='color:#475569;font-size:13px'>None provided.</span>"
                st.markdown(items_html, unsafe_allow_html=True)

        # Similar papers from unified result
        sp = unified.get("similar_papers", "Not available")
        if isinstance(sp, list) and sp:
            with st.expander("📚 Similar Papers (from unified analysis)", expanded=False):
                cards_html = '<div class="papers-scroll" style="max-height:320px">'
                for item in sp:
                    cards_html += f'<div class="paper-card"><div class="paper-title">{strip_html(str(item))}</div></div>'
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)
        elif isinstance(sp, str) and sp.lower() not in ("not available", ""):
            with st.expander("📚 Similar Papers (from unified analysis)", expanded=False):
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>{strip_html(sp)}</div>", unsafe_allow_html=True)

        # Plagiarism field
        plag_val = unified.get("plagiarism", "No comparison data available")
        st.markdown("<div class='result-section-header'>🔍 Plagiarism / Similarity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='plagiarism-box'>{strip_html(str(plag_val))}</div>", unsafe_allow_html=True)

    elif unified and unified.get("error"):
        st.warning(f"⚠️ {unified['error']}")

    # ── 🌐 Academic Intelligence Section ────────────────────
    st.divider()
    st.markdown("""
    <div style="margin:20px 0">
        <h3>🌐 Multi-Source Academic Intelligence</h3>
        <p style="color:#94a3b8;font-size:14px">
            Compare your paper with real-world research using Semantic Scholar + OpenAlex.
            Powered by external academic APIs — no fake data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Full analysis run button
    col_full, col_spacer = st.columns([1, 4])
    with col_full:
        if st.button("🚀 Run Full Academic Analysis", use_container_width=True, type="primary"):
            with st.status("🌐 Querying Semantic Scholar + OpenAlex...", expanded=True) as status:
                st.write("📡 Connecting to academic APIs...")
                
                # Debug log
                st.write(f"🔍 Debug: Requesting POST {ACAD_FULL}")
                print(f"DEBUG: Requesting POST {ACAD_FULL} with filename={st.session_state.loaded_file}")
                
                r = requests.post(ACAD_FULL, json={"filename": st.session_state.loaded_file}, timeout=120)
                
                # Debug log
                st.write(f"🔍 Debug: Response Status {r.status_code}")
                print(f"DEBUG: Response Status {r.status_code}")
                
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.academic_similar = {
                        "status": data.get("status"),
                        "document_title": data.get("document_title"),
                        "papers": data.get("similar_papers", [])
                    }
                    st.session_state.academic_quality = {
                        "status": data.get("status"),
                        "document_title": data.get("document_title"),
                        "quality_index": data.get("quality_index", {})
                    }
                    st.session_state.academic_trends = {
                        "status": data.get("status"),
                        "document_title": data.get("document_title"),
                        "trends": data.get("trends", {}),
                        "related_concepts": data.get("related_concepts", []),
                        "trending_keywords": data.get("trending_keywords", [])
                    }
                    st.session_state.academic_suggestions = {
                        "status": data.get("status"),
                        "document_title": data.get("document_title"),
                        "suggestions": data.get("suggestions", []),
                        "missing_citations": data.get("missing_citations", [])
                    }
                    st.session_state.academic_novelty = {
                        "status": data.get("status"),
                        "document_title": data.get("document_title"),
                        "novelty_score": data.get("novelty_score"),
                        "interpretation": data.get("interpretation", "")
                    }
                    status.update(label="✅ Academic Intelligence Merged", state="complete", expanded=False)
                    _snapshot_mode_local("researcher")
                    st.rerun()
                else:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Full analysis failed: {r.text[:200]}")

    # ── 5 tabs for Academic Intelligence ─────────────────────
    ai_tabs = st.tabs(["📚 Similar Papers", "⭐ Quality Index", "📈 Trends", "💡 Suggestions", "🔬 Novelty Check"])

    # ── Tab 1: Similar Papers ────────────────────────────────
    with ai_tabs[0]:
        sim_data = st.session_state.academic_similar
        if sim_data and sim_data.get("papers"):
            papers = sim_data["papers"]
            st.caption(f"Found {len(papers)} related papers · Semantic Scholar + OpenAlex")

            cards_html = '<div class="papers-scroll">'
            for p in papers[:10]:
                title       = strip_html(p.get("title", "Untitled"))
                authors_raw = p.get("authors", [])
                authors     = strip_html(", ".join(str(a) for a in authors_raw[:3]))
                if len(authors_raw) > 3:
                    authors += f" +{len(authors_raw)-3}"
                year        = p.get("year") or "?"
                citations   = p.get("citation_count", 0)
                venue       = strip_html(p.get("venue", "") or "")
                source_api  = strip_html(p.get("source_api", "") or "")
                url         = p.get("url", "") or ""
                abstract    = strip_html(p.get("abstract", "") or "")

                meta_parts = []
                if authors: meta_parts.append(f"<b>{authors}</b>")
                meta_parts.append(str(year))
                if citations: meta_parts.append(f"{citations:,} citations")
                if venue:  meta_parts.append(venue[:40])
                meta_str = " &nbsp;·&nbsp; ".join(meta_parts)

                badge_html = ""
                if source_api:
                    badge_html = f'<span class="paper-badge badge-api">{source_api}</span>'
                if citations:
                    badge_html += f'<span class="paper-badge badge-cite">📌 {citations:,}</span>'
                link_html = ""
                if url:
                    link_html = f'<a class="paper-link" href="{url}" target="_blank" rel="noopener">View ↗</a>'

                cards_html += f"""
                <div class="paper-card" title="{title}">
                    <div class="paper-title">{title}</div>
                    <div class="paper-meta">{meta_str}</div>
                    {'<div class="paper-abstract">' + abstract[:240] + ('…' if len(abstract) > 240 else '') + '</div>' if abstract else ''}
                    <div class="paper-footer">
                        {badge_html}
                        {link_html}
                    </div>
                </div>"""
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Full Academic Analysis' above to discover similar papers from Semantic Scholar and OpenAlex.")

    # ── Tab 2: Quality Index ─────────────────────────────────
    with ai_tabs[1]:
        qi_data = st.session_state.academic_quality
        if qi_data and qi_data.get("quality_index"):
            q = qi_data["quality_index"]
            overall = q.get("overall", 0)
            interp = q.get("interpretation", "")

            st.markdown(f"""
            <div style="text-align:center;padding:16px 0">
                <div style="font-size:42px;font-weight:800;color:{'#34d399' if overall >= 7 else '#fbbf24' if overall >= 5 else '#f87171'}">{overall}</div>
                <div style="color:#94a3b8;font-size:13px">/ 10 — Overall Quality Score</div>
                <div style="color:#c4b5fd;font-size:14px;margin-top:8px;max-width:500px;margin-left:auto;margin-right:auto">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

            bar_colors = {
                "Citation Score": "#818cf8",
                "Novelty Score": "#f472b6",
                "Recency Score": "#34d399",
                "Source Quality": "#fbbf24"
            }
            for lbl, key in [("Citation Score","citation_score"),("Novelty Score","novelty_score"),
                             ("Recency Score","recency_score"),("Source Quality","source_quality")]:
                st.markdown(score_bar(lbl, q.get(key, 0), bar_colors[lbl]), unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, (lbl, key) in zip(
                [mc1, mc2, mc3, mc4],
                [("Citations","citation_score"),("Novelty","novelty_score"),
                 ("Recency","recency_score"),("Source","source_quality")]
            ):
                with col:
                    st.metric(lbl, f"{q.get(key, 0):.1f}")
        else:
            st.info("Click 'Run Full Academic Analysis' to compute the enhanced quality index using external data.")

    # ── Tab 3: Trends ────────────────────────────────────────
    with ai_tabs[2]:
        trend_data = st.session_state.academic_trends
        if trend_data and trend_data.get("trends"):
            domain_trends = trend_data["trends"]
            
            # Smart Topic Detection Warning
            if domain_trends.get("is_generic"):
                st.warning(f"⚠️ {domain_trends.get('warning_message', 'Topic is generic.')}")
            
            trends = domain_trends.get("overall_trends", {})
            yearly = trends.get("yearly_counts", [])
            keyword = domain_trends.get("keyword", trends.get("keyword", ""))
            trend_label = trends.get("trend", "unknown")
            total = trends.get("total", 0)

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span style="font-size:16px;font-weight:600;color:#e2e8f0">Topic: {keyword}</span>
                <span style="background:rgba(52,211,153,0.15);color:#34d399;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600">{trend_label.title()}</span>
            </div>
            <div style="color:#94a3b8;font-size:13px;margin-bottom:16px">Total publications in last 10 years: {total:,}</div>
            """, unsafe_allow_html=True)

            subdomains = domain_trends.get("subdomains", [])
            
            if yearly or subdomains:
                # Prepare data for multi-line graph
                plot_data = []
                if yearly:
                    plot_data.append(go.Scatter(
                        x=[y["year"] for y in yearly],
                        y=[y["count"] for y in yearly],
                        mode="lines+markers",
                        name="Overall" if subdomains else keyword,
                        line=dict(width=3, color="#818cf8")
                    ))
                
                colors = ["#34d399", "#f472b6", "#fbbf24", "#38bdf8", "#a78bfa"]
                for i, sub in enumerate(subdomains):
                    sub_yearly = sub.get("trends", {}).get("yearly_counts", [])
                    if sub_yearly:
                        plot_data.append(go.Scatter(
                            x=[y["year"] for y in sub_yearly],
                            y=[y["count"] for y in sub_yearly],
                            mode="lines+markers",
                            name=sub["name"],
                            line=dict(width=2, color=colors[i % len(colors)])
                        ))
                
                fig = go.Figure(data=plot_data)
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    xaxis_title="Year",
                    yaxis_title="Publications",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Subdomain breakdown
            if subdomains:
                st.markdown("##### 🔍 Subdomain Breakdown")
                tabs_names = [keyword + " (Overall)"] + [s["name"] for s in subdomains]
                sub_tabs = st.tabs(tabs_names)
                
                # Overall Tab
                with sub_tabs[0]:
                    trending = domain_trends.get("main_trending_keywords", [])
                    if trending:
                        cols = st.columns(min(len(trending), 4))
                        for i, tw in enumerate(trending[:8]):
                            with cols[i % 4]:
                                st.markdown(f"""
                                <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);border-radius:10px;padding:8px 12px;margin:4px 0;text-align:center">
                                    <div style="font-size:13px;font-weight:600;color:#fbbf24">{tw.get("keyword","")}</div>
                                    <div style="font-size:11px;color:#94a3b8">{tw.get("works_count",0):,} works</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Subdomain Tabs
                for i, sub in enumerate(subdomains):
                    with sub_tabs[i+1]:
                        s_total = sub.get("trends", {}).get("total", 0)
                        s_trend = sub.get("trends", {}).get("trend", "unknown")
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                            <span style="color:#94a3b8;font-size:13px">Total Pubs: <b>{s_total:,}</b></span>
                            <span style="background:rgba(52,211,153,0.15);color:#34d399;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{s_trend.title()}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        s_trending = sub.get("trending_keywords", [])
                        if s_trending:
                            cols = st.columns(min(len(s_trending), 4))
                            for j, tw in enumerate(s_trending[:8]):
                                with cols[j % 4]:
                                    st.markdown(f"""
                                    <div style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:8px 12px;margin:4px 0;text-align:center">
                                        <div style="font-size:13px;font-weight:600;color:#38bdf8">{tw.get("keyword","")}</div>
                                        <div style="font-size:11px;color:#94a3b8">{tw.get("works_count",0):,} works</div>
                                    </div>
                                    """, unsafe_allow_html=True)
            else:
                # Trending keywords (Original logic)
                trending = domain_trends.get("main_trending_keywords", [])
                if trending:
                    st.markdown("##### 🔥 Trending Keywords")
                    cols = st.columns(min(len(trending), 4))
                    for i, tw in enumerate(trending[:8]):
                        with cols[i % 4]:
                            st.markdown(f"""
                            <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);border-radius:10px;padding:8px 12px;margin:4px 0;text-align:center">
                                <div style="font-size:13px;font-weight:600;color:#fbbf24">{tw.get("keyword","")}</div>
                                <div style="font-size:11px;color:#94a3b8">{tw.get("works_count",0):,} works</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Full Academic Analysis' to see yearly publication trends and trending keywords.")

    # ── Tab 4: Suggestions ───────────────────────────────────
    with ai_tabs[3]:
        sugg_data = st.session_state.academic_suggestions
        if sugg_data and sugg_data.get("suggestions"):
            suggestions = sugg_data["suggestions"]
            for s in suggestions:
                priority = s.get("priority", "Medium")
                color = "#34d399" if priority == "High" else "#fbbf24" if priority == "Medium" else "#94a3b8"
                st.markdown(f"""
                <div class="ai-card" style="opacity:1;animation:none;margin:8px 0;padding:14px 18px">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                        <span style="background:rgba({"52,211,153" if priority=="High" else "251,191,36" if priority=="Medium" else "148,163,184"},0.15);color:{color};padding:2px 10px;border-radius:12px;font-size:11px;font-weight:700">{priority}</span>
                        <span style="font-weight:600;color:#e2e8f0;font-size:14px">{s.get("category","")}</span>
                    </div>
                    <div style="color:#c4b5fd;font-size:13px;font-weight:500">{s.get("suggestion","")}</div>
                    <div style="color:#94a3b8;font-size:12px;margin-top:4px">{s.get("details","")}</div>
                </div>
                """, unsafe_allow_html=True)

            # Missing citations
            missing = sugg_data.get("missing_citations", [])
            if missing:
                st.markdown("##### 📚 Missing Citations")
                for p in missing[:5]:
                    st.markdown(f"""
                    <div class="doc-card">
                        <div style="font-weight:600;color:#e2e8f0;font-size:13px">{p.get("title","")}</div>
                        <div style="color:#94a3b8;font-size:12px">{', '.join(p.get('authors',[]))} · {p.get('year','?')} · {p.get('citation_count',0)} citations</div>
                        <div style="color:#cbd5e1;font-size:12px;margin-top:4px">{p.get('reason','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Full Academic Analysis' to get actionable suggestions and missing citations.")

    # ── Tab 5: Novelty Check ─────────────────────────────────
    with ai_tabs[4]:
        nov_data = st.session_state.academic_novelty
        if nov_data and nov_data.get("novelty_score") is not None:
            score = nov_data["novelty_score"]
            interp = nov_data.get("interpretation", "")
            st.markdown(f"""
            <div style="text-align:center;padding:16px 0">
                <div style="font-size:42px;font-weight:800;color:{'#34d399' if score >= 7 else '#fbbf24' if score >= 5 else '#f87171'}">{score:.1f}</div>
                <div style="color:#94a3b8;font-size:13px">/ 10 — Novelty Score</div>
                <div style="color:#c4b5fd;font-size:14px;margin-top:8px;max-width:500px;margin-left:auto;margin-right:auto">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

            # Novelty explanation
            st.markdown("""
            <div class="ai-card" style="opacity:1;animation:none">
                <div class="card-header">ℹ️ How is this calculated?</div>
                <div class="card-content">
                    We extract your paper's title + abstract, then compare word overlap (Jaccard similarity)
                    with the top 8 most similar papers from Semantic Scholar. Lower overlap = higher novelty.
                    This is a fast heuristic — for a thorough novelty review, consult your domain expert.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Full Academic Analysis' to check your paper's novelty against external literature.")

    # ── RAG Chat in researcher mode ──────────────────────────
    st.divider()
    st.markdown("### 🤖 RAG Query (Researcher)")
    st.caption("Ask targeted questions about your paper while in researcher mode.")

    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
        if msg.get("role", "assistant") == "assistant" and st.session_state.show_ml:
            pid = str(idx)
            if pid in st.session_state.pipeline_data:
                with st.expander("🔬 ML Pipeline Details", expanded=False):
                    render_pipeline(st.session_state.pipeline_data[pid])

    if prompt := st.chat_input("Ask a targeted research question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder  = st.empty()
            full_resp    = ""
            pipeline_res = None
            try:
                STEPS = [
                    ("🔍","Analyzing query…"),("✨","Enhancing…"),
                    ("📄","Retrieving…"),("⭐","Re-ranking…"),("🤖","Answering…"),
                ]
                loader = st.empty()
                if st.session_state.show_ml:
                    for i in range(len(STEPS)):
                        loading_pipeline_animation(loader, STEPS, i)

                with st.spinner("🧠 Processing…"):
                    r = requests.post(API_URL, json={
                        "username":   st.session_state.username,
                        "question":   prompt,
                        "filename":   st.session_state.loaded_file,
                        "session_id": st.session_state.session_id,
                        "mode":       "researcher",
                    }, timeout=300)

                loader.empty()

                if r.status_code == 200:
                    data         = r.json()
                    answer       = data.get("answer", "")
                    pipeline_res = data
                    full_resp = answer
                    placeholder.markdown(full_resp)
                    cache_txt = "Hit" if data.get("cache_hit") else "Miss"
                    st.caption(f"Response time: {data.get('execution_time_ms', 0)} ms | Cache: {cache_txt}")
                else:
                    st.error(f"❌ Backend error {r.status_code}")
            except Exception as e:
                st.error(f"⚠️ Network error: {e}")

        if full_resp:
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
            st.session_state.mode_store["researcher"]["last_input"] = prompt
            if pipeline_res:
                idx = str(len(st.session_state.messages) - 1)
                st.session_state.pipeline_data[idx] = pipeline_res

            sid = st.session_state.session_id
            if sid not in st.session_state.past_history:
                st.session_state.past_history[sid] = {
                    "title": (prompt[:30] + "…") if len(prompt) > 30 else prompt,
                    "messages": st.session_state.messages.copy()
                }
            else:
                st.session_state.past_history[sid]["messages"] = st.session_state.messages.copy()
            _snapshot_mode_local("researcher")
            _sync_mode_to_backend("researcher")


# ─────────────────────────────────────────────────────────────
# 🎓  ACADEMIC EVALUATION ENGINE MODE
# ─────────────────────────────────────────────────────────────
elif st.session_state.mode == "eval_engine":
    st.markdown("### 🎓 Academic Evaluation Engine")
    st.markdown("Evaluate your paper on a per-question basis with detailed analysis and improvement suggestions.")

    if not st.session_state.loaded_file:
        st.info("👆 Load a document from the sidebar to run the evaluation engine.")
        st.stop()

    with st.container(border=True):
        st.markdown("##### 📝 Questions to Evaluate")
        q_list = st.session_state.mode_store["eval_engine"]["questions"]
        q_text = st.text_area("Enter questions (one per line)", value="\n".join(q_list), height=150)
        new_q_list = [q.strip() for q in q_text.split("\n") if q.strip()]
        st.session_state.mode_store["eval_engine"]["questions"] = new_q_list

        if st.button("🚀 Run Academic Evaluation", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing paper on a per-question basis..."):
                try:
                    r = requests.post(
                        f"{BASE}/academic/evaluate-questions",
                        json={
                            "questions": new_q_list,
                            "filename": st.session_state.loaded_file
                        },
                        timeout=300
                    )
                    if r.status_code == 200:
                        st.session_state.mode_store["eval_engine"]["eval_result"] = r.json().get("results")
                        st.toast("Evaluation complete!", icon="✅")
                    else:
                        st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(f"Network error: {e}")

    st.divider()

    res = st.session_state.mode_store["eval_engine"].get("eval_result")
    if res:
        summary = res.get("summary", {})
        st.markdown(f"#### 📊 Overall Score: {summary.get('overall_score', 0)}/10")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("##### ✅ Key Strengths")
            for s in summary.get("key_strengths", []):
                st.markdown(f"<div class='contrib-item'><span class='contrib-bullet'>•</span>{s}</div>", unsafe_allow_html=True)
        with sc2:
            st.markdown("##### ⚠️ Key Weaknesses")
            for w in summary.get("key_weaknesses", []):
                st.markdown(f"<div class='weakness-item'><span class='weakness-bullet'>•</span>{w}</div>", unsafe_allow_html=True)

        st.markdown("#### 📋 Detailed Evaluations")
        for i, ev in enumerate(res.get("evaluations", [])):
            with st.expander(f"Q{i+1}: {ev.get('question')} — Score: {ev.get('score')}/10"):
                st.markdown(f"**Answer:**\n{ev.get('answer')}")
                st.markdown(f"**Justification:**\n{ev.get('justification')}")
                
                st.markdown("---")
                st.markdown("**🔍 Analysis Details**")
                details = ev.get("details", {})
                st.markdown(details.get("analysis", ""))
                
                col_g, col_i = st.columns(2)
                with col_g:
                    st.markdown("**🚩 Gaps**")
                    for g in details.get("gaps", []):
                        st.markdown(f"- {g}")
                with col_i:
                    st.markdown("**💡 Suggestions**")
                    for imp in details.get("improvements", []):
                        st.markdown(f"- {imp}")
                
                st.markdown("**🛠 Actionable Improvements**")
                for imp in details.get("improvements", []):
                    st.markdown(f"<div class='suggestion-item'><span class='suggestion-bullet'>→</span>{imp}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#64748b">
            <div style="font-size:44px;margin-bottom:14px">🎓</div>
            <h3 style="color:#94a3b8;-webkit-text-fill-color:#94a3b8;font-weight:500">Academic Evaluation Engine</h3>
            <p style="font-size:15px">Enter your research questions above and click "Run Academic Evaluation".</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 📊  ML EVALUATION MODE
# ─────────────────────────────────────────────────────────────
elif st.session_state.mode == "eval":
    st.markdown("### 📊 ML Pipeline Evaluation")
    st.markdown("Benchmark the system against standard questions to measure accuracy, precision, and recall.")

    if not st.session_state.loaded_file:
        st.info("👆 Load a document from the sidebar to run evaluation.")
        st.stop()

    eval_mode = st.radio(
        "Evaluation Mode",
        options=["quick", "full"],
        horizontal=True,
        format_func=lambda m: "Quick (3 queries)" if m == "quick" else "Full (10 queries)"
    )

    # Run evaluation button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
            r = requests.post(
                EVAL_URL,
                json={
                    "username": st.session_state.username,
                    "filename": st.session_state.loaded_file,
                    "mode": eval_mode
                },
                timeout=120
            )
            if r.status_code == 200:
                st.session_state.eval_job_id = r.json().get("job_id")
                st.session_state.eval_result = None
                st.rerun()
            else:
                st.error(f"Error: {r.text[:200]}")
    
    with col2:
        st.caption("Quick mode uses 3 representative queries (objective/challenges/solutions). Full mode runs all 10.")

    st.divider()

    if st.session_state.get("eval_job_id"):
        job_id = st.session_state.eval_job_id
        
        prog_container = st.empty()
        
        with st.spinner("🧪 Evaluation running in background..."):
            while True:
                try:
                    r = requests.get(f"{BASE}/status/{job_id}", timeout=120)
                    if r.status_code == 200:
                        status_data = r.json()
                        status = status_data.get("status")
                        
                        if status == "completed":
                            st.session_state.eval_result = status_data.get("result")
                            st.session_state.eval_job_id = None
                            _snapshot_mode_local("eval")
                            _sync_mode_to_backend("eval")
                            st.rerun()
                            break
                        elif status == "error":
                            st.error(f"Evaluation failed: {status_data.get('error')}")
                            st.session_state.eval_job_id = None
                            break
                        else:
                            step = status_data.get("step", "Processing...")
                            curr = status_data.get("progress", 0)
                            tot = status_data.get("total", 1)
                            if tot == 0: tot = 1
                            pct = curr / tot
                            
                            prog_container.markdown(f'''
                            <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:10px; margin: 20px 0;">
                                <h4 style="margin-top:0">⏳ Evaluation in progress...</h4>
                                <p style="color:#a5b4fc;">{step}</p>
                                <div style="background:rgba(255,255,255,0.1); height:10px; border-radius:5px;">
                                    <div style="background:#818cf8; height:10px; border-radius:5px; width:{pct*100}%; transition: width 0.5s;"></div>
                                </div>
                                <p style="text-align:right; font-size:12px; margin-top:5px; margin-bottom:0;">{curr}/{tot}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            time.sleep(2)
                    else:
                        st.error("Lost connection to status endpoint.")
                        break
                except Exception as e:
                    st.error(f"Polling error: {e}")
                    time.sleep(2)

    eval_data = st.session_state.eval_result

    if not eval_data:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#64748b">
            <div style="font-size:44px;margin-bottom:14px">📊</div>
            <h3 style="color:#94a3b8;-webkit-text-fill-color:#94a3b8;font-weight:500">System Evaluation</h3>
            <p style="font-size:15px">Click "Run Full Evaluation" to benchmark the RAG pipeline.</p>
        </div>""", unsafe_allow_html=True)
    else:
        ds_info = eval_data.get("dataset_info", {})
        metrics = eval_data.get("metrics", {})
        results = eval_data.get("results", [])
        m_info  = eval_data.get("model_info", {})

        # ── 1. Key Metrics Cards ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}", help="Percentage of correct answers (similarity ≥ threshold)")
        c2.metric("Precision", f"{metrics.get('precision', 0):.1%}", help="True Positives / (True Positives + False Positives)")
        c3.metric("Recall", f"{metrics.get('recall', 0):.1%}", help="True Positives / (True Positives + False Negatives)")
        c4.metric("F1 Score", f"{metrics.get('f1', 0):.1%}", help="Harmonic mean of precision and recall")
        st.caption(
            f"Avg response time: {metrics.get('total_time_s', 0) / max(metrics.get('total', 1), 1):.2f}s | "
            f"Cache hit rate: {metrics.get('cache_hit_rate', 0) * 100:.1f}% | "
            f"Total evaluation time: {metrics.get('total_time_s', 0):.2f}s"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_dist, col_acc, col_info = st.columns([3, 2, 2])

        # ── 2. Visualizations ──
        with col_dist:
            st.markdown("##### 📈 Similarity Distribution")
            scores = eval_data.get("scores", [])
            if scores:
                fig = px.histogram(
                    x=scores,
                    nbins=10,
                    range_x=[0, 1],
                    color_discrete_sequence=['#818cf8'],
                    labels={'x': 'Cosine Similarity', 'count': 'Frequency'},
                    title="Score Distribution"
                )
                fig.add_vline(x=metrics.get('threshold', 0.6), line_dash="dash", line_color="red", annotation_text="Threshold")
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col_acc:
            st.markdown("##### 🎯 Accuracy Chart")
            correct = metrics.get('correct', 0)
            total = metrics.get('total', 1)
            incorrect = total - correct
            
            fig_acc = px.pie(
                names=['Correct', 'Incorrect'],
                values=[correct, incorrect],
                color_discrete_sequence=['#10b981', '#ef4444'],
                hole=0.4,
                title="Model Accuracy"
            )
            fig_acc.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_acc, use_container_width=True)

        # ── 3. Dataset & Model Info ──
        with col_info:
            st.markdown("##### 📚 Dataset")
            st.markdown(f"- **Total Chunks:** {ds_info.get('total_chunks', 0)}")
            st.markdown(f"- **Train/Test Split:** {ds_info.get('split_ratio', '')}")
            st.markdown(f"- **Eval Questions:** {ds_info.get('eval_questions', 0)}")
            domains = ds_info.get('domains', [])
            if domains:
                # Show first 3 domains to avoid taking up too much vertical space, or show all in small text
                st.markdown(f"- **Topics:** {', '.join(domains[:3])}...")
            
            st.markdown("##### ⚙️ Model Architecture")
            st.markdown(f"- **LLM:** `{m_info.get('llm', '')}`")
            st.markdown(f"- **Embedder:** `{m_info.get('embedding_model', '')}`")
            st.markdown(f"- **Re-ranker:** `{m_info.get('reranking_model', '')}`")
            st.caption(m_info.get('fine_tuning', ''))

        # ── 4. Detailed Results Table ──
        st.markdown("##### 📋 Detailed Evaluation Results")
        
        for res in results:
            sim   = res.get("similarity", 0)
            is_ok = res.get("correct", False)
            icon  = "✅" if is_ok else "❌"
            color = "#10b981" if is_ok else "#ef4444"
            
            with st.expander(f"{icon} {res.get('domain', 'Question')} (Score: {sim:.2f})"):
                st.markdown(f"**Query:** {res.get('query', '')}")
                st.markdown(f"**Answer:**\n> {res.get('model_answer', '')}")
                st.markdown(f"**Response Time:** {res.get('response_time_s', res.get('time_taken', 0))} s")
                st.markdown(f"**Cache:** {res.get('cache_status', 'Hit' if res.get('cached') else 'Miss')}")
                st.markdown(f"**Similarity Score:** {res.get('similarity_score', sim):.4f}")