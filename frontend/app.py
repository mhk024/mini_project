import streamlit as st
import requests
import time
import uuid

# =========================
# 🔗 API URLs
# =========================
API_URL      = "http://127.0.0.1:8000/ask"
UPLOAD_URL   = "http://127.0.0.1:8000/upload"
LOGIN_URL    = "http://127.0.0.1:8000/login"
SIGNUP_URL   = "http://127.0.0.1:8000/signup"
HISTORY_URL  = "http://127.0.0.1:8000/history"
FILES_URL    = "http://127.0.0.1:8000/files"
SET_FILE_URL = "http://127.0.0.1:8000/set_file"

st.set_page_config(page_title="RAG AI Dashboard", page_icon="🧠", layout="wide")

# =========================
# 🎨 PREMIUM CSS
# =========================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated gradient title */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #667eea 75%, #764ba2 100%);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 6s ease infinite;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Pipeline step cards */
.pipeline-card {
    background: linear-gradient(135deg, rgba(30, 30, 60, 0.95), rgba(20, 20, 50, 0.9));
    border: 1px solid rgba(120, 100, 255, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideUp 0.5s ease-out forwards;
    opacity: 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.pipeline-card:hover {
    border-color: rgba(120, 100, 255, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(100, 80, 255, 0.15);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Card headers */
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-size: 15px;
    font-weight: 600;
    color: #c4b5fd;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.card-header .icon {
    font-size: 22px;
}

/* Card content */
.card-content {
    color: #e2e8f0;
    font-size: 15px;
    line-height: 1.7;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    border-left: 3px solid;
}

/* Step-specific accent colors */
.step-original .card-content { border-left-color: #818cf8; }
.step-enhanced .card-content { border-left-color: #a78bfa; }
.step-retrieved .card-content { border-left-color: #38bdf8; }
.step-reranked .card-content { border-left-color: #fbbf24; }
.step-answer .card-content { border-left-color: #34d399; }

/* Step number badges */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin-right: 6px;
}
.step-badge-1 { background: linear-gradient(135deg, #818cf8, #6366f1); }
.step-badge-2 { background: linear-gradient(135deg, #a78bfa, #8b5cf6); }
.step-badge-3 { background: linear-gradient(135deg, #38bdf8, #0ea5e9); }
.step-badge-4 { background: linear-gradient(135deg, #fbbf24, #f59e0b); }
.step-badge-5 { background: linear-gradient(135deg, #34d399, #10b981); }

/* Score badges */
.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-left: 8px;
}
.score-high { background: rgba(52, 211, 153, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
.score-mid { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.score-low { background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

/* Document card */
.doc-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: all 0.2s ease;
}
.doc-card:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.15);
}
.doc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 13px;
    color: #94a3b8;
}
.doc-source {
    color: #818cf8;
    font-weight: 500;
}
.doc-preview {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
}

/* Pipeline connector line */
.pipeline-connector {
    text-align: center;
    color: rgba(120, 100, 255, 0.4);
    font-size: 24px;
    margin: 0;
    line-height: 1;
}

/* Toggle styling */
.ml-toggle-container {
    background: linear-gradient(135deg, rgba(100, 80, 255, 0.1), rgba(160, 100, 255, 0.1));
    border: 1px solid rgba(120, 100, 255, 0.2);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* Chat message animations */
div[data-testid="stChatMessage"] {
    animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Button interactions */
div[data-testid="stButton"] > button {
    transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    border-radius: 10px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: scale(1.03);
    filter: brightness(1.15);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.97);
}

/* Expander styling */
div[data-testid="stExpander"] {
    border: 1px solid rgba(120, 100, 255, 0.15) !important;
    border-radius: 12px !important;
    background: rgba(30, 30, 60, 0.5) !important;
}

/* Loading pulse animation */
.loading-pulse {
    animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Progress step indicator */
.progress-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    margin: 4px 0;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.3s ease;
}
.progress-step.active {
    background: rgba(120, 100, 255, 0.15);
    color: #c4b5fd;
}
.progress-step.done {
    color: #34d399;
}
.progress-step.waiting {
    color: #64748b;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# 🧠 SESSION STATE
# =========================
defaults = {
    "logged_in": False,
    "username": "",
    "messages": [],
    "past_history": {},
    "session_id": uuid.uuid4().hex,
    "loaded_file": "",
    "show_ml_details": False,
    "pipeline_data": {},   # Stores pipeline data keyed by message index
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Restore login from URL
query_params = st.query_params
if "user" in query_params and not st.session_state.logged_in:
    st.session_state.logged_in = True
    st.session_state.username = query_params["user"]


# =========================
# 🔧 HELPER: Score Badge HTML
# =========================
def score_badge_html(score):
    """Return HTML for a color-coded score badge."""
    if score >= 0.7:
        cls = "score-high"
    elif score >= 0.4:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score:.3f}</span>'


def render_document_card(doc, index):
    """Render a single document card with expandable content."""
    score = doc.get("score", 0)
    source = doc.get("source", "Unknown")
    content = doc.get("content", "")
    preview = content[:150] + "..." if len(content) > 150 else content

    score_html = score_badge_html(score)
    
    st.markdown(f"""
    <div class="doc-card">
        <div class="doc-header">
            <span class="doc-source">📎 {source}</span>
            {score_html}
        </div>
        <div class="doc-preview">{preview}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if len(content) > 150:
        with st.expander(f"📖 View full content — Document {index + 1}", expanded=False):
            st.markdown(content)


def render_pipeline(data):
    """Render the full ML pipeline visualization."""
    
    # Step 1: Original Query
    st.markdown(f"""
    <div class="pipeline-card step-original" style="animation-delay: 0s;">
        <div class="card-header">
            <span class="step-badge step-badge-1">1</span>
            <span class="icon">🔍</span> Original Query
        </div>
        <div class="card-content">{data.get('original_query', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="pipeline-connector">↓</div>', unsafe_allow_html=True)
    
    # Step 2: Enhanced Query
    enhanced = data.get('enhanced_query', data.get('original_query', 'N/A'))
    st.markdown(f"""
    <div class="pipeline-card step-enhanced" style="animation-delay: 0.1s;">
        <div class="card-header">
            <span class="step-badge step-badge-2">2</span>
            <span class="icon">✨</span> Enhanced Query <small style="color: #94a3b8; font-weight: normal; text-transform: none;">(ML-Rewritten for better retrieval)</small>
        </div>
        <div class="card-content">{enhanced}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="pipeline-connector">↓</div>', unsafe_allow_html=True)
    
    # Step 3: Retrieved Documents
    retrieved_docs = data.get('retrieved_documents', [])
    st.markdown(f"""
    <div class="pipeline-card step-retrieved" style="animation-delay: 0.2s;">
        <div class="card-header">
            <span class="step-badge step-badge-3">3</span>
            <span class="icon">📄</span> Retrieved Documents <small style="color: #94a3b8; font-weight: normal; text-transform: none;">({len(retrieved_docs)} found via vector similarity)</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, doc in enumerate(retrieved_docs[:5]):
        render_document_card(doc, i)
    
    st.markdown('<div class="pipeline-connector">↓</div>', unsafe_allow_html=True)
    
    # Step 4: Re-ranked Documents
    reranked_docs = data.get('documents', [])
    st.markdown(f"""
    <div class="pipeline-card step-reranked" style="animation-delay: 0.3s;">
        <div class="card-header">
            <span class="step-badge step-badge-4">4</span>
            <span class="icon">⭐</span> Re-ranked Documents <small style="color: #94a3b8; font-weight: normal; text-transform: none;">(Cross-Encoder scored &amp; sorted)</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, doc in enumerate(reranked_docs[:5]):
        render_document_card(doc, i)
    
    st.markdown('<div class="pipeline-connector">↓</div>', unsafe_allow_html=True)
    
    # Step 5: Final Answer
    st.markdown(f"""
    <div class="pipeline-card step-answer" style="animation-delay: 0.4s;">
        <div class="card-header">
            <span class="step-badge step-badge-5">5</span>
            <span class="icon">🤖</span> Generated Answer <small style="color: #94a3b8; font-weight: normal; text-transform: none;">(LLM output using re-ranked context)</small>
        </div>
        <div class="card-content">{data.get('answer', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# 🔐 AUTH FUNCTIONS
# =========================
def login_user(username, password):
    with st.spinner("🔐 Authenticating..."):
        res = requests.post(LOGIN_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.query_params["user"] = username

        try:
            hist = requests.get(f"{HISTORY_URL}/{username}")
            if hist.status_code == 200:
                st.session_state.past_history = hist.json().get("history", {})
        except:
            pass

        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex
        st.toast(f"Welcome back, {username}!", icon="👋")
        time.sleep(0.5)
        st.rerun()
    else:
        st.error("❌ Invalid credentials")


def signup_user(username, password):
    with st.spinner("📝 Creating account..."):
        res = requests.post(SIGNUP_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.toast("Signup successful! You can now login.", icon="🎉")
        st.balloons()
    else:
        st.error(res.json().get("detail", "Signup failed"))


# =========================
# 🔑 LOGIN PAGE
# =========================
if not st.session_state.logged_in:
    st.markdown("")
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>🧠 RAG AI Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 30px;'>Intelligent Document Analysis with ML Pipeline Transparency</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Signup"])
        
        with tab1:
            with st.container(border=True):
                u = st.text_input("Username", key="login_u", placeholder="Enter your username")
                p = st.text_input("Password", type="password", key="login_p", placeholder="Enter your password")
                st.markdown("")
                if st.button("🔓 Login", use_container_width=True, type="primary"):
                    if u and p:
                        login_user(u, p)
                    else:
                        st.warning("Please enter username and password")

        with tab2:
            with st.container(border=True):
                u = st.text_input("New Username", key="signup_u", placeholder="Choose a username")
                p = st.text_input("New Password", type="password", key="signup_p", placeholder="Choose a password")
                st.markdown("")
                if st.button("🚀 Create Account", use_container_width=True, type="primary"):
                    if u and p:
                        signup_user(u, p)
                    else:
                        st.warning("Please enter username and password")

    st.stop()


# =========================
# 🧭 SIDEBAR
# =========================
with st.sidebar:
    st.markdown(f"### 👤 Welcome, **{st.session_state.username}**")

    # ML Details Toggle
    st.markdown("")
    st.markdown('<div class="ml-toggle-container">', unsafe_allow_html=True)
    st.session_state.show_ml_details = st.toggle(
        "🔬 Show ML Pipeline Details",
        value=st.session_state.show_ml_details,
        help="Toggle to see the full RAG pipeline: query enhancement, retrieval, re-ranking, and generation"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.show_ml_details:
        st.caption("📊 Pipeline visualization is **ON** — you'll see all ML steps")
    else:
        st.caption("💬 Normal chat mode — only final answers shown")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pipeline_data = {}
            st.toast("Chat cleared!", icon="🧹")
            time.sleep(0.5)
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.query_params.clear()
            st.toast("Logged out", icon="👋")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # File selection
    st.subheader("📂 Documents")
    try:
        with st.spinner("Fetching files..."):
            files = requests.get(FILES_URL, timeout=5).json().get("files", [])
    except:
        files = []

    selected = st.selectbox("Select a file to query", [""] + files)

    if selected:
        if st.button("📥 Load File", use_container_width=True, type="primary"):
            st.session_state.loaded_file = ""
            with st.spinner(f"Loading `{selected}`..."):
                res = requests.post(SET_FILE_URL, json={"filename": selected}, timeout=600)
            if res.status_code == 200:
                st.session_state.loaded_file = selected
                st.toast(f"Loaded {selected}", icon="✅")
                time.sleep(0.5)
                st.rerun()
            else:
                err_msg = res.text
                try: err_msg = res.json().get('detail', res.text)
                except: pass
                st.error(f"Failed: {err_msg}")

    if st.session_state.get("loaded_file"):
        st.success(f"📄 **Active:** {st.session_state.loaded_file}")
    else:
        st.info("No document active.")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pipeline_data = {}
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

    # Chat History
    st.subheader("📝 Recent Chats")
    loaded_doc = st.session_state.get("loaded_file")

    if loaded_doc:
        past_hist = st.session_state.get("past_history")
        if past_hist and isinstance(past_hist, dict):
            has_visible = False
            for sid, sdata in reversed(list(past_hist.items())):
                msgs = sdata.get("messages", [])
                sess_filename = msgs[-1].get("filename") if msgs else None
                if sess_filename == loaded_doc:
                    has_visible = True
                    title = sdata.get("title", "Chat")
                    is_active = (sid == st.session_state.session_id)
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(f"💬 {title}", key=f"chat_{sid}", use_container_width=True, type=btn_type):
                        st.session_state.session_id = sid
                        st.session_state.messages = msgs
                        st.session_state.pipeline_data = {}
                        st.rerun()
            if not has_visible:
                st.info(f"No history for '{loaded_doc}'.")
        else:
            st.info(f"No history yet.")
    else:
        st.info("Select a document first.")

    st.divider()

    # Upload
    st.subheader("⬆ Upload New File")
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")

    if file:
        if st.button("☁️ Upload to Knowledge Base", use_container_width=True, type="primary"):
            st.session_state.loaded_file = ""
            with st.status("Uploading file...", expanded=True) as status:
                st.write("Sending file to server...")
                files_payload = {"file": (file.name, file.getvalue(), file.type)}
                res = requests.post(UPLOAD_URL, files=files_payload, timeout=300)
                if res.status_code == 200:
                    status.update(label="Upload complete!", state="complete", expanded=False)
                    st.session_state.loaded_file = file.name
                    st.toast(f"'{file.name}' uploaded!", icon="🚀")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    err_msg = res.text
                    try: err_msg = res.json().get('detail', res.text)
                    except: pass
                    status.update(label="Upload failed", state="error", expanded=True)
                    st.error(f"Error: {err_msg}")


# =========================
# 💬 MAIN CHAT AREA
# =========================
st.title("🧠 RAG AI Dashboard")

if st.session_state.get("loaded_file"):
    st.caption(f"📄 Currently querying: **{st.session_state.loaded_file}**")
else:
    st.warning("⚠️ Please load a document from the sidebar to get started.")

# Show mode indicator
if st.session_state.show_ml_details:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(100, 80, 255, 0.1), rgba(56, 189, 248, 0.1));
                border: 1px solid rgba(120, 100, 255, 0.2); border-radius: 12px;
                padding: 12px 20px; margin-bottom: 16px;
                display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px;">🔬</span>
        <span style="color: #c4b5fd; font-weight: 500;">ML Pipeline Mode</span>
        <span style="color: #94a3b8; font-size: 13px;">— Full pipeline visualization enabled. Each response shows all ML processing steps.</span>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #64748b;">
        <div style="font-size: 48px; margin-bottom: 16px;">🧠</div>
        <h3 style="color: #94a3b8; -webkit-text-fill-color: #94a3b8; font-weight: 500;">Ready to analyze your documents</h3>
        <p style="font-size: 15px;">Upload a document and ask me anything. Toggle <strong>ML Pipeline Details</strong> to see how the AI processes your query.</p>
    </div>
    """, unsafe_allow_html=True)

# Display message history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
    # Show pipeline details after assistant messages (if toggle is ON and data exists)
    if msg["role"] == "assistant" and st.session_state.show_ml_details:
        pipeline_key = str(idx)
        if pipeline_key in st.session_state.pipeline_data:
            with st.expander("🔬 View ML Pipeline Details", expanded=False):
                render_pipeline(st.session_state.pipeline_data[pipeline_key])


# =========================
# 🧠 USER INPUT
# =========================
if prompt := st.chat_input("Ask something about your document..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        pipeline_result = None

        try:
            # Show loading with pipeline steps
            if st.session_state.show_ml_details:
                loading_container = st.empty()
                loading_steps = [
                    ("🔍", "Analyzing original query..."),
                    ("✨", "Enhancing query with ML..."),
                    ("📄", "Retrieving documents from vector store..."),
                    ("⭐", "Re-ranking with Cross-Encoder..."),
                    ("🤖", "Generating answer with LLM..."),
                ]
                
                for i, (icon, step_text) in enumerate(loading_steps):
                    steps_html = ""
                    for j, (s_icon, s_text) in enumerate(loading_steps):
                        if j < i:
                            steps_html += f'<div class="progress-step done">✅ {s_text}</div>'
                        elif j == i:
                            steps_html += f'<div class="progress-step active loading-pulse">{s_icon} {s_text}</div>'
                        else:
                            steps_html += f'<div class="progress-step waiting">⏳ {s_text}</div>'
                    
                    loading_container.markdown(f"""
                    <div class="pipeline-card" style="opacity: 1;">
                        <div class="card-header">
                            <span class="icon">⚙️</span> Processing RAG Pipeline...
                        </div>
                        {steps_html}
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.4)

            with st.spinner("🧠 Processing your query through the RAG pipeline..."):
                res = requests.post(
                    API_URL,
                    json={
                        "username": st.session_state.username,
                        "question": prompt,
                        "filename": st.session_state.get("loaded_file"),
                        "session_id": st.session_state.session_id
                    },
                    timeout=120
                )

            # Clear loading animation
            if st.session_state.show_ml_details:
                loading_container.empty()

            if res.status_code == 200:
                response_data = res.json()
                answer = response_data.get("answer", "")
                pipeline_result = response_data

                # Fake streaming effect for the answer
                for chunk in answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)

                message_placeholder.markdown(full_response)
            else:
                st.error("❌ Error from backend")

        except Exception as e:
            st.error(f"⚠️ Network error: {str(e)}")

    # Save response & pipeline data
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Store pipeline data keyed by message index
        if pipeline_result:
            msg_idx = str(len(st.session_state.messages) - 1)
            st.session_state.pipeline_data[msg_idx] = pipeline_result

        # Update local past_history
        if st.session_state.session_id not in st.session_state.past_history:
            st.session_state.past_history[st.session_state.session_id] = {
                "title": prompt[:30] + "..." if len(prompt) > 30 else prompt,
                "messages": st.session_state.messages.copy()
            }
        else:
            st.session_state.past_history[st.session_state.session_id]["messages"] = st.session_state.messages.copy()

        # Show pipeline immediately for new messages
        if st.session_state.show_ml_details and pipeline_result:
            with st.expander("🔬 View ML Pipeline Details", expanded=True):
                render_pipeline(pipeline_result)