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

st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="wide")

# =========================
# 🎨 CUSTOM STYLES & ANIMATIONS
# =========================
CUSTOM_CSS = """
<style>
/* Smooth fade-in for chat messages */
div[data-testid="stChatMessage"] {
    animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Button interactions (scale and bright/dim) */
div[data-testid="stButton"] > button {
    transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: scale(1.03);
    filter: brightness(1.15);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.95);
    filter: brightness(0.85);
    box-shadow: none !important;
}

/* Fun gradient animation for page titles */
h1 {
    background: linear-gradient(45deg, #FF4B2B, #FF416C, #4A00E0, #8E2DE2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 8s ease infinite;
    background-size: 300% 300%;
}
@keyframes gradientText {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Float animation for alert boxes (success/info) */
div[data-testid="stElementContainer"] > div[data-testid="stAlert"] {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="stElementContainer"] > div[data-testid="stAlert"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# 🧠 SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "past_history" not in st.session_state:
    st.session_state.past_history = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


# =========================
# 🔐 AUTH FUNCTIONS
# =========================
def login_user(username, password):
    with st.spinner("Authenticating..."):
        res = requests.post(LOGIN_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username = username

        # Load history
        try:
            with st.spinner("Loading chat history..."):
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
        st.error("Invalid credentials")


def signup_user(username, password):
    with st.spinner("Creating account..."):
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
    st.markdown("<h1 style='text-align: center;'>🤖 RAG Chat App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Your intelligent document assistant</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Signup"])

        with tab1:
            with st.container(border=True):
                u = st.text_input("Username", key="login_u")
                p = st.text_input("Password", type="password", key="login_p")
                if st.button("Login", use_container_width=True, type="primary"):
                    if u and p:
                        login_user(u, p)
                    else:
                        st.warning("Please enter username and password")

        with tab2:
            with st.container(border=True):
                u = st.text_input("New Username", key="signup_u")
                p = st.text_input("New Password", type="password", key="signup_p")
                if st.button("Signup", use_container_width=True, type="primary"):
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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.toast("Chat history cleared!", icon="🧹")
            st.snow()
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.toast("Logged out", icon="👋")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # File selection
    st.subheader("📂 Documents")
    try:
        with st.spinner("Fetching available files..."):
            files = requests.get(FILES_URL, timeout=5).json().get("files", [])
    except:
        files = []

    selected = st.selectbox("Select a file to query", [""] + files)

    if "loaded_file" not in st.session_state:
        st.session_state.loaded_file = ""

    if selected:
        if st.button("Load File 📥", use_container_width=True, type="primary"):
            st.session_state.loaded_file = "" # Deactivate previous
            with st.spinner(f"Loading and processing `{selected}`..."):
                res = requests.post(SET_FILE_URL, json={"filename": selected}, timeout=600)
            if res.status_code == 200:
                st.session_state.loaded_file = selected
                st.toast(f"Successfully loaded {selected}", icon="✅")
            else:
                err_msg = res.text
                try: err_msg = res.json().get('detail', res.text)
                except: pass
                st.error(f"Failed to load: {err_msg}")

    if st.session_state.get("loaded_file"):
        st.success(f"📄 **Active Document:** {st.session_state.loaded_file}")
    else:
        st.info("No document currently active.")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

    # Chat History
    st.subheader("📝 Recent Chats")
    loaded_doc = st.session_state.get("loaded_file")
    
    if loaded_doc:
        past_hist = st.session_state.get("past_history")
        if past_hist:
            if isinstance(past_hist, dict):
                has_visible_chats = False
                for sid, sdata in reversed(list(past_hist.items())):
                    # Get filename of this session
                    msgs = sdata.get("messages", [])
                    sess_filename = msgs[-1].get("filename") if msgs else None
                    
                    if sess_filename == loaded_doc:
                        has_visible_chats = True
                        title = sdata.get("title", "Chat")
                        is_active = (sid == st.session_state.session_id)
                        button_type = "primary" if is_active else "secondary"
                        if st.sidebar.button(f"💬 {title}", key=f"chat_{sid}", use_container_width=True, type=button_type):
                            st.session_state.session_id = sid
                            st.session_state.messages = msgs
                            st.rerun()
                
                if not has_visible_chats:
                    st.info(f"No history yet for '{loaded_doc}'.")
            else:
                st.sidebar.info("Legacy chat found. Send a message to upgrade your history structure.")
        else:
            st.info(f"No history yet for '{loaded_doc}'.")
    else:
        st.info("Select a document to retrieve its past conversations.")

    st.divider()

    # Upload
    st.subheader("⬆ Upload New File")
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")

    if file:
        if st.button("Upload to Knowledge Base ☁️", use_container_width=True, type="primary"):
            st.session_state.loaded_file = "" # Deactivate previous
            with st.status("Uploading file...", expanded=True) as status:
                st.write("Sending file to server...")
                # Important: use getvalue() to send the file bytes correctly
                files_payload = {"file": (file.name, file.getvalue(), file.type)}
                res = requests.post(UPLOAD_URL, files=files_payload, timeout=300)
                if res.status_code == 200:
                    status.update(label="Upload complete!", state="complete", expanded=False)
                    st.session_state.loaded_file = file.name
                    st.toast(f"'{file.name}' uploaded successfully!", icon="🚀")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    err_msg = res.text
                    try:
                        err_msg = res.json().get('detail', res.text)
                    except:
                        pass
                    status.update(label=f"Upload failed", state="error", expanded=True)
                    st.error(f"Error Details: {err_msg}")

# =========================
# 💬 CHAT UI
# =========================
st.title("💬 Chat with your document")
if st.session_state.get("loaded_file"):
    st.caption(f"Currently querying: **{st.session_state.loaded_file}**")
else:
    st.warning("Please load a document from the sidebar to get accurate document-based answers.")

if not st.session_state.messages:
    # Empty state
    st.info("👋 Hello! Load a document and ask me anything about it.")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# 🧠 USER INPUT
# =========================
if prompt := st.chat_input("Ask something about the document..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Searching knowledge base..."):
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

            if res.status_code == 200:
                answer = res.json().get("answer", "")

                # 🔥 Fake streaming effect
                for chunk in answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)

                message_placeholder.markdown(full_response)

            else:
                st.error("❌ Error from backend")

        except Exception as e:
            st.error(f"⚠️ Network error: {str(e)}")

    # Save response
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Update local past_history
        if st.session_state.session_id not in st.session_state.past_history:
            st.session_state.past_history[st.session_state.session_id] = {
                "title": prompt[:30] + "..." if len(prompt) > 30 else prompt,
                "messages": st.session_state.messages.copy()
            }
        else:
            st.session_state.past_history[st.session_state.session_id]["messages"] = st.session_state.messages.copy()