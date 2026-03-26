import streamlit as st
import requests
import time

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
# 🧠 SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# 🔐 AUTH FUNCTIONS
# =========================
def login_user(username, password):
    res = requests.post(LOGIN_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username = username

        # Load history
        try:
            hist = requests.get(f"{HISTORY_URL}/{username}")
            if hist.status_code == 200:
                st.session_state.messages = hist.json().get("history", [])
        except:
            pass

        st.rerun()
    else:
        st.error("Invalid credentials")


def signup_user(username, password):
    res = requests.post(SIGNUP_URL, json={"username": username, "password": password})
    if res.status_code == 200:
        st.success("Signup successful!")
    else:
        st.error(res.json().get("detail", "Signup failed"))


# =========================
# 🔑 LOGIN PAGE
# =========================
if not st.session_state.logged_in:
    st.title("🤖 RAG Chat App")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            login_user(u, p)

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Signup"):
            signup_user(u, p)

    st.stop()


# =========================
# 🧭 SIDEBAR
# =========================
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

    st.divider()

    # File selection
    st.subheader("📂 Documents")
    try:
        files = requests.get(FILES_URL, timeout=5).json().get("files", [])
    except:
        files = []

    selected = st.selectbox("Select file", [""] + files)

    if "loaded_file" not in st.session_state:
        st.session_state.loaded_file = ""

    if selected and st.button("Load File"):
        res = requests.post(SET_FILE_URL, json={"filename": selected}, timeout=300)
        if res.status_code == 200:
            st.session_state.loaded_file = selected
            st.success(f"✅ Loaded: {selected}")
        else:
            st.error("Failed to load")

    if st.session_state.loaded_file:
        st.info(f"📄 Active: {st.session_state.loaded_file}")

    st.divider()

    # Upload
    st.subheader("⬆ Upload")
    file = st.file_uploader("PDF/TXT", type=["pdf", "txt"])

    if st.button("Upload"):
        if file:
            files = {"file": (file.name, file, file.type)}
            res = requests.post(UPLOAD_URL, files=files, timeout=300)
            if res.status_code == 200:
                st.session_state.loaded_file = file.name
                st.success("Uploaded!")
            else:
                st.error("Upload failed")

# =========================
# 💬 CHAT UI
# =========================
st.title("💬 Chat with your document")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# 🧠 USER INPUT
# =========================
if prompt := st.chat_input("Ask something..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Assistant response container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            res = requests.post(
                API_URL,
                json={"username": st.session_state.username, "question": prompt},
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
                message_placeholder.markdown("❌ Error from backend")

        except Exception as e:
            message_placeholder.markdown(f"⚠️ {str(e)}")

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": full_response})