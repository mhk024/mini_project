import streamlit as st
import requests

# Backend URLs
API_URL    = "http://127.0.0.1:8000/ask"
UPLOAD_URL = "http://127.0.0.1:8000/upload"
LOGIN_URL  = "http://127.0.0.1:8000/login"
SIGNUP_URL = "http://127.0.0.1:8000/signup"

st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_user(username, password):
    try:
        response = requests.post(LOGIN_URL, json={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Error: Cannot connect to the backend. Make sure the FastAPI server is running.")

def signup_user(username, password):
    try:
        response = requests.post(SIGNUP_URL, json={"username": username, "password": password})
        if response.status_code == 200:
            st.success("Signup successful! Please log in.")
        else:
            st.error(f"Signup failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Error: Cannot connect to the backend. Make sure the FastAPI server is running.")


if not st.session_state.logged_in:
    st.title("Welcome to RAG Q&A Bot 🤖")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_username and login_password:
                login_user(login_username, login_password)
            else:
                st.warning("Please enter username and password")
                
    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("New Username", key="signup_username")
        signup_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if signup_username and signup_password:
                signup_user(signup_username, signup_password)
            else:
                st.warning("Please enter username and password")

else:
    # --- MAIN LOGGED IN APP ---
    st.title(f"🤖 RAG Q&A Bot (Logged in as {st.session_state.username})")
    st.write("Ask questions and get answers based on the loaded knowledge base.")

    # Sidebar for file upload
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
            
        st.divider()
        st.header("Upload Document")
        st.write("Upload a file to train the RAG model.")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
        
        if st.button("Upload Document"):
            if uploaded_file is not None:
                with st.spinner('Uploading...'):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(UPLOAD_URL, files=files)
                        
                        if response.status_code == 200:
                            st.success("File uploaded successfully!")
                        else:
                            st.error(f"Failed to upload file. Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Error: Cannot connect to the backend.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please select a file first.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare logic for getting bot response
        try:
            with st.spinner('Thinking...'):
                response = requests.post(
                    API_URL, 
                    json={"question": prompt}, 
                    timeout=30 # Add timeout to prevent hanging if backend is slow
                )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer found.")
            elif response.status_code == 503:
                 answer = "Error: RAG not loaded. Please wait for the backend to finish initializing."
            else:
                 answer = f"Error: Failed to get response from backend. Status code: {response.status_code}\nDetails: {response.text}"

        except requests.exceptions.ConnectionError:
            answer = "Error: Cannot connect to the backend. Make sure the FastAPI server is running on http://localhost:8000"
        except Exception as e:
            answer = f"An unexpected error occurred: {e}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
