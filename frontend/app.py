import streamlit as st
import requests

# Backend URLs
API_URL = "http://localhost:8000/ask"
UPLOAD_URL = "http://localhost:8000/upload"

st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖")

st.title("🤖 RAG Q&A Bot")
st.write("Ask questions and get answers based on the loaded knowledge base.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    st.write("Upload a file to train the RAG model.")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv", "docx"])
    
    if st.button("Upload"):
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
