"""
Streamlit UI for the Multi-Agent Analyst system.
"""

import os
import uuid
import streamlit as st
import requests

# Configuration - Use environment variable or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state for conversation memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page config
st.set_page_config(
    page_title="Multi-Agent Analyst",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Multi-Agent Analyst")
st.markdown("Query the system using RAG, Database, or Forecasting agents.")

# Sidebar - Health Check
with st.sidebar:
    st.header("⚙️ System Status")
    
    if st.button("Check Health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['status'].upper()}")
                st.info(data['message'])
            else:
                st.error(f"❌ Unhealthy (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Is the server running?")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Session management
    st.markdown("### 🧠 Session")
    st.caption(f"ID: {st.session_state.session_id[:8]}...")
    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    **RAG (Knowledge):**
    - What are the error codes?
    - Explain the troubleshooting steps
    
    **Database:**
    - Show ticket count for last 3 days
    - Get statistics from 2026-01-25
    
    **Forecasting:**
    - Forecast tickets for next 5 days
    - Predict ticket volume for 7 days
    """)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("work"):
            with st.expander("🔍 Agent Work Details", expanded=False):
                st.json(msg["work"])

# Chat input at the bottom
if user_input := st.chat_input("Type your message..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": user_input, "session_id": st.session_state.session_id},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.write(data['answer'])
                    with st.expander("🔍 Agent Work Details", expanded=False):
                        st.json(data['work'])
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": data['answer'],
                        "work": data['work']
                    })
                else:
                    error_msg = response.json().get('detail', 'Unknown error')
                    st.error(f"Error: {error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the server is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.caption("Multi-Agent Analyst | RAG • Database • Forecasting")
