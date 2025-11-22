import streamlit as st
import re
import time
from vector_store import VectorStoreManager
from llm_qa import QAEngine
import config

# ==========================================
# 🎨 UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="IMF Insight",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_citation_text(text):
    """Cleans raw chunk text for better readability."""
    text = re.sub(r'###\s*[A-Z\s]+', '', text)
    text = text.replace('|', ' ').replace('---', '')
    text = text.replace('</div>', '').replace('<div>', '')
    text = " ".join(text.split())
    if len(text) > 300:
        text = text[:300] + "..."
    return text

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0e1117; }
        h1 { color: #ffffff; font-weight: 700; letter-spacing: -0.02em; }
        .stChatMessage { background-color: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 2rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #0d1117; border: 1px solid #30363d; }
        .stMarkdown p { font-size: 1.15rem; line-height: 1.6; color: #e6edf3; }
        .citation-box { margin-top: 1rem; background: #0d1117; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; }
        .citation-header { background: #21262d; padding: 10px 20px; border-bottom: 1px solid #30363d; font-size: 0.75rem; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
        .citation-item { padding: 14px 20px; border-bottom: 1px solid #21262d; transition: background 0.2s; }
        .citation-item:last-child { border-bottom: none; }
        .citation-item:hover { background: #1f242c; }
        .page-badge { display: inline-block; background: rgba(63, 185, 80, 0.15); color: #3fb950; padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; margin-bottom: 6px; border: 1px solid rgba(63, 185, 80, 0.2); }
        .quote-text { font-size: 0.9rem; color: #c9d1d9; line-height: 1.5; }
        .stTextInput > div > div > input { border-radius: 12px; background-color: #0d1117; border: 1px solid #30363d; color: #fff; padding: 12px; }
        #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    if not config.VECTOR_STORE_PATH.exists():
        return None
    try:
        manager = VectorStoreManager()
        # Force reload of retriever to ensure connection is fresh
        retriever = manager.get_retriever()
        qa_engine = QAEngine()
        return {'retriever': retriever, 'qa_engine': qa_engine}
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None

def main():
    inject_custom_css()
    
    st.markdown("# 📘 IMF Insight")
    st.markdown("<p style='color: #8b949e; font-size: 1.1rem;'>Simple answers from complex financial reports.</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    resources = load_resources()

    with st.sidebar:
        st.markdown("### 🟢 System Status")
        if resources:
            st.markdown(f"<div style='font-size:0.85rem; color:#8b949e; margin-bottom:1rem;'>Connected to <strong>{config.VECTOR_STORE_PATH.name}</strong></div>", unsafe_allow_html=True)
            # Button to clear history - Critical for long sessions with small models
            if st.button("✨ New Conversation (Clear Memory)", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            st.error("System Offline")
            st.info("Please run the pipeline first.")

    # Render History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("🔎 View Evidence", expanded=False):
                    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                    st.markdown('<div class="citation-header">Source Highlights</div>', unsafe_allow_html=True)
                    for cit in message["citations"]:
                        clean_snippet = clean_citation_text(cit['snippet'])
                        st.markdown(f"""
                        <div class="citation-item">
                            <div class="page-badge">PAGE {cit['page']}</div>
                            <div class="quote-text">"{clean_snippet}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Input Loop
    if prompt := st.chat_input("Ask me a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                with st.spinner("Thinking..."):
                    if not resources:
                        st.error("Backend not loaded.")
                        st.stop()

                    retriever = resources['retriever']
                    qa = resources['qa_engine']
                    
                    # 1. Retrieve Docs
                    # We deliberately do NOT pass chat_history to the retriever here
                    # to keep the context window clean for the 1B model.
                    docs = retriever.invoke(prompt)
                    
                    # 2. Generate Answer
                    result = qa.answer_question(prompt, docs)
                    
                    answer = result['answer']
                    citations = result['citations']

                placeholder.markdown(answer)
                
                if citations:
                    with st.expander("🔎 View Evidence", expanded=True):
                        st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                        st.markdown('<div class="citation-header">Source Highlights</div>', unsafe_allow_html=True)
                        for cit in citations:
                            clean_snippet = clean_citation_text(cit['snippet'])
                            st.markdown(f"""
                            <div class="citation-item">
                                <div class="page-badge">PAGE {cit['page']}</div>
                                <div class="quote-text">"{clean_snippet}"</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations
                })
                
            except Exception as e:
                st.error("I ran into a momentary issue. Please try asking again!")
                # In cloud/hybrid modes, printing the error to console helps debugging logs
                print(f"Inference Error: {e}")

if __name__ == "__main__":
    main()