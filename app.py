# -*- coding: utf-8 -*-
# File: app.py
# File: app.py
import streamlit as st
import os
import streamlit as st
import os
import dotenv
import uuid
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from nda_chatbot import NDADocumentChatbot
import tempfile
from typing import Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Strada NDA Analysis Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        color: #1565c0 !important;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #4caf50;
        color: #2e7d32 !important;
    }
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Fix for chat history visibility */
    .stExpanderContent {
        background-color: #ffffff !important;
    }
    .chat-history-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False

def validate_api_key(api_key: str) -> bool:
    """Simple validation for API key format"""
    return api_key and api_key.startswith('sk-') and len(api_key) > 40

def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong style="color: #1565c0;">You:</strong><br>
            <span style="color: #1976d2;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong style="color: #2e7d32;">Assistant:</strong><br>
            <span style="color: #388e3c;">{message}</span>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Company Logo and Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Option 1: If you have a logo file in your repo
        try:
            st.image("assets/logo.png", width=200)  # Adjust width as needed
        except:
            # Option 2: Try URL-based logo (replace with your actual logo URL)
            try:
                st.image("https://your-website.com/strada-partners-logo.png", width=200)
            except:
                # Option 3: Fallback to text-based logo if image not found
                st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <h2 style="color: #1f77b4; margin: 0;">📄 Strada Partners</h2>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Legal Document Analysis</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📄 Strada NDA Analysis Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload an NDA document and chat with AI to get comprehensive analysis and answers to your questions.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="openai_api_key",
            help="Enter your OpenAI API key to enable the chatbot"
        )
        
        if api_key:
            if validate_api_key(api_key):
                if not st.session_state.api_key_valid:
                    try:
                        st.session_state.chatbot = NDADocumentChatbot(openai_api_key=api_key)
                        st.session_state.api_key_valid = True
                        st.success("✅ API key validated!")
                    except Exception as e:
                        st.error(f"❌ Error initializing chatbot: {str(e)}")
            else:
                st.error("❌ Invalid API key format")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
        
        st.markdown("---")
        
        # Memory controls
        if st.session_state.chatbot:
            st.subheader("💭 Memory Management")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_memory_btn"):
                st.session_state.chatbot.clear_memory()
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
            
            # Show conversation count and memory stats
            if hasattr(st.session_state.chatbot, 'memory'):
                history = st.session_state.chatbot.get_conversation_history()
                memory_stats = st.session_state.chatbot.get_memory_stats()
                
                st.info(f"📊 Conversations: {len(history)}")
                st.info(f"💭 Memory: {memory_stats['memory_messages']}/{memory_stats['memory_limit']} messages")
                
                # Show memory usage bar
                if memory_stats['memory_usage_percent'] > 0:
                    st.progress(memory_stats['memory_usage_percent'] / 100, 
                              text=f"Memory Usage: {memory_stats['memory_usage_percent']:.0f}%")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Enter API Key**: Add your OpenAI API key above
            2. **Upload NDA**: Upload a PDF NDA document
            3. **Chat**: Ask questions or request analysis
            
            **Example Queries:**
            - "Analyze this NDA comprehensively"
            - "Who are the parties involved?"
            - "What are the confidentiality obligations?"
            - "How long does this NDA last?"
            """)
    
    # Main content area
    if not st.session_state.api_key_valid:
        st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
        return
    
    # File upload section
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose an NDA PDF file",
        type=['pdf'],
        help="Upload a PDF NDA document for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document
        with st.spinner("📖 Loading NDA document..."):
            success = st.session_state.chatbot.load_nda_document(tmp_file_path)
            
            if success:
                st.session_state.document_loaded = True
                st.success(f"✅ NDA loaded successfully!")
                
                # Show document stats
                try:
                    stats = st.session_state.chatbot.get_nda_stats()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Pages", f"{stats.get('pages', 'N/A')}")
                    with col2:
                        st.metric("📝 Words", f"{stats.get('words', 'N/A'):,}")
                    with col3:
                        st.metric("⏱️ Reading Time", stats.get('reading_time', 'N/A'))
                except:
                    st.info("Document loaded successfully")
            else:
                st.error("❌ Failed to load the document. Please try again.")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
    
    # Chat interface
    if st.session_state.document_loaded:
        st.header("💬 Chat with Your NDA")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 Chat History")
            
            # Create a container with better styling
            for i, exchange in enumerate(st.session_state.chat_history):
                with st.expander(f"💬 Conversation {i+1}: {exchange['user'][:50]}{'...' if len(exchange['user']) > 50 else ''}", expanded=False):
                    st.markdown(f"""
                    <div class="chat-history-container">
                        <div class="chat-message user-message">
                            <strong style="color: #1565c0;">You:</strong><br>
                            <span style="color: #1976d2;">{exchange['user']}</span>
                        </div>
                        <div class="chat-message assistant-message">
                            <strong style="color: #2e7d32;">Assistant:</strong><br>
                            <span style="color: #388e3c;">{exchange['assistant']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<hr style='margin: 0.5rem 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Chat input with Enter key support
        user_input = st.chat_input(
            "Ask a question about the NDA...",
            key="chat_input_field"
        )
        
        # Quick action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Full Analysis", use_container_width=True, key="btn_analysis"):
                user_input = "Analyze this NDA comprehensively"
        
        with col2:
            if st.button("👥 Parties", use_container_width=True, key="btn_parties"):
                user_input = "Who are the parties involved in this NDA?"
        
        with col3:
            if st.button("🔒 Obligations", use_container_width=True, key="btn_obligations"):
                user_input = "What are the confidentiality obligations?"
        
        with col4:
            if st.button("⏰ Duration", use_container_width=True, key="btn_duration"):
                user_input = "How long does this NDA last?"
        
        # Process user input (from chat_input or quick action buttons)
        if user_input:
            with st.spinner("🤔 Analyzing..."):
                try:
                    # Get response from chatbot
                    result = st.session_state.chatbot.chat_with_nda(user_input)
                    
                    # Display new messages  
                    st.subheader("💬 Latest Response")
                    
                    # User message
                    st.markdown(f"""
                    <div class="chat-history-container">
                        <div class="chat-message user-message">
                            <strong style="color: #1565c0;">You:</strong><br>
                            <span style="color: #1976d2;">{user_input}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Assistant response with proper markdown rendering
                    st.markdown("**🤖 Assistant:**")
                    st.markdown(result["response"])
                    
                    # Show intent and sources info
                    col1, col2 = st.columns(2)
                    with col1:
                        intent_emoji = {"SUMMARIZE": "📋", "QUESTION": "❓", "GENERAL": "💬"}
                        st.info(f"{intent_emoji.get(result['intent'], '🤖')} Intent: {result['intent']}")
                    
                    with col2:
                        if result.get("sources"):
                            st.info(f"📚 Sources: {len(result['sources'])} document sections")
                    
                    # Show sources if available
                    if result.get("sources"):
                        with st.expander("📖 View Source Documents"):
                            for i, doc in enumerate(result["sources"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.markdown("---")
                    
                    # Update chat history
                    st.session_state.chat_history = st.session_state.chatbot.get_conversation_history()
                    
                    # Auto-refresh to show new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing request: {str(e)}")
    
    else:
        st.info("👆 Please upload an NDA document to start chatting!")
    
    # Footer with logo option
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    

if __name__ == "__main__":
    main()
