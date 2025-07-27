"""
Professional AI Chat Application using Streamlit and LangChain
Advanced interface with memory, prompt templates, and conversation chains.
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY_PREFIX = "sk-"
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_MEMORY_WINDOW = 10
MAX_TOKEN_LIMIT = 1000

# Custom Streamlit Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit integration."""
    
    def __init__(self):
        self.container = st.empty()
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.text += token
        self.container.markdown(self.text)

# Page Configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .conversation-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return bool(api_key and api_key.startswith(API_KEY_PREFIX))

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'total_tokens_used' not in st.session_state:
        st.session_state.total_tokens_used = 0

def create_prompt_template(persona: str) -> PromptTemplate:
    """Create a custom prompt template based on selected persona."""
    
    personas = {
        "Assistant": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses.
        
Current conversation:
{history}
Human: {input}
AI Assistant:""",
        
        "Teacher": """You are an experienced teacher and educator. Explain concepts clearly, provide examples, 
        and encourage learning. Break down complex topics into understandable parts.
        
Current conversation:
{history}
Human: {input}
AI Teacher:""",
        
        "Developer": """You are a senior software developer with expertise in multiple programming languages and technologies. 
        Provide practical coding solutions, best practices, and technical guidance.
        
Current conversation:
{history}
Human: {input}
AI Developer:""",
        
        "Creative": """You are a creative writing assistant. Help with storytelling, creative ideas, 
        and imaginative content. Be inspiring and think outside the box.
        
Current conversation:
{history}
Human: {input}
AI Creative:""",
        
        "Analyst": """You are a data analyst and business intelligence expert. Provide analytical insights, 
        break down complex data, and offer strategic recommendations.
        
Current conversation:
{history}
Human: {input}
AI Analyst:"""
    }
    
    template = personas.get(persona, personas["Assistant"])
    return PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

def create_conversation_chain(api_key: str, model: str, persona: str, 
                            memory_type: str, temperature: float) -> Optional[ConversationChain]:
    """Create a LangChain conversation chain with memory."""
    try:
        # Initialize the language model
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=temperature,
            streaming=True
        )
        
        # Create memory based on type
        if memory_type == "Window":
            memory = ConversationBufferWindowMemory(
                k=MAX_MEMORY_WINDOW,
                return_messages=True
            )
        else:  # Summary
            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=MAX_TOKEN_LIMIT,
                return_messages=True
            )
        
        # Create prompt template
        prompt = create_prompt_template(persona)
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        
        return conversation
        
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {str(e)}")
        logger.error(f"Error creating conversation chain: {e}")
        return None

def generate_response(chain: ConversationChain, user_input: str) -> Optional[str]:
    """Generate response using LangChain conversation chain."""
    try:
        response = chain.predict(input=user_input)
        return response.strip()
    except Exception as e:
        error_message = str(e).lower()
        
        if "authentication" in error_message or "api key" in error_message:
            st.error("❌ Invalid API key. Please check your OpenAI API key.")
        elif "rate limit" in error_message or "quota" in error_message:
            st.error("⏱️ Rate limit exceeded. Please try again later.")
        elif "insufficient" in error_message or "billing" in error_message:
            st.error("💳 Insufficient credits. Please check your OpenAI account balance.")
        else:
            st.error(f"❌ Error generating response: {str(e)}")
        
        logger.error(f"Error generating response: {e}")
        return None

def export_conversation_advanced() -> Dict[str, Any]:
    """Export conversation with metadata."""
    return {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(st.session_state.chat_history),
        "estimated_tokens": st.session_state.total_tokens_used,
        "conversation": st.session_state.chat_history
    }

def display_chat_history() -> None:
    """Display chat history with improved formatting."""
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        for i, (human_msg, ai_msg, timestamp) in enumerate(st.session_state.chat_history):
            with st.container():
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown(f"**{i+1}**")
                with col2:
                    st.markdown(f"<small>{timestamp}</small>", unsafe_allow_html=True)
                
                # User message
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You:</strong> {human_msg}
                </div>
                """, unsafe_allow_html=True)
                
                # AI message
                st.markdown(f"""
                <div class="ai-message">
                    <strong>🤖 AI:</strong> {ai_msg}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")

def main() -> None:
    """Main application logic."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Professional AI Assistant</h1>
        <p>Powered by LangChain & OpenAI with Advanced Memory Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model Selection
        model = st.selectbox(
            "🤖 Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o-mini"],
            help="Choose the AI model"
        )
        
        # Persona Selection
        persona = st.selectbox(
            "🎭 AI Persona",
            options=["Assistant", "Teacher", "Developer", "Creative", "Analyst"],
            help="Choose the AI's personality and expertise"
        )
        
        # Memory Type
        memory_type = st.selectbox(
            "🧠 Memory Type",
            options=["Window", "Summary"],
            help="Window: Remembers last N messages, Summary: Summarizes old conversations"
        )
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Creativity level: 0 = deterministic, 2 = very creative"
            )
            
            auto_save = st.checkbox(
                "💾 Auto-save conversations",
                value=True,
                help="Automatically save conversation history"
            )
        
        # Initialize or update conversation chain
        if st.button("🔄 Initialize/Update Chain", use_container_width=True):
            if validate_api_key(api_key):
                with st.spinner("Initializing AI chain..."):
                    st.session_state.conversation_chain = create_conversation_chain(
                        api_key, model, persona, memory_type, temperature
                    )
                if st.session_state.conversation_chain:
                    st.success("✅ Chain initialized successfully!")
            else:
                st.error("❌ Please enter a valid API key")
        
        # Session Info
        st.markdown("### 📊 Session Info")
        st.info(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.metric("Messages", len(st.session_state.chat_history))
        st.metric("Est. Tokens", st.session_state.total_tokens_used)
        
        # Actions
        st.markdown("### 🛠️ Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.total_tokens_used = 0
                if st.session_state.conversation_chain:
                    st.session_state.conversation_chain.memory.clear()
                st.success("Cleared!")
        
        with col2:
            if st.session_state.chat_history:
                export_data = json.dumps(export_conversation_advanced(), indent=2)
                st.download_button(
                    "📥 Export",
                    data=export_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Main Chat Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💭 Chat with AI")
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Enter your message:",
                height=100,
                placeholder="Ask me anything...",
                help="Type your question or message here"
            )
            
            col_submit, col_example = st.columns([1, 1])
            
            with col_submit:
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)
            
            with col_example:
                if st.form_submit_button("💡 Example", use_container_width=True):
                    examples = {
                        "Assistant": "Explain quantum computing in simple terms",
                        "Teacher": "How do I learn Python programming effectively?",
                        "Developer": "Show me a Python function to sort a list",
                        "Creative": "Write a short story about a time traveler",
                        "Analyst": "Analyze the pros and cons of remote work"
                    }
                    user_input = examples.get(persona, "Hello! How can you help me today?")
                    submitted = True
        
        # Process input
        if submitted and user_input.strip():
            if not st.session_state.conversation_chain:
                st.warning("⚠️ Please initialize the conversation chain first!")
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                with st.spinner("🤔 AI is thinking..."):
                    response = generate_response(st.session_state.conversation_chain, user_input)
                
                if response:
                    # Display current conversation
                    st.markdown("### 💬 Current Conversation")
                    
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>👤 You ({timestamp}):</strong> {user_input}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="ai-message">
                        <strong>🤖 AI ({timestamp}):</strong> {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.chat_history.append((user_input, response, timestamp))
                    
                    # Estimate tokens (rough calculation)
                    estimated_tokens = len(user_input.split()) + len(response.split())
                    st.session_state.total_tokens_used += estimated_tokens
                    
                    st.success("✅ Response generated and saved!")
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        # Persona-specific suggestions
        suggestions = {
            "Assistant": ["Explain a concept", "Help with a task", "Answer questions"],
            "Teacher": ["Learn a subject", "Get explanations", "Practice problems"],
            "Developer": ["Code review", "Debug help", "Best practices"],
            "Creative": ["Story ideas", "Creative writing", "Brainstorming"],
            "Analyst": ["Data analysis", "Market research", "Strategic planning"]
        }
        
        st.markdown(f"**{persona} Mode Suggestions:**")
        for suggestion in suggestions.get(persona, []):
            st.markdown(f"• {suggestion}")
        
        # Memory status
        if st.session_state.conversation_chain:
            st.markdown("### 🧠 Memory Status")
            memory_content = st.session_state.conversation_chain.memory.buffer
            if memory_content:
                st.info(f"Memory contains {len(memory_content)} messages")
            else:
                st.info("Memory is empty")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Initialize the chain before chatting
        - Choose appropriate persona for your task
        - Use specific, clear questions
        - Memory persists across questions
        - Export important conversations
        """)
    
    # Display full chat history
    if st.session_state.chat_history:
        display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        f"Professional AI Assistant • Persona: {persona} • Memory: {memory_type}<br>"
        "<small>Built with LangChain, Streamlit & OpenAI</small>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()