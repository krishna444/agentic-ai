import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from config import settings
load_dotenv()

st.set_option("client.toolbarMode", "viewer")

#st.title("🤖 Qwen 3.6 Chatbot")

# Model options mapping
MODEL_OPTIONS = {
    "GROQ": "groq",
    "GEMINI": "genai"
}

# 1. Add model selection to the sidebar
selected_option = st.sidebar.selectbox(
    "Choose AI Model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)
#Dynamic Title based on selection
st.title(f"🤖 Chatbot: {selected_option}")

#st.divider()
selected_model_key = MODEL_OPTIONS[selected_option]


# ---- Memory Store Initialization ---
# Session state store to hold message history objects per thread
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]



if "llm_groq" not in st.session_state:
    st.session_state.llm_groq=ChatGroq(
        model="qwen/qwen3.6-27b",#settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
        #reasoning_effort="none"        
    )
if "llm_genai" not in st.session_state:
    st.session_state.llm_genai=ChatGoogleGenerativeAI(
        #model="gemma-4-31b-it",
        model=settings.GEMINI_MODEL,#"gemini-3.5-flash-lite",
        api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
        #thinking_budget=-1        
    )

base_llm = st.session_state.llm_groq if selected_model_key == "groq" else st.session_state.llm_genai

# Dynamic LLM resolution
active_llm = RunnableWithMessageHistory(
    runnable=base_llm, 
    get_session_history=get_session_history,
    input_message_key="input",
    history_message_key="history"
)

#st.sidebar.info(active_llm.model)
st.sidebar.info(f"Model# **{base_llm.model}**")

THREAD_ID="user-krishna-session-1"
config={
    "configurable":{
        "session_id":THREAD_ID}}

# Render conversation history on rerun
current_history = get_session_history(THREAD_ID)
for msg in current_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("What is on your mind?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        
        
    with st. chat_message("assistant"):
        def response_generator():            
            stream=active_llm.stream(prompt,config=config)
            for chunk in stream:
                if isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk.content, list) and len(chunk.content)>0:
                    for block in chunk.content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
        st.write_stream(response_generator())