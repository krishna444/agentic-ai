import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import settings

st.set_option("client.toolbarMode", "viewer")

# ---- Login gate ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        if username == settings.APP_USERNAME and password == settings.APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# ---- Sidebar: user + logout ----
with st.sidebar:
    st.write(f"👤 **{settings.APP_USERNAME}**")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.clear()
        st.rerun()
    st.divider()

PROVIDERS = {
    "GROQ": {
        "models": [settings.GROQ_MODEL],
        "api_key": settings.GROQ_API_KEY,
    },
    "GEMINI": {
        "models": [settings.GOOGLE_MODEL],
        "api_key": settings.GOOGLE_API_KEY,
        "base_url": settings.GOOGLE_BASE_URL,
    },
    "OCI": {
        "models": [settings.OCI_MODEL_GEMMA, settings.OCI_MODEL_QWEN],
        "api_key": settings.OCI_LLM_API_KEY,
        "base_url": settings.OCI_LLM_BASE_URL,
        "disable_thinking": True,
    },
    "OPENROUTER": {
        "models": [settings.OPENROUTER_MODEL_DEEPSEEK, settings.OPENROUTER_MODEL_GLM],
        "api_key": settings.OPENROUTER_API_KEY,
        "base_url": settings.OPENROUTER_BASE_URL,
    },
}


def build_llm(provider_key: str, model: str):
    spec = PROVIDERS[provider_key]
    if provider_key == "GROQ":
        return ChatGroq(
            model=model,
            api_key=spec["api_key"],
            temperature=0.7,
        )
    if provider_key == "GEMINI":
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=spec["api_key"],
            temperature=0.7,
            thinking_level="minimal",
        )
    extra_kwargs = {}
    if provider_key == "OCI":
        if spec.get("disable_thinking"):
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        extra_kwargs["temperature"] = None
    return ChatOpenAI(
        model=model,
        api_key=spec["api_key"],
        base_url=spec["base_url"],
        temperature=extra_kwargs.get("temperature", 0.2),
        **{k: v for k, v in extra_kwargs.items() if k != "temperature"},
    )


# ---- Provider / Model selection ----
selected_provider = st.sidebar.selectbox(
    "Choose Provider:",
    options=list(PROVIDERS.keys()),
    index=0,
)
provider_spec = PROVIDERS[selected_provider]

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    options=provider_spec["models"],
    index=0,
)
st.title(f"🤖 Chatbot: {selected_provider} / {selected_model}")

# ---- LLM cache (per provider+model) ----
if "llm_cache" not in st.session_state:
    st.session_state.llm_cache = {}

llm_cache_key = f"{selected_provider}::{selected_model}"
if llm_cache_key not in st.session_state.llm_cache:
    st.session_state.llm_cache[llm_cache_key] = build_llm(selected_provider, selected_model)
base_llm = st.session_state.llm_cache[llm_cache_key]

st.sidebar.info(f"Model# **{selected_model}**")

# ---- Memory Store ----
if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def _flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
            for block in content
        )
    return str(content)


def _normalize_messages(messages):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    normalized = [SystemMessage(content=settings.SYSTEM)]
    normalized += [
        HumanMessage(content=_flatten_content(m.content))
        if m.type == "human"
        else AIMessage(content=_flatten_content(m.content))
        for m in messages
    ]
    return normalized


chat_with_memory = _normalize_messages | base_llm

active_llm = RunnableWithMessageHistory(
    runnable=chat_with_memory,
    get_session_history=get_session_history,
)

THREAD_ID = "user-krishna-session-1"
config = {"configurable": {"session_id": THREAD_ID}}

# Render conversation history on rerun
current_history = get_session_history(THREAD_ID)
for msg in current_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("What is on your mind?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def response_generator():
            stream = active_llm.stream(prompt, config=config)
            for chunk in stream:
                if isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk.content, list) and len(chunk.content) > 0:
                    for block in chunk.content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
        st.write_stream(response_generator())
