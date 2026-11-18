import html
import time

import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import settings

st.set_option("client.toolbarMode", "viewer")

st.markdown(
    """
    <style>
        [data-testid="stChatMessage"] {
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
            margin-bottom: 0rem;
        }
        [data-testid="stChatMessageContent"] p {
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        expected = settings.users_map.get(username.strip())
        if expected is not None and password == expected:
            st.session_state.authenticated = True
            st.session_state.username = username.strip()
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

# ---- Sidebar: user + logout ----
with st.sidebar:
    st.title("Welcome to AI Chatbot")
    st.write(f"👤 **{st.session_state.username}**")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.clear()
        st.rerun()
    st.divider()

PROVIDERS = {
    "OCI": {
        "models": [settings.OCI_MODEL_GEMMA, settings.OCI_MODEL_QWEN],
        "api_key": settings.OCI_LLM_API_KEY,
        "base_url": settings.OCI_LLM_BASE_URL,
        "disable_thinking": True,
    },
    "GROQ": {
        "models": [settings.GROQ_MODEL],
        "api_key": settings.GROQ_API_KEY,
    },
    "GEMINI": {
        "models": [settings.GOOGLE_MODEL],
        "api_key": settings.GOOGLE_API_KEY,
        "base_url": settings.GOOGLE_BASE_URL,
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

THREAD_ID = f"user-{st.session_state.username}-session-1"
config = {"configurable": {"session_id": THREAD_ID}}

def _render_user_message(text: str):
    safe = html.escape(text)
    st.markdown(
        f'<span style="color:#0B62A4;font-weight:600;">{safe}</span>',
        unsafe_allow_html=True,
    )


# Render conversation history on rerun
current_history = get_session_history(THREAD_ID)
for msg in current_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        if role == "user":
            _render_user_message(msg.content)
        else:
            st.markdown(msg.content)

if prompt := st.chat_input("What is on your mind?"):
    with st.chat_message("user"):
        _render_user_message(prompt)

    with st.chat_message("assistant"):
        stats = {}

        def response_generator():
            start = time.perf_counter()
            out_tokens = 0
            in_tokens = None
            full_text = ""
            stream = active_llm.stream(prompt, config=config)
            for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None)
                if usage and usage.get("output_tokens"):
                    out_tokens = usage["output_tokens"]
                if usage and usage.get("input_tokens"):
                    in_tokens = usage["input_tokens"]
                if isinstance(chunk.content, list) and chunk.content:
                    text = "".join(
                        block["text"]
                        for block in chunk.content
                        if isinstance(block, dict) and "text" in block
                    )
                elif isinstance(chunk.content, str):
                    text = chunk.content
                else:
                    text = ""
                if not text:
                    continue
                full_text += text
                yield text
            stats["out_tokens"] = out_tokens or max(len(full_text) // 4, 1)
            stats["in_tokens"] = in_tokens
            stats["elapsed"] = max(time.perf_counter() - start, 0.001)

        st.write_stream(response_generator())

        out_tokens = stats.get("out_tokens", 0)
        elapsed = stats.get("elapsed", 0)
        in_tokens = stats.get("in_tokens")
        st.markdown(
            f"<small style='color:gray;'>"
            f"{out_tokens} tokens · {elapsed:.1f}s · {out_tokens / elapsed:.1f} tok/s"
            + (f" · prompt {in_tokens} tokens" if in_tokens else "")
            + "</small>",
            unsafe_allow_html=True,
        )
