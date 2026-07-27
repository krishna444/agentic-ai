# Agentic AI - Multi-Model AI Chatbot

A flexible, agentic AI chatbot application built with **Streamlit** and **LangChain**. This project supports streaming dynamic responses from leading AI model providers (such as Groq and Google GenAI) and features clean containerized deployment using **Docker** and **`uv`**.

---

## 📋 Table of Contents

- [Agentic AI - Multi-Model AI Chatbot](#agentic-ai---multi-model-ai-chatbot)
  - [📋 Table of Contents](#-table-of-contents)
  - [🏗 Architecture \& Design](#-architecture--design)
  - [🛠 Tech Stack \& Tools](#-tech-stack--tools)
  - [📁 Project Structure](#-project-structure)
  - [💡 Key Concepts: Agent Creation \& Calling](#-key-concepts-agent-creation--calling)
    - [1. Agent / LLM Creation (`app/prompt_service.py`)](#1-agent--llm-creation-appprompt_servicepy)
    - [2. Invoking and Streaming Responses](#2-invoking-and-streaming-responses)
  - [🚀 Local Setup \& Development](#-local-setup--development)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up Environment Variables](#2-set-up-environment-variables)
    - [3. Install Dependencies using `uv`](#3-install-dependencies-using-uv)
  - [🖥 Running the Streamlit UI](#-running-the-streamlit-ui)
  - [🐳 Deployment via Docker](#-deployment-via-docker)
    - [1. Build and Run with Docker Compose](#1-build-and-run-with-docker-compose)
    - [2. Verify Container Status](#2-verify-container-status)
    - [3. Access the Deployed App](#3-access-the-deployed-app)
  - [🔑 Environment Variables](#-environment-variables)

---

## 🏗 Architecture & Design

The application follows a clean 3-tier modular architecture:

1. **Configuration (`app/config.py`)**: Manages model definitions, technical identifiers, system prompts, and model provider configurations.
2. **Prompt & LLM Service (`app/prompt_service.py`)**: Handles dynamic model instantiation, LLM agent creation, system prompt wrapping, and unified response streaming across heterogeneous providers.
3. **UI View Layer (`app/view.py`)**: Renders an interactive Streamlit chat interface with sidebar model selection, dynamic model identifier captions, and full message history management.

---

## 🛠 Tech Stack & Tools

As defined in `pyproject.toml` and `deployment/requirements.txt`:

* **Framework & UI:** `streamlit` (>=1.30.0)
* **LLM Orchestration:** `langchain`, `langchain-core`
* **Model Integrations:**
  * `langchain-groq` (e.g., `qwen/qwen3.6-27b`)
  * `langchain-google-genai` (e.g., `gemini-3.5-flash-lite`)
* **Environment Management:** `python-dotenv`
* **Fast Python Package Installer:** `uv` (Astral)
* **Containerization:** `Docker` & `Docker Compose`

---

## 📁 Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py            # Model configs, provider maps, system prompts
│   ├── prompt_service.py   # Agent creation and invocation/streaming logic
│   └── view.py              # Streamlit interface entry point
├── deployment/
│   ├── Dockerfile           # Docker image definition using uv
│   ├── docker-compose.yml   # Multi-container orchestration
│   └── requirements.txt     # Dependency list for deployment
├── .env                     # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
├── .gitignore               # Ignored files (.env, .venv, etc.)
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Lockfile for deterministic builds
└── README.md                # Documentation
```

---

## 💡 Key Concepts: Agent Creation & Calling

### 1. Agent / LLM Creation (`app/prompt_service.py`)

Models are dynamically instantiated based on the configuration mapping defined in `app/config.py`:

```python
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm_agent(model_key: str):
    """Instantiates the configured LLM agent based on the user-selected key."""
    if model_key == "groq":
        return ChatGroq(
            model="qwen/qwen3.6-27b",
            temperature=0.7,
            reasoning_effort="none"
        )
    elif model_key == "genai":
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.7,
            thinking_budget=-1
        )
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
```

### 2. Invoking and Streaming Responses

To ensure real-time user feedback, responses are continuously streamed from the active LLM agent:

```python
def stream_agent_response(agent, user_prompt: str):
    """Yields streamed chunks from the agent for Streamlit rendering."""
    # Construct input with system prompt or direct prompt
    stream = agent.stream(user_prompt)
    for chunk in stream:
        if isinstance(chunk.content, str):
            yield chunk.content
        elif isinstance(chunk.content, list) and len(chunk.content) > 0:
            for block in chunk.content:
                if isinstance(block, dict) and "text" in block:
                    yield block["text"]
                elif hasattr(block, "text"):
                    yield block.text
```

---

## 🚀 Local Setup & Development

### 1. Clone the Repository

```bash
git clone https://github.com/krishna444/agentic-ai.git
cd agentic-ai
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Install Dependencies using `uv`

If you have `uv` installed locally:

```bash
uv sync
```

---

## 🖥 Running the Streamlit UI

To start the Streamlit web application locally, execute the view script via `uv`:

```bash
uv run streamlit run app/view.py
```

Or using standard Python virtual environment:

```bash
streamlit run app/view.py
```

Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🐳 Deployment via Docker

Deployment files are grouped in the `deployment/` directory.

### 1. Build and Run with Docker Compose

Run the deployment setup from your project root:

```bash
docker compose -f deployment/docker-compose.yml up -d --build
```

Alternatively, navigate into the `deployment` folder:

```bash
cd deployment
docker compose up -d --build
```

### 2. Verify Container Status

```bash
docker compose -f deployment/docker-compose.yml ps
```

### 3. Access the Deployed App

Open your web browser and go to:
```text
http://<YOUR_SERVER_IP>:8501
```

To view live container logs:
```bash
docker compose -f deployment/docker-compose.yml logs -f
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Yes | API key for Groq inference (`qwen/qwen3.6-27b`). |
| `GOOGLE_API_KEY` | Yes | API key for Google GenAI (`gemini-3.5-flash-lite`). |