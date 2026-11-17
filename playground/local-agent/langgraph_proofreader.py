import os
import sys
import time
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# 1. Structured Output Schema (used with Qwen3 via .with_structured_output)
class ProofreadIssue(BaseModel):
    line_number: int = Field(description="1-based line number where the error occurs")
    error_type: str = Field(description="Either 'spelling' or 'grammar'")
    original_text: str = Field(description="Exact incorrect word or phrase")
    suggested_fix: str = Field(description="Corrected word or phrase")
    explanation: str = Field(description="Brief reason why this is incorrect")

class QwenAnalysis(BaseModel):
    issues: list[ProofreadIssue] = Field(description="All spelling and grammar errors found")
    corrected_text: str = Field(description="Full corrected document text WITHOUT line-number prefixes")

# 2. Graph State (shared memory passed from node to node)
class ProofreadState(TypedDict):
    file_path: str
    numbered_content: str
    issues: list[dict]
    corrected_text: str
    summary: str
    approved: bool
    saved_path: str

# 3. Initialize Both OCI Router Models
qwen_llm = ChatOpenAI(
    base_url=os.getenv("OCI_LLM_BASE_URL", "http://10.0.0.13:8081/v1"),
    api_key=os.getenv("OCI_LLM_API_KEY", "not-needed"),
    model=os.getenv("PROOFREADER_MODEL_QWEN", "Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf"),
    temperature=0.1,
)

gemma_llm = ChatOpenAI(
    base_url=os.getenv("OCI_LLM_BASE_URL", "http://10.0.0.13:8081/v1"),
    api_key=os.getenv("OCI_LLM_API_KEY", "not-needed"),
    model=os.getenv("PROOFREADER_MODEL_GEMMA", "gemma-4-E2B-it-UD-Q4_K_XL.gguf"),
    temperature=0.2,
)

# 4. Define Graph Nodes
def read_file_node(state: ProofreadState) -> dict:
    """Node 1 (Deterministic Python): Reads file and adds line numbers."""
    print(f"[Node 1 - Read] Loading '{state['file_path']}'...")
    path = Path(state["file_path"]).expanduser().resolve()
    content = path.read_text(encoding="utf-8")
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(content.splitlines(), start=1))
    return {"numbered_content": numbered}

def qwen_proofread_node(state: ProofreadState) -> dict:
    """Node 2 (Qwen3-30B-A3B): Finds errors and produces corrected text in 1 single turn."""
    print("[Node 2 - Qwen3] Analyzing grammar/spelling and generating fixes...")
    structured_qwen = qwen_llm.with_structured_output(QwenAnalysis)

    prompt = (
        "You are a meticulous proofreader. Analyze the line-numbered text below.\n"
        "Return all spelling/grammar issues with accurate 1-based line numbers, plus the full "
        "corrected text with line-number prefixes removed.\n\n"
        f"{state['numbered_content']}"
    )
    analysis: QwenAnalysis = structured_qwen.invoke(prompt)
    return {
        "issues": [issue.model_dump() for issue in analysis.issues],
        "corrected_text": analysis.corrected_text,
    }

def gemma_summarize_node(state: ProofreadState) -> dict:
    """Node 3 (Gemma 4 E2B): Generates a fast executive summary of the changes."""
    print("[Node 3 - Gemma 4 E2B] Summarizing editorial changes...")
    prompt = (
        f"Write a crisp 2-sentence summary of this corrected document ({len(state['issues'])} fixes applied):\n\n"
        f"{state['corrected_text']}"
    )
    response = gemma_llm.invoke(prompt)
    return {"summary": str(response.content)}

def human_approval_node(state: ProofreadState) -> dict:
    """Node 4 (Human-in-the-Loop): Displays proposed fixes and asks for confirmation."""
    print("\n--- Proposed Fixes ---")
    for issue in state["issues"]:
        print(
            f"Line {issue['line_number']} [{issue['error_type']}]: "
            f"'{issue['original_text']}' -> '{issue['suggested_fix']}' ({issue['explanation']})"
        )
    print(f"\nSummary (Gemma): {state['summary']}")

    choice = input("\nSave corrected file to disk? [y/N]: ").strip().lower()
    return {"approved": choice == "y"}

def write_file_node(state: ProofreadState) -> dict:
    """Node 5 (Deterministic Python): Saves '<name>_corrected.<ext>' if approved."""
    src = Path(state["file_path"]).expanduser().resolve()
    dest = src.with_name(f"{src.stem}_corrected{src.suffix}")
    dest.write_text(state["corrected_text"].rstrip("\n") + "\n", encoding="utf-8")
    print(f"[Node 5 - Write] Saved corrected file to: {dest}")
    return {"saved_path": str(dest)}

# 5. Conditional Edge Routing
def route_after_approval(state: ProofreadState) -> str:
    """Routes to `write_file` if user typed 'y', otherwise skips directly to END."""
    if state["approved"]:
        return "write_file"
    print("[Skipped] Changes discarded by user.")
    return END

# 6. Assemble and Compile the StateGraph
builder = StateGraph(ProofreadState)
builder.add_node("read_file", read_file_node)
builder.add_node("qwen_proofread", qwen_proofread_node)
builder.add_node("gemma_summarize", gemma_summarize_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("write_file", write_file_node)

builder.add_edge(START, "read_file")
builder.add_edge("read_file", "qwen_proofread")
builder.add_edge("qwen_proofread", "gemma_summarize")
builder.add_edge("gemma_summarize", "human_approval")
builder.add_conditional_edges("human_approval", route_after_approval)
builder.add_edge("write_file", END)

graph = builder.compile()

if __name__ == "__main__":
    TARGET_FILE = "drafts/notes.txt"
    start_time = time.perf_counter()

    final_state = graph.invoke({"file_path": TARGET_FILE})

    elapsed = time.perf_counter() - start_time
    print(f"\nGraph finished in {elapsed:.2f}s")
    sys.exit(0)   
    

def sqrt(num):
    return num ** 0.5

if __name__ == "__main__":
    print(sqrt(16))  # Output: 4.0
    print(sqrt(25))  # Output: 5.0
    print(sqrt(2))   # Output: 1.4142135623730951
    
    
def sum(a, b):
    return a + b

def multiply(a, b):
    return a * b

