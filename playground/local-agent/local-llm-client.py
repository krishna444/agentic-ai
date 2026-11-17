import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

@dataclass
class WorkspaceDeps:
    allowed_dir: Path

class ProofreadIssue(BaseModel):
    line_number: int = Field(
        description="Required 1-based line number (1, 2, 3...) where the error occurs"
    )
    error_type: str = Field(description="Must be either 'spelling' or 'grammar'")
    original_text: str = Field(description="Exact incorrect word or phrase from the file")
    suggested_fix: str = Field(description="Corrected word or phrase")
    explanation: str = Field(description="Short explanation of the mistake")

class ProofreadReport(BaseModel):
    source_file: str = Field(description="Name of the file that was proofread")
    corrected_file: str = Field(description="Name of the saved corrected file, or 'none' if no errors")
    total_issues: int = Field(description="Total count of items in the issues list")
    issues: list[ProofreadIssue] = Field(description="List of all grammar and spelling errors found")
    summary: str = Field(description="Executive summary produced by the fast summarizer sub-agent")

provider = OpenAIProvider(
    base_url=os.getenv("OCI_LLM_BASE_URL", "http://10.0.0.13:8081/v1"),
    api_key=os.getenv("OCI_LLM_API_KEY", "not-needed"),
)


qwen_model = OpenAIChatModel(
    model_name=os.getenv("PROOFREADER_MODEL_QWEN", "gemma-4-E2B-it-UD-Q4_K_XL.gguf"),
    provider=provider,
)

gemma_model = OpenAIChatModel(
    model_name=os.getenv("PROOFREADER_MODEL_GEMMA", "gemma-4-E2B-it-UD-Q4_K_XL.gguf"),
    provider=provider,
)

# 2. Fast Specialist Sub-Agent (Gemma 4 E2B)
summarizer_agent = Agent(
    model=gemma_model,
    model_settings={"temperature": 0.2},
    system_prompt=(
        "You are a concise technical editor. Given the corrected document text and the count of fixes, "
        "write a crisp 2-sentence executive summary describing what the document is about and its editorial status."
    ),
)

proofreader = Agent(
    model=qwen_model,
    deps_type=WorkspaceDeps,
    output_type=ProofreadReport,
    model_settings={"temperature": 0.1},
    retries=2,
    system_prompt=(
        "You are a senior copy editor orchestrating a proofreading workflow. Follow these exact steps:\n"
        "1. Call `read_file_with_lines` on the requested filename.\n"
        "2. Identify all spelling and grammar mistakes.\n"
        "3. If mistakes exist, call `write_corrected_file` with the full corrected text (without line numbers).\n"
        "4. Call `delegate_summary_to_gemma` with the final clean text and number of issues found.\n"
        "5. Return the complete `ProofreadReport` using the summary returned by `delegate_summary_to_gemma`."
    ),
)

@proofreader.tool
def read_file_with_lines(ctx: RunContext[WorkspaceDeps], filename: str) -> str:
    """Reads a file inside the configured workspace directory."""
    print(f"[Qwen Tool 1] Reading '{filename}'...")
    target = (ctx.deps.allowed_dir / filename).resolve()
    if not target.is_relative_to(ctx.deps.allowed_dir.resolve()):
        return f"Error: Access denied for '{filename}'."
    if not target.is_file():
        return f"Error: '{filename}' not found."

    content = target.read_text(encoding="utf-8")
    if len(content) > 12000:
        return "Error: File exceeds ~3,000 tokens."

    return "\n".join(f"{i}: {line}" for i, line in enumerate(content.splitlines(), start=1))

@proofreader.tool
def write_corrected_file(
    ctx: RunContext[WorkspaceDeps], filename: str, corrected_content: str
) -> str:
    """Saves the corrected file inside the configured workspace directory."""
    print(f"[Qwen Tool 2] Saving corrected file for '{filename}'...")
    src = (ctx.deps.allowed_dir / filename).resolve()
    if not src.is_relative_to(ctx.deps.allowed_dir.resolve()):
        return "Error: Access denied."

    dest = src.with_name(f"{src.stem}_corrected{src.suffix}")
    dest.write_text(corrected_content.rstrip("\n") + "\n", encoding="utf-8")
    return f"Saved corrected file to {dest.name}"

@proofreader.tool
async def delegate_summary_to_gemma(
    ctx: RunContext[WorkspaceDeps], clean_text: str, issue_count: int
) -> str:
    """Delegates executive summary generation to the fast Gemma 4 E2B sub-agent."""
    print("[Qwen -> Gemma Delegation] Generating executive summary on Gemma 4 E2B...")
    sub_result = await summarizer_agent.run(
        f"Issues fixed: {issue_count}\n\nDocument text:\n{clean_text}",
        usage=ctx.usage,  # Merges Gemma's token & request usage into the parent run!
    )
    return sub_result.output


@proofreader.output_validator
def validate_report(ctx: RunContext[WorkspaceDeps], report: ProofreadReport) -> ProofreadReport:
    """Verifies logical consistency, repairs line numbers deterministically, and checks output file."""
    if report.total_issues != len(report.issues):
        raise ModelRetry(
            f"Mismatch: `total_issues` is {report.total_issues}, but `issues` has {len(report.issues)} items."
        )

    src_path = (ctx.deps.allowed_dir / report.source_file).resolve()
    if src_path.is_file():
        lines = src_path.read_text(encoding="utf-8").splitlines()
        max_lines = len(lines)

        for issue in report.issues:
            # Deterministically resolve or verify line_number using exact substring match
            if issue.line_number < 1 or issue.line_number > max_lines:
                matched_line = next(
                    (idx for idx, text in enumerate(lines, start=1) if issue.original_text in text),
                    None,
                )
                if matched_line is not None:
                    issue.line_number = matched_line
                else:
                    raise ModelRetry(
                        f"Invalid line_number {issue.line_number}. '{report.source_file}' only has {max_lines} lines."
                    )

    if len(report.issues) > 0:
        expected_dest = src_path.with_name(f"{src_path.stem}_corrected{src_path.suffix}")
        if not expected_dest.is_file():
            raise ModelRetry(
                f"You found {len(report.issues)} issues, but '{expected_dest.name}' was not written yet."
            )

    return report

def proofread_file(file_path: str) -> None:
    """Entrypoint function to proofread any user-specified file."""
    target = Path(file_path).expanduser().resolve()
    if not target.is_file():
        print(f"Error: File not found -> {target}")
        sys.exit(1)

    # Sandbox the agent strictly to the parent folder of your chosen file
    deps = WorkspaceDeps(allowed_dir=target.parent)

    print(f"Running proofreader agent on '{target.name}'...")
    start_time = time.perf_counter()
    result = proofreader.run_sync(
        f"Proofread '{target.name}' and fix any errors.", deps=deps
    )
    elapsed = time.perf_counter() - start_time

    print("\n--- Proofread Report ---")
    print(result.output.model_dump_json(indent=2))

    # Fixed deprecation warning: `result.usage` is a property in PydanticAI v2.68+
    usage = result.usage
    print(
        f"\nCompleted in {elapsed:.2f}s | LLM turns: {usage.requests} | Output tokens: {usage.output_tokens}"
    )
def squart(num):
    
    