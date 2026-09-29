import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_agent.tools import agent_graph

DATASET_FILE = Path(__file__).resolve().parent / "dataset.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_voice_compliance(text: str) -> Dict[str, Any]:
    """Evaluates strict constraints needed for realistic Text-to-Speech synthesis."""
    # 1. Markdown check (markdown syntax sounds unnatural or breaks TTS)
    markdown_patterns = [
        r"\*\*.*?\*\*",       # bold
        r"(?<!\w)\*.*?\*(?!\w)", # italic
        r"`.*?`",             # code / backticks
        r"^#+\s",             # headings
        r"\[.*?\]\(.*?\)",     # links
        r"^\s*[-*+]\s+",      # bullet lists
        r"^\s*\d+\.\s+",      # numbered lists
        r"\|.*?\|",           # tables
    ]
    markdown_violations = []
    for pattern in markdown_patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            markdown_violations.append(pattern)

    # 2. Raw URL check
    has_urls = bool(re.search(r"(https?://\S+|www\.\S+)", text, re.IGNORECASE))

    # 3. Sentence count (Voice agents need concise 1-2 sentence replies)
    # Split by ., !, ? followed by space or end of string
    raw_sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]
    sentence_count = len(raw_sentences)
    is_concise = 1 <= sentence_count <= 3

    passed = (len(markdown_violations) == 0) and (not has_urls) and is_concise

    return {
        "passed": passed,
        "sentence_count": sentence_count,
        "is_concise": is_concise,
        "has_markdown": len(markdown_violations) > 0,
        "has_urls": has_urls,
        "markdown_violations": markdown_violations,
    }


def extract_agent_execution(messages: List[Any]) -> Dict[str, Any]:
    """Extracts tool calls and the final speech message from LangChain state messages."""
    tools_called = []
    final_speech = ""

    for msg in messages:
        # Check tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                if name:
                    tools_called.append(name)
        
        # In LangGraph create_agent, the last AI message with string content is the final output
        if msg.type == "ai" and msg.content and isinstance(msg.content, str):
            final_speech = msg.content

    return {
        "tools_called": tools_called,
        "final_speech": final_speech.strip(),
    }


async def run_single_eval(case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single test case against agent_graph."""
    user_input = case["input"]
    expected_tool = case.get("expected_tool")
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])

    start_time = time.perf_counter()
    try:
        state = await agent_graph.ainvoke({"messages": [HumanMessage(content=user_input)]})
        elapsed_sec = round(time.perf_counter() - start_time, 2)
        execution = extract_agent_execution(state.get("messages", []))
    except Exception as exc:
        elapsed_sec = round(time.perf_counter() - start_time, 2)
        return {
            "id": case["id"],
            "category": case.get("category", "general"),
            "input": user_input,
            "error": str(exc),
            "passed": False,
            "elapsed_sec": elapsed_sec,
        }

    tools_called = execution["tools_called"]
    final_speech = execution["final_speech"]

    # 1. Tool check
    if expected_tool:
        tool_passed = expected_tool in tools_called
    else:
        # If no tool expected, check that none were called
        tool_passed = len(tools_called) == 0

    # 2. Voice compliance check
    voice_eval = evaluate_voice_compliance(final_speech)

    # 3. Content assertions
    lower_speech = final_speech.lower()
    missing_keywords = [kw for kw in must_contain if kw.lower() not in lower_speech]
    forbidden_found = [kw for kw in must_not_contain if kw.lower() in lower_speech]
    content_passed = (len(missing_keywords) == 0) and (len(forbidden_found) == 0)

    # Overall pass criteria: tool correct + voice safe + content correct
    overall_passed = tool_passed and voice_eval["passed"] and content_passed

    return {
        "id": case["id"],
        "category": case.get("category", "general"),
        "input": user_input,
        "expected_tool": expected_tool,
        "tools_called": tools_called,
        "tool_passed": tool_passed,
        "final_speech": final_speech,
        "voice_eval": voice_eval,
        "missing_keywords": missing_keywords,
        "forbidden_found": forbidden_found,
        "content_passed": content_passed,
        "elapsed_sec": elapsed_sec,
        "passed": overall_passed,
    }


async def run_all_evals(dataset_path: Path = DATASET_FILE):
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("\n" + "=" * 70)
    print(f"🚀 RUNNING VOICE AGENT EVALS ({len(cases)} test cases)")
    print("=" * 70)

    results = []
    passed_count = 0
    total_latency = 0.0

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] Evaluating '{case['id']}'...", end=" ", flush=True)
        res = await run_single_eval(case)
        results.append(res)
        total_latency += res.get("elapsed_sec", 0.0)

        if res.get("passed"):
            passed_count += 1
            print(f"✅ PASSED ({res['elapsed_sec']}s)")
        else:
            print(f"❌ FAILED ({res.get('elapsed_sec', 0)}s)")
            # Print brief diagnostic
            if not res.get("tool_passed"):
                print(f"    └─ Tool mismatch: Expected '{res.get('expected_tool')}', Got: {res.get('tools_called')}")
            if res.get("voice_eval") and not res["voice_eval"]["passed"]:
                print(f"    └─ Voice issue: Markdown={res['voice_eval']['has_markdown']}, Sentences={res['voice_eval']['sentence_count']}")
            if res.get("missing_keywords"):
                print(f"    └─ Missing keywords: {res['missing_keywords']}")
            if res.get("forbidden_found"):
                print(f"    └─ Forbidden keywords found: {res['forbidden_found']}")
            if res.get("error"):
                print(f"    └─ Error: {res['error']}")

    # Summary Statistics
    pass_rate = (passed_count / len(cases)) * 100 if cases else 0
    avg_latency = round(total_latency / len(cases), 2) if cases else 0

    print("\n" + "=" * 70)
    print("📊 EVALUATION SCORECARD")
    print("=" * 70)
    print(f"Total Cases     : {len(cases)}")
    print(f"Passed          : {passed_count}")
    print(f"Failed          : {len(cases) - passed_count}")
    print(f"Pass Rate       : {pass_rate:.1f}%")
    print(f"Average Latency : {avg_latency}s")
    print("=" * 70)

    # Save detailed JSON report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = REPORTS_DIR / f"report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_cases": len(cases),
            "passed": passed_count,
            "pass_rate": pass_rate,
            "average_latency_sec": avg_latency,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"[REPORT SAVED] {report_file}\n")


if __name__ == "__main__":
    asyncio.run(run_all_evals())
