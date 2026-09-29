import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import pytest
from langchain_core.messages import HumanMessage
from langchain_agent.tools import agent_graph
from evals.run_evals import evaluate_voice_compliance, extract_agent_execution


DATASET_FILE = Path(__file__).resolve().parent / "dataset.json"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    ALL_TEST_CASES = json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ALL_TEST_CASES, ids=[c["id"] for c in ALL_TEST_CASES])
async def test_agent_behavior(case):
    """Parametrized pytest suite checking tool choice, voice constraints, and grounding."""
    user_input = case["input"]
    expected_tool = case.get("expected_tool")
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])

    # 1. Run agent graph
    state = await agent_graph.ainvoke({"messages": [HumanMessage(content=user_input)]})
    execution = extract_agent_execution(state.get("messages", []))
    tools_called = execution["tools_called"]
    final_speech = execution["final_speech"]

    # 2. Check tool selection
    if expected_tool:
        assert expected_tool in tools_called, (
            f"Expected tool '{expected_tool}' was not called. Tools invoked: {tools_called}"
        )
    else:
        assert len(tools_called) == 0, (
            f"Expected no tools for '{case['id']}', but tools were called: {tools_called}"
        )

    # 3. Check voice constraints (no markdown, concise sentences, no raw URLs)
    compliance = evaluate_voice_compliance(final_speech)
    assert not compliance["has_markdown"], (
        f"Spoken text contains markdown formatting: {compliance['markdown_violations']}\nText: {final_speech}"
    )
    assert not compliance["has_urls"], (
        f"Spoken text contains raw URLs which cannot be read by TTS.\nText: {final_speech}"
    )
    assert compliance["is_concise"], (
        f"Response too long for voice telephony ({compliance['sentence_count']} sentences, max 3).\nText: {final_speech}"
    )

    # 4. Check keyword assertions
    lower_speech = final_speech.lower()
    for kw in must_contain:
        assert kw.lower() in lower_speech, (
            f"Missing required keyword '{kw}' in response: {final_speech}"
        )
    for kw in must_not_contain:
        assert kw.lower() not in lower_speech, (
            f"Found forbidden keyword '{kw}' in response: {final_speech}"
        )
