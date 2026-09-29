
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings

MASTER_TRANSCRIPTS = Path("data") / "transcripts" / "conversations.jsonl"

AUDIT_PROMPT = """You are an expert QA auditor for a customer support voice AI receptionist.
Review the following telephone dialogue between the Caller and the AI Assistant (Abhishek).

Evaluate the call on these criteria and return ONLY a valid JSON object:
{
  "goal_resolved": boolean (did the caller get their question answered or inquiry recorded?),
  "lead_captured": boolean (if caller asked for a demo/callback, was their info collected?),
  "tone_and_politeness": integer from 1 to 5 (5 being exceptionally warm and professional),
  "voice_constraints_respected": boolean (were replies concise and natural without robotic formatting?),
  "escalated": boolean (was human escalation triggered or needed?),
  "hallucination_detected": boolean (did the assistant invent non-existent company facts?),
  "summary": string (one sentence summary of the call outcome)
}
"""

def audit_transcripts(limit: int = 10):
    if not MASTER_TRANSCRIPTS.exists():
        print(f"[INFO] No transcripts found yet at: {MASTER_TRANSCRIPTS}")
        return

    llm = ChatGroq(model=settings.groq_model, temperature=0.0)

    with open(MASTER_TRANSCRIPTS, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("[INFO] Transcripts file is empty.")
        return

    recent_calls = lines[-limit:]
    print("\n" + "=" * 75)
    print(f"📋 AUDITING LAST {len(recent_calls)} PRODUCTION CALL TRANSCRIPTS")
    print("=" * 75)

    audits = []
    for call_str in recent_calls:
        try:
            call = json.loads(call_str)
        except Exception:
            continue

        call_id = call.get("call_id", "unknown")
        turns = call.get("turns", [])
        if not turns:
            continue

        dialogue_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])

        try:
            response = llm.invoke([
                SystemMessage(content=AUDIT_PROMPT),
                HumanMessage(content=f"Call ID: {call_id}\n\nDialogue:\n{dialogue_text}")
            ])
            # Parse json from response
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.split("\n", 1)[1].rsplit("\n", 1)[0]
            audit_result = json.loads(cleaned_content)
        except Exception as e:
            audit_result = {"error": str(e), "summary": "Failed to audit call."}

        audits.append((call_id, audit_result))

        print(f"\n📞 Call ID: {call_id}")
        if "error" in audit_result:
            print(f"   ⚠️ Error: {audit_result['error']}")
        else:
            print(f"   • Goal Resolved : {'✅ Yes' if audit_result.get('goal_resolved') else '❌ No'}")
            print(f"   • Lead Captured : {'✅ Yes' if audit_result.get('lead_captured') else '⚪ N/A or No'}")
            print(f"   • Tone / Quality: {audit_result.get('tone_and_politeness', 'N/A')}/5")
            print(f"   • Voice Safe    : {'✅ Yes' if audit_result.get('voice_constraints_respected') else '❌ No'}")
            print(f"   • Summary       : {audit_result.get('summary', '')}")

    print("\n" + "=" * 75)
    print(f"Audit completed for {len(audits)} calls.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    audit_transcripts()
