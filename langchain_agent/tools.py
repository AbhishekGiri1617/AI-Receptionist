import json
from datetime import datetime
from pathlib import Path
from typing import List

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings, business_config
from db import get_embedder, get_qdrant_client
from langchain_agent.rag import retrieve, build_context



# ---------------------------------------------------------------------------
# Shared singletons — initialized once at import time
# ---------------------------------------------------------------------------

embedder = get_embedder()
qdrant = get_qdrant_client()


# Verify Qdrant collection status gracefully
try:
    _count = qdrant.get_collection(settings.collection_name).points_count
    print(f"[OK] Connected to Qdrant collection '{settings.collection_name}' ({_count} chunks indexed)")
except Exception as exc:
    print(
        f"[INFO] Collection '{settings.collection_name}' not found yet at '{settings.qdrant_path}'. "
        "Run `python ingest.py` after adding documents to your docs/ folder."
    )


# ---------------------------------------------------------------------------
# Storage helpers for inquiries and escalations
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INQUIRIES_FILE = DATA_DIR / "inquiries.jsonl"
ESCALATIONS_FILE = DATA_DIR / "escalations.jsonl"


# ---------------------------------------------------------------------------
# Generic LangChain Tools
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search company documentation, FAQs, policies, services, and product details.
    Use this whenever the caller asks any question about what the company does,
    its offerings, pricing, operating hours, policies, or procedures."""
    chunks = retrieve(qdrant, embedder, query, top_k=settings.top_k)
    if not chunks:
        company = business_config.get("company_name", "the company")
        return f"No relevant information found in {company}'s documentation for: '{query}'."
    return build_context(chunks)


@tool
def capture_inquiry(customer_name: str, contact_info: str, inquiry_details: str) -> str:
    """Record a customer lead, contact inquiry, or request for a callback.
    Provide the customer's name, their phone number or email, and the details of their request."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_name": customer_name.strip(),
        "contact_info": contact_info.strip(),
        "inquiry_details": inquiry_details.strip(),
    }

    try:
        with open(INQUIRIES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to save inquiry: {e}")

    print(f"[INQUIRY CAPTURED] Name: {customer_name} | Contact: {contact_info} | Note: {inquiry_details}")

    return (
        f"Thank you {customer_name}. I have recorded your inquiry, and our team will get in touch with you at {contact_info}."
    )


@tool
def escalate_to_human(reason: str, caller_name: str = "Customer") -> str:
    """Escalate to a human support representative or log an urgent escalation ticket.
    Use this when the caller explicitly asks for a human agent, has a complex dispute,
    or has an issue that cannot be answered using the knowledge base."""
    ticket_id = "TICKET-" + str(abs(hash(reason + datetime.now().isoformat())) % 100000).zfill(5)
    timestamp = datetime.now().isoformat()

    entry = {
        "ticket_id": ticket_id,
        "timestamp": timestamp,
        "caller_name": caller_name,
        "reason": reason,
    }

    try:
        with open(ESCALATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to save escalation: {e}")

    print(f"[ESCALATION] ID={ticket_id} time={timestamp} reason={reason!r}")

    return (
        f"I have opened escalation ticket {ticket_id}. "
        "A team member has been notified and will follow up with you as soon as possible."
    )


# ---------------------------------------------------------------------------
# Dynamic System Prompt Generator
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Constructs a clean system prompt from business_config."""
    company = business_config.get("company_name", "Our Company")
    agent = business_config.get("agent_name", "Assistant")
    persona_cfg = business_config.get("persona", {})

    fmt_vars = {"company_name": company, "agent_name": agent}

    raw_role = persona_cfg.get(
        "role",
        f"You are {agent}, a friendly and professional voice assistant representing {company}."
    )
    try:
        role = raw_role.format(**fmt_vars)
    except Exception:
        role = raw_role

    raw_tone = persona_cfg.get(
        "tone",
        "Warm, concise, and helpful. Keep responses to 1-2 sentences. Avoid markdown or bullet lists."
    )
    try:
        tone = raw_tone.format(**fmt_vars)
    except Exception:
        tone = raw_tone

    raw_guidelines = persona_cfg.get("guidelines", [
        "Use search_knowledge_base to answer questions about products, services, or policies.",
        "If you do not find the information, do not make it up. Offer to record an inquiry or escalate.",
    ])

    guidelines = []
    for g in raw_guidelines:
        try:
            guidelines.append(g.format(**fmt_vars))
        except Exception:
            guidelines.append(g)

    guidelines_formatted = "\n".join(f"- {g}" for g in guidelines)

    prompt = (
        f"COMPANY: {company}\n"
        f"AGENT NAME: {agent}\n\n"
        f"ROLE:\n{role}\n\n"
        f"TONE & STYLE:\n{tone}\n\n"
        f"GUIDELINES:\n"
        f"{guidelines_formatted}\n\n"
        f"VOICE CONSTRAINTS:\n"
        f"- Always keep your responses conversational, natural, and concise (1-2 sentences maximum).\n"
        f"- Never output bullet points, asterisks, tables, or URLs, as they cannot be spoken clearly.\n"
        f"- If you cannot find the answer in the knowledge base, politely state so and offer to take an inquiry or escalate."
    )
    return prompt



SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# Build the LangChain agent graph
# ---------------------------------------------------------------------------

enabled_tools_cfg = business_config.get("tools", {})
ALL_TOOLS = []

if enabled_tools_cfg.get("enable_knowledge_base", True):
    ALL_TOOLS.append(search_knowledge_base)

if enabled_tools_cfg.get("enable_inquiry_capture", True):
    ALL_TOOLS.append(capture_inquiry)

if enabled_tools_cfg.get("enable_escalation", True):
    ALL_TOOLS.append(escalate_to_human)

llm = ChatGroq(model=settings.groq_model, temperature=0.0)

agent_graph = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)

# Backward-compatibility alias
lauki_agent = agent_graph
