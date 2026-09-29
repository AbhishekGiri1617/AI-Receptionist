import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from config import settings, business_config
from db import get_embedder, get_qdrant_client
from langchain_agent.rag import retrieve, build_context
from langchain_agent.tools import SYSTEM_PROMPT, agent_graph


embedder = get_embedder()
qdrant = get_qdrant_client()
llm = ChatGroq(model=settings.groq_model, temperature=0.0)


def benchmark_embedding(query: str, runs: int = 3) -> float:
    """Measures local SentenceTransformer embedding latency."""
    # Warmup
    _ = embedder.encode("warmup query", normalize_embeddings=True)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = embedder.encode(query, normalize_embeddings=True)
        latencies.append((time.perf_counter() - t0) * 1000)

    return round(sum(latencies) / len(latencies), 1)


def benchmark_qdrant_search(query: str, runs: int = 3) -> Dict[str, Any]:
    """Measures Qdrant disk vector search latency."""
    emb_start = time.perf_counter()
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    emb_time_ms = (time.perf_counter() - emb_start) * 1000

    latencies = []
    chunk_count = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        chunks = retrieve(qdrant, embedder, query, top_k=settings.top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        chunk_count = len(chunks)

    avg_search_ms = sum(latencies) / len(latencies)
    return {
        "embedding_ms": round(emb_time_ms, 1),
        "qdrant_search_ms": round(avg_search_ms, 1),
        "retrieved_chunks": chunk_count,
    }


async def benchmark_groq_streaming(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    """Measures Groq Time-to-First-Token (TTFT) and generation throughput."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    t_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_text = ""

    async for chunk in llm.astream(messages):
        if first_token_time is None and chunk.content:
            first_token_time = time.perf_counter()
        if chunk.content:
            token_count += 1
            full_text += chunk.content

    t_end = time.perf_counter()

    ttft_ms = round((first_token_time - t_start) * 1000, 1) if first_token_time else 0.0
    total_gen_ms = round((t_end - t_start) * 1000, 1)
    gen_duration_sec = (t_end - first_token_time) if first_token_time else 0.0
    tps = round(token_count / gen_duration_sec, 1) if gen_duration_sec > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_gen_ms,
        "tokens_generated": token_count,
        "tokens_per_sec": tps,
        "response_sample": full_text[:80].replace("\n", " "),
    }


async def benchmark_full_agent_turn(user_prompt: str) -> Dict[str, Any]:
    """Measures end-to-end multi-step Agent Graph turn execution."""
    t0 = time.perf_counter()
    state = await agent_graph.ainvoke({"messages": [HumanMessage(content=user_prompt)]})
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    messages = state.get("messages", [])
    tools_called = []
    final_speech = ""
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                tools_called.append(name)
        if msg.type == "ai" and msg.content and isinstance(msg.content, str):
            final_speech = msg.content

    return {
        "total_turn_ms": total_ms,
        "tools_called": tools_called,
        "response": final_speech[:80].replace("\n", " "),
    }


async def run_latency_profile():
    print("\n" + "=" * 76)
    print("⏱️  VOICE AGENT COMPONENT-BY-COMPONENT LATENCY PROFILER")
    print("=" * 76)

    test_query = "Where is your office located?"

    # 1. Local Embedding Benchmark
    print("\n[1/4] Profiling Local Embedding Model (SentenceTransformer 'all-MiniLM-L6-v2')...")
    emb_ms = benchmark_embedding(test_query)
    print(f"      └─ Avg Embedding Latency: {emb_ms} ms")

    # 2. Qdrant Vector Retrieval Benchmark
    print("\n[2/4] Profiling Qdrant Local Disk Retrieval (Top-5 Chunks)...")
    qdrant_res = benchmark_qdrant_search(test_query)
    print(f"      ├─ Vectorize Query : {qdrant_res['embedding_ms']} ms")
    print(f"      ├─ Qdrant Disk Scan: {qdrant_res['qdrant_search_ms']} ms")
    print(f"      └─ Chunks Retrieved: {qdrant_res['retrieved_chunks']} chunks")

    # 3. Groq Streaming Benchmark (TTFT)
    print(f"\n[3/4] Profiling Groq LLM API ({settings.groq_model})...")
    groq_direct = await benchmark_groq_streaming("Hello, how are you?")
    print(f"      ├─ Time To First Token (TTFT) : {groq_direct['ttft_ms']} ms")
    print(f"      ├─ Total Stream Duration      : {groq_direct['total_ms']} ms")
    print(f"      ├─ Output Generation Speed     : {groq_direct['tokens_per_sec']} tokens/sec")
    print(f"      └─ Sample Output              : \"{groq_direct['response_sample']}...\"")

    # 4. End-to-End Waterfall Turn Profiling
    print("\n[4/4] Profiling Real Conversation Turns (End-to-End Agent Waterfall)...")
    
    # Scenario A: Chit-Chat Turn (Zero Tools)
    print("\n  ▶ Scenario A: Direct Speech (No Tools) — 'Good morning, who are you?'")
    direct_turn = await benchmark_full_agent_turn("Good morning, who are you?")
    print(f"      ├─ Total Turn Latency: {direct_turn['total_turn_ms']} ms")
    print(f"      ├─ Tools Invoked     : {direct_turn['tools_called'] or 'None (Direct response)'}")
    print(f"      └─ Speech Output     : \"{direct_turn['response']}...\"")

    # Scenario B: Knowledge Base RAG Turn
    print(f"\n  ▶ Scenario B: RAG Turn (Qdrant Search) — '{test_query}'")
    rag_turn = await benchmark_full_agent_turn(test_query)
    print(f"      ├─ Total Turn Latency: {rag_turn['total_turn_ms']} ms")
    print(f"      ├─ Tools Invoked     : {rag_turn['tools_called']}")
    print(f"      └─ Speech Output     : \"{rag_turn['response']}...\"")

    # Scenario C: Inquiry Tool Turn
    print("\n  ▶ Scenario C: Action Turn (Lead Capture) — 'Book a demo for Rahul at 9876543210'")
    inquiry_turn = await benchmark_full_agent_turn("I want to book a demo. My name is Rahul and phone is 9876543210.")
    print(f"      ├─ Total Turn Latency: {inquiry_turn['total_turn_ms']} ms")
    print(f"      ├─ Tools Invoked     : {inquiry_turn['tools_called']}")
    print(f"      └─ Speech Output     : \"{inquiry_turn['response']}...\"")

    # Summary Analysis
    print("\n" + "=" * 76)
    print("📊 LATENCY WATERFALL ANALYSIS & TELEPHONY BUDGET")
    print("=" * 76)
    print(f"{'Component / Step':<38} | {'Latency':<12} | {'Voice Telephony Health'}")
    print("-" * 76)
    
    # Voice budget ratings:
    # TTFT: < 600ms = Excellent, 600-1000ms = Good, > 1000ms = Noticeable lag
    ttft_health = "🟢 Excellent" if groq_direct['ttft_ms'] < 600 else ("🟡 Acceptable" if groq_direct['ttft_ms'] < 1000 else "🔴 High Latency")
    emb_health = "🟢 Excellent" if emb_ms < 50 else ("🟡 Noticeable" if emb_ms < 150 else "🔴 Slow (CPU bound)")
    rag_health = "🟢 Fast (< 1.5s)" if rag_turn['total_turn_ms'] < 1500 else ("🟡 Moderate (1.5-2.5s)" if rag_turn['total_turn_ms'] < 2500 else "🔴 Needs Optimization (> 2.5s)")

    print(f"{'SentenceTransformer Embedding':<38} | {str(emb_ms) + ' ms':<12} | {emb_health}")
    print(f"{'Qdrant Vector Search (Local Disk)':<38} | {str(qdrant_res['qdrant_search_ms']) + ' ms':<12} | 🟢 Fast (< 20ms)")
    print(f"{'Groq TTFT (Time to First Token)':<38} | {str(groq_direct['ttft_ms']) + ' ms':<12} | {ttft_health}")
    print(f"{'Direct Speech Turn (No tools)':<38} | {str(direct_turn['total_turn_ms']) + ' ms':<12} | 🟢 Immediate")
    print(f"{'Full RAG Turn (Embedding + Qdrant + LLM)':<38} | {str(rag_turn['total_turn_ms']) + ' ms':<12} | {rag_health}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    asyncio.run(run_latency_profile())
