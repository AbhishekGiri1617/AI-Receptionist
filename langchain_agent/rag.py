import sys
from typing import Optional

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings, business_config
from db import get_embedder, get_qdrant_client



def retrieve(
    client: Optional[QdrantClient] = None,
    embedder: Optional[SentenceTransformer] = None,
    query: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Embed the query and return the top-k most similar chunks from Qdrant."""
    client = client or get_qdrant_client()
    embedder = embedder or get_embedder()
    try:
        query_vector = embedder.encode(query).tolist()
        hits = client.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]
    except Exception as exc:
        print(f"[WARN] Qdrant retrieval failed: {exc}")
        return []


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk.get('chunk_text', chunk.get('content', ''))}")
    return "\n\n---\n\n".join(parts)


def rag(query: str, top_k: Optional[int] = None, client=None, embedder=None):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format as context
      3. Send context + query to Groq LLM

    Returns:
        tuple: (answer_text, context_used)
    """
    top_k = top_k or settings.top_k
    embedder = embedder or get_embedder()
    client = client or get_qdrant_client()


    company_name = business_config.get("company_name", "Our Company")

    # Step 1 — Retrieve
    chunks = retrieve(client, embedder, query, top_k=top_k)
    if not chunks:
        return (
            f"I couldn't find specific information about that in {company_name}'s documentation. "
            "Would you like me to note down your inquiry or transfer you to a team member?",
            "",
        )

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate answer
    system_prompt = (
        f"You are a helpful customer support assistant for {company_name}.\n"
        "Answer the user's question using ONLY the context provided below.\n"
        "If the context does not contain enough information, state that clearly and do not hallucinate.\n"
        "Keep responses brief, polite, and conversational for voice phone calls "
        "(1 to 2 sentences max). Do not use bullet points or tables."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.2,
    )
    messages = [
        ("system", system_prompt),
        ("user", user_message),
    ]

    response = llm.invoke(messages)
    return response.content, context


def main():
    """CLI interface for testing RAG independently."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    company_name = business_config.get("company_name", "Our Company")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = f"What services or products does {company_name} offer?"

    print(f"Question: {query}\n")
    print("Generating answer...\n")

    answer, context = rag(query)

    print(f"ANSWER:\n{answer}")
    print(f"\n{'='*60}")
    print(f"\nSOURCES:\n{context}")


if __name__ == "__main__":
    main()

