from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import settings, business_config


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}


def load_document(source: Path):
    """
    Parse a document using Docling or plain text loader.
    """
    print(f"  Parsing: {source.name}")
    if source.suffix.lower() == ".txt":
        text = source.read_text(encoding="utf-8")
        return text

    converter = DocumentConverter()
    result = converter.convert(str(source))
    return result.document


def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dictionary.
    Preserves headings as a breadcrumb trail prepended to content.
    """
    headings = getattr(getattr(doc_chunk, "meta", None), "headings", None) or []
    content = doc_chunk.text.strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings": headings,
        "content": content,
        "chunk_text": chunk_text,
    }


def chunk_plain_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Simple chunker for raw text files."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        content = " ".join(chunk_words).strip()
        if content:
            chunks.append({
                "headings": [],
                "content": content,
                "chunk_text": content,
            })
    return chunks


def ingest_documents(docs_dir: str) -> None:
    """
    Ingestion pipeline:
    1. Scan docs_dir for supported documents (PDF, DOCX, TXT, MD)
    2. Parse and chunk each document
    3. Generate embeddings
    4. Index in Qdrant collection
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        print(f"Created docs directory at '{docs_dir}'. Add your business documents and run again.")
        return

    doc_files = [
        f for f in sorted(docs_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not doc_files:
        print(f"No documents found in '{docs_dir}' with extensions: {SUPPORTED_EXTENSIONS}")
        print("Please add your PDFs, DOCX, Markdown, or TXT documents and try again.")
        return

    print(f"\nFound {len(doc_files)} document(s) in '{docs_dir}':")
    for f in doc_files:
        print(f"  - {f.name}")

    # Step 1: Parse and chunk all documents
    print("\n--- Step 1: Parsing & Chunking ---")
    all_chunks = []

    for file_path in doc_files:
        try:
            if file_path.suffix.lower() == ".txt":
                text = load_document(file_path)
                file_chunks = chunk_plain_text(text, chunk_size=settings.chunk_size)
            else:
                doc = load_document(file_path)
                chunker = HierarchicalChunker(max_characters=settings.chunk_size)
                file_chunks = [convert_chunk(c) for c in chunker.chunk(doc)]

            print(f"  [OK] {file_path.name} -> {len(file_chunks)} chunks")
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"  [FAIL] {file_path.name}: {e}")
            continue

    if not all_chunks:
        print("No chunks created. Check your documents and try again.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 2: Generate embeddings
    print("\n--- Step 2: Generating Embeddings ---")
    embedder = SentenceTransformer(settings.embedding_model)
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
    print(f"[OK] Embedding shape: {embeddings.shape}")

    # Step 3: Index in Qdrant
    print(f"\n--- Step 3: Indexing in Qdrant collection '{settings.collection_name}' ---")
    client = QdrantClient(path=settings.qdrant_path)
    dim = embedder.get_embedding_dimension()

    # Drop previous collection if it exists to ensure no stale vectors hinder new ones
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "headings": chunk["headings"],
                "content": chunk["content"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    client.upsert(collection_name=settings.collection_name, points=points, wait=True)

    info = client.get_collection(settings.collection_name)
    print(f"[OK] Indexed {info.points_count} points in collection '{settings.collection_name}'")

    client.close()
    print("\n[DONE] Ingestion complete. Your generic voice agent is ready to query these documents.")


def reset_collection():
    """Wipe the current collection to clear all vectors."""
    try:
        client = QdrantClient(path=settings.qdrant_path)
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)
            print(f"[OK] Deleted collection '{settings.collection_name}'. Vector database is reset.")
        else:
            print(f"[INFO] Collection '{settings.collection_name}' does not exist.")
        client.close()
    except Exception as e:
        print(f"[ERROR] Could not reset collection: {e}")


if __name__ == "__main__":
    import sys
    if "--clean" in sys.argv or "--reset" in sys.argv:
        reset_collection()
        sys.exit(0)

    company = business_config.get("company_name", "Knowledge Base")
    print(f"=== {company} — Document Ingestion Pipeline ===")
    print(f"  Docs folder:  {settings.docs_dir}")
    print(f"  Embedding:    {settings.embedding_model}")
    print(f"  Qdrant path:  {settings.qdrant_path}")
    print(f"  Collection:   {settings.collection_name}")
    ingest_documents(settings.docs_dir)

