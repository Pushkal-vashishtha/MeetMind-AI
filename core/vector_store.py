"""
core/vector_store.py  —  Upgraded with per-session collections,
metadata enrichment, and similarity threshold filtering.
"""

from __future__ import annotations
import os
import uuid
import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR       = "vector_db"
COLLECTION_NAME  = "meeting_transcript"   # default — overridden per session
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},   # cosine similarity
    )


# ── Text splitter ─────────────────────────────────────────────────────────────

def split_transcript(transcript: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # prefer sentence boundaries
    )
    return splitter.split_text(transcript)


# ── Build vector store ────────────────────────────────────────────────────────

def build_vector_store(
    transcript: str,
    session_id: str | None = None,
    collection_name: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Embed the transcript and store in Chroma.

    Args:
        transcript:       Full transcript string.
        session_id:       Optional UUID for this session. Auto-generated if None.
        collection_name:  Override the Chroma collection name.
        chunk_size:       Token window per chunk.
        chunk_overlap:    Overlap between chunks for context continuity.

    Returns:
        Chroma vector store instance.
    """
    print("[VectorStore] Building index…")

    sid = session_id or str(uuid.uuid4())[:8]
    col = collection_name or f"session_{sid}"

    chunks = split_transcript(transcript, chunk_size, chunk_overlap)
    print(f"[VectorStore] {len(chunks)} chunks created.")

    # Enrich each chunk with metadata
    now = datetime.datetime.utcnow().isoformat()
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i,
                "session_id":  sid,
                "created_at":  now,
                "char_count":  len(chunk),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings    = get_embeddings()
    vector_store  = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=col,
        persist_directory=CHROMA_DIR,
    )
    print(f"[VectorStore] Indexed {len(docs)} chunks → collection '{col}'")
    return vector_store


# ── Load existing store ───────────────────────────────────────────────────────

def load_vector_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ── Retriever factory ─────────────────────────────────────────────────────────

def get_retriever(
    vector_store: Chroma,
    k: int = 4,
    search_type: str = "similarity",
    score_threshold: float | None = None,
):
    """
    Get a retriever from the vector store.

    Args:
        k:                Number of chunks to retrieve.
        search_type:      "similarity" | "mmr" (Max Marginal Relevance for diversity)
        score_threshold:  Optional minimum similarity score (0–1).
    """
    search_kwargs: dict = {"k": k}

    if search_type == "mmr":
        # MMR reduces redundancy among retrieved chunks
        search_kwargs["fetch_k"] = k * 3
        search_kwargs["lambda_mult"] = 0.7   # 0 = diversity, 1 = relevance

    if score_threshold is not None:
        search_type = "similarity_score_threshold"
        search_kwargs["score_threshold"] = score_threshold

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


# ── List all sessions ─────────────────────────────────────────────────────────

def list_sessions() -> list[str]:
    """Return all collection names stored in the Chroma DB."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return [c.name for c in client.list_collections()]