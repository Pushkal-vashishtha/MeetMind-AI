"""
core/rag_engine.py  —  Upgraded with hybrid search (BM25 + dense)
and optional cross-encoder reranking.
"""

from __future__ import annotations
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from core.vector_store import build_vector_store, load_vector_store, get_retriever


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3):
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )


# ── Document formatter ────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        chunk_id = doc.metadata.get("chunk_index", i)
        parts.append(f"[Excerpt {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Hybrid Retriever (BM25 + Dense) ──────────────────────────────────────────

def build_hybrid_retriever(docs, vector_store, k: int = 6):
    """
    Combines keyword (BM25) and semantic (Chroma) retrieval.
    BM25 catches exact keyword matches; dense catches semantic similarity.
    Result is deduplicated by LangChain's EnsembleRetriever.
    """
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    dense_retriever = get_retriever(vector_store, k=k)

    # weights: tune these — 0.4 BM25 + 0.6 dense works well for meeting transcripts
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    return hybrid


# ── Optional reranker ─────────────────────────────────────────────────────────

def rerank_docs(docs, query: str, top_k: int = 4):
    """
    Cross-encoder reranking using a local model (no API cost).
    Falls back gracefully if sentence-transformers isn't installed.
    Install: pip install sentence-transformers
    """
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except ImportError:
        # Reranker not installed — return top_k docs as-is
        return docs[:top_k]


# ── RAG prompt ────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are an expert meeting assistant with access to the meeting transcript.

RULES:
1. Answer ONLY from the provided context excerpts.
2. If the answer is not in the context, say: "I couldn't find that in the transcript."
3. If quoting someone, say "According to the transcript, …"
4. Be concise and precise. Use bullet points for multi-part answers.
5. If asked about action items, decisions, or questions — be comprehensive.

Context from the meeting transcript:
{context}"""


# ── Main builder ──────────────────────────────────────────────────────────────

def build_rag_chain(transcript: str, use_hybrid: bool = True, use_reranker: bool = False):
    """
    Build a production-grade RAG chain.

    Args:
        transcript:   Full meeting transcript string.
        use_hybrid:   If True, uses BM25 + dense retrieval (recommended).
        use_reranker: If True, adds cross-encoder reranking (slower but more accurate).

    Returns:
        A LangChain runnable. Call with: chain.invoke("your question")
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    # Build docs for BM25
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_text(transcript)
    docs     = [Document(page_content=c, metadata={"chunk_index": i}) for i, c in enumerate(chunks)]

    vector_store = build_vector_store(transcript)
    llm = get_llm()

    if use_hybrid:
        retriever = build_hybrid_retriever(docs, vector_store, k=6)
    else:
        retriever = get_retriever(vector_store, k=4)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", "{question}"),
    ])

    def retrieve_and_format(question: str) -> str:
        retrieved = retriever.invoke(question)
        if use_reranker:
            retrieved = rerank_docs(retrieved, question, top_k=4)
        return format_docs(retrieved)

    rag_chain = (
        {
            "context":  RunnableLambda(retrieve_and_format),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── Cross-session loader ──────────────────────────────────────────────────────

def load_rag_chain(use_hybrid: bool = False) -> object:
    """
    Load a previously persisted Chroma index without re-processing the transcript.
    Use this to restore a session without re-running the pipeline.
    """
    vector_store = load_vector_store()
    retriever    = get_retriever(vector_store, k=4)
    llm          = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", "{question}"),
    ])

    return (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# ── Ask helper ────────────────────────────────────────────────────────────────

def ask_question(rag_chain, question: str, verbose: bool = False) -> str:
    if verbose:
        print(f"[RAG] Q: {question}")
    answer = rag_chain.invoke(question)
    if verbose:
        print(f"[RAG] A: {answer[:200]}…")
    return answer