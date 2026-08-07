"""
rag_engine.py

Core RAG engine for the Zenvyrolabs Multimodal RAG Assistant.

Sprint 2 (LLM configuration + Bug 1) and Sprint 3 (Bug 2) are both in this
revision. Ingestion, process_and_store_document(), is unchanged from
Sprint 2 and is not repeated in detail here.

  1. LLM configuration. Gemini Flash is the primary model (cloud, high
     quality); Ollama is a local fallback used when no Gemini key is set
     or Gemini fails to initialize (quota, outage, offline dev). Selection
     happens once at import time.

  2. Bug 1 - Page-Aware Retrieval ("Vector Blindness"). Every chunk's
     page_content is prefixed with a "[Page N]" marker before it is
     embedded, and the same number is kept in metadata (`page_number`).
     A question naming an explicit page is retrieved with an exact
     metadata filter instead of relying on semantic similarity alone.

  3. Bug 2 - Adaptive Retrieval & Hierarchical Summarization ("Myopic
     Context"). query_rag_system() no longer always pulls a fixed k=30.
     Each question is classified into one of three retrieval tiers:
       - local  - an ordinary question (includes the Bug 1 page-lookup
         path). Small k, one LLM call.
       - scoped - a chapter/section-level request. Moderate k via
         similarity search, then map-reduce summarization.
       - global - a whole-book request. Every indexed chunk for the
         active book_type is fetched (page-ordered, capped and evenly
         sampled if the book is very large), then summarized
         hierarchically in small batches so no single LLM call ever
         sees more than a few thousand characters, independent of book
         length. This is what keeps it viable on modest hardware and
         keeps API usage bounded rather than growing with the book.
     All three tiers preserve page citations, and retrieval is now
     scoped to the active book_type (previously, Sprint 2's local path
     searched across every indexed book).

Out of scope here, tracked for later sprints per PROJECT_DOCUMENTATION.md:
  - Bug 3, conversation memory (Sprint 4)
  - Bug 4, manga OCR / Vision pipeline (Sprint 5)
"""

import os
import re
import logging
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag_engine")

# --------------------------------------------------------------------------
# Vector DB and embeddings
# --------------------------------------------------------------------------

DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

# --------------------------------------------------------------------------
# LLM selection: Gemini Flash primary, Ollama local fallback
# --------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _build_llm():
    """
    Selects an LLM backend at import time.

    Order of preference:
      1. Gemini Flash (cloud) - used whenever GEMINI_API_KEY is set.
      2. Ollama (local) - offline / quota-exhaustion fallback.
      3. None - the app stays importable and reports a clear error at
         query time instead of crashing on startup when neither is
         configured yet.

    Both provider imports are done lazily inside this function so the
    app can still start if only one of the two provider packages is
    installed.
    """
    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.2,
            )
            logger.info("LLM ready: Gemini (%s)", GEMINI_MODEL)
            return model
        except Exception as exc:
            logger.warning("Gemini init failed (%s). Falling back to Ollama.", exc)

    try:
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
        logger.info("LLM ready: Ollama (%s)", OLLAMA_MODEL)
        return model
    except Exception as exc:
        logger.warning("Ollama fallback unavailable (%s). No LLM configured.", exc)
        return None


llm = _build_llm()

# --------------------------------------------------------------------------
# Ingestion - page-aware chunking (Bug 1, write side)
# --------------------------------------------------------------------------


def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Loads a PDF, splits it into page-aware chunks, and stores it in ChromaDB.

    Every chunk's page_content is prefixed with "[Page N]" (N is 1-indexed,
    matching what a reader sees in a PDF viewer) before embedding, so the
    embedding model has an actual token to match against page-specific
    questions. The same number is kept in metadata as `page_number` for
    exact filtering on the query side.
    """
    if book_type == "manga":
        # Manga/OCR pipeline is Sprint 5 (Bug 4). For now this still runs
        # the text pipeline below, which will extract little or nothing
        # from an image-only PDF - that gap is intentional until Sprint 5.
        logger.info("book_type=manga: OCR pipeline not implemented yet (Sprint 5); attempting text extraction.")

    loader = PyMuPDFLoader(file_path, mode="page")
    docs = loader.load()

    if not docs:
        raise ValueError(f"No content could be extracted from {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    source_name = os.path.basename(file_path)

    for chunk in chunks:
        # PyMuPDFLoader's "page" metadata is 0-indexed; PDF viewers count
        # from 1, and that's the number a user will actually ask about.
        pdf_page_index = chunk.metadata.get("page", 0)
        page_number = pdf_page_index + 1

        chunk.metadata["page_number"] = page_number
        chunk.metadata["book_type"] = book_type
        chunk.metadata["source"] = source_name

        chunk.page_content = f"[Page {page_number}]\n{chunk.page_content}"

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    logger.info("Stored %d page-aware chunks from %s", len(chunks), source_name)
    return "Success"


# --------------------------------------------------------------------------
# Query handling - adaptive retrieval (Bug 2) + grounded citations (Bug 1)
# --------------------------------------------------------------------------

_PAGE_QUESTION_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)

# Retrieval breadth per tier. Small for an ordinary question; wider for a
# named chapter/section; global fetches the whole book instead of using k
# at all (see _fetch_all_chunks_for_book).
RETRIEVAL_K_LOCAL = int(os.environ.get("RETRIEVAL_K_LOCAL", 6))
RETRIEVAL_K_SCOPED = int(os.environ.get("RETRIEVAL_K_SCOPED", 40))

# Hierarchical summarization tuning (scoped + global tiers).
MAX_CHUNKS_GLOBAL = int(os.environ.get("MAX_CHUNKS_GLOBAL", 250))
GLOBAL_FETCH_SAFETY_LIMIT = int(os.environ.get("GLOBAL_FETCH_SAFETY_LIMIT", 3000))
MAP_BATCH_CHAR_BUDGET = int(os.environ.get("MAP_BATCH_CHAR_BUDGET", 6000))
REDUCE_BATCH_SIZE = int(os.environ.get("REDUCE_BATCH_SIZE", 10))

_PROMPT_TEMPLATE = """You are a helpful Interactive Study Tutor. Answer the question based ONLY on the following context from the textbook.
Every excerpt below starts with a [Page N] marker showing where it came from. When your answer uses an excerpt, cite its page like "(Page 12)".
If the context does not contain the answer, say "I cannot find the answer to this in the textbook." Never guess or invent a page number.

Context:
{context}

Question: {question}

Answer:"""

_MAP_PROMPT_TEMPLATE = """You are helping summarize part of a book to answer a reader's question.

Excerpt (tagged with [Page N] markers):
{excerpt}

Question the reader ultimately wants answered: {question}

In 3-5 sentences, note anything in this excerpt relevant to that question. Keep any page numbers you mention. If nothing here is relevant, say "Nothing relevant on these pages." Do not answer the question yet - these are just notes."""

_CONDENSE_PROMPT_TEMPLATE = """Combine the notes below into one denser set of notes about {scope}, for this question: {question}

Notes:
{notes}

Merge overlapping points and keep every page number mentioned. Do not answer the question yet - this is still notes, a short paragraph."""

_REDUCE_PROMPT_TEMPLATE = """You are combining notes taken from different parts of a book to answer a reader's question about {scope}.

Notes (each from a different part of the book):
{notes}

Question: {question}

Write the final answer now, grounded only in the notes above. Cite pages using the format (Page N) wherever the notes give you one. If the notes don't cover something, leave it out rather than guessing."""

_prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
_map_prompt = ChatPromptTemplate.from_template(_MAP_PROMPT_TEMPLATE)
_condense_prompt = ChatPromptTemplate.from_template(_CONDENSE_PROMPT_TEMPLATE)
_reduce_prompt = ChatPromptTemplate.from_template(_REDUCE_PROMPT_TEMPLATE)

# Chapter/section number patterns intentionally overlap in what they can
# match (e.g. "Part 3" vs "Section 3") - order doesn't matter because they
# all resolve to the same "scoped" retrieval strategy. See _classify_query.
_GLOBAL_PATTERNS = [
    re.compile(r"\b(entire|whole|full)\s+book\b", re.IGNORECASE),
    re.compile(r"\bbook\s+(as a whole|overall|in general)\b", re.IGNORECASE),
    re.compile(r"\b(summar\w*|overview)\b.*\bbook\b", re.IGNORECASE),
    re.compile(r"\bbook\b.*\b(summar\w*|overview)\b", re.IGNORECASE),
    re.compile(r"\bbest\b.*\bin the book\b", re.IGNORECASE),
    re.compile(r"\bthroughout the book\b", re.IGNORECASE),
]
_SCOPED_PATTERNS = [
    re.compile(r"\bchapter\s+[ivxlcdm\d]+\b", re.IGNORECASE),
    re.compile(r"\bsection\s+\d+(\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bpart\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bsummar\w*\s+(this\s+)?(chapter|section|part)\b", re.IGNORECASE),
]


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _extract_requested_page(user_message: str) -> Optional[int]:
    """Returns the page number a question explicitly names, if any."""
    match = _PAGE_QUESTION_PATTERN.search(user_message)
    return int(match.group(1)) if match else None


def _classify_query(user_message: str) -> Tuple[str, Optional[str]]:
    """
    Buckets a question into a retrieval tier - "global", "scoped", or
    "local" - plus a human-readable label for scoped/global questions
    (used later to keep the summarization prompt referring to the right
    scope by name).

    This is regex/keyword based on purpose: running every question
    through an LLM just to classify it would cost more, in latency and
    tokens, than most of the retrieval it's choosing between. A future
    Query Analysis node (PROJECT_DOCUMENTATION.md, Part 6) can replace
    this without changing anything downstream of the classification.

    "chapter" and "section" currently resolve to the same "scoped"
    strategy - ingestion only records page_number (Bug 1), not chapter
    or section boundaries, so there is no structural way to treat them
    differently yet. That's a natural extension of process_and_store_
    document() later, not something this function can fix on its own.
    """
    for pattern in _GLOBAL_PATTERNS:
        if pattern.search(user_message):
            return "global", "the whole book"

    for pattern in _SCOPED_PATTERNS:
        match = pattern.search(user_message)
        if match:
            return "scoped", match.group(0)

    return "local", None


def _build_filter(**conditions) -> Optional[dict]:
    """
    Builds a Chroma `where` filter from keyword conditions, dropping any
    that are None. Two or more conditions are combined with an explicit
    "$and" - current Chroma versions require this for multi-key filters,
    a bare multi-key dict is no longer reliably treated as an implicit AND.
    """
    clauses = [{key: value} for key, value in conditions.items() if value is not None]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _stride_sample(items: list, target_count: int) -> list:
    """
    Evenly samples `items` down to `target_count`, preserving order.
    Used instead of a plain truncation so a capped global summary still
    covers the whole book (beginning, middle, and end) rather than just
    whichever pages happened to come back first.
    """
    if target_count <= 0 or len(items) <= target_count:
        return items
    step = len(items) / target_count
    return [items[int(i * step)] for i in range(target_count)]


def _fetch_all_chunks_for_book(vector_db: Chroma, book_type: str) -> List[Document]:
    """
    Pulls every indexed chunk for the active book_type, in page order.

    Unlike similarity_search, this ignores relevance ranking entirely -
    a whole-book summary needs coverage, not the chunks nearest to some
    query embedding. Chroma's .get() has no ORDER BY, so page ordering
    happens client-side; GLOBAL_FETCH_SAFETY_LIMIT bounds how much a
    single fetch can pull back regardless of book size.
    """
    raw = vector_db.get(
        where=_build_filter(book_type=book_type),
        limit=GLOBAL_FETCH_SAFETY_LIMIT,
        include=["documents", "metadatas"],
    )
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]
    docs.sort(key=lambda d: d.metadata.get("page_number", 0))
    return docs


def _batch_by_char_budget(chunks: List[Document], budget: int) -> List[List[Document]]:
    """
    Greedily groups already page-ordered chunks into batches whose
    combined page_content stays under `budget` characters, so each MAP
    call below gets a small, bounded amount of context no matter how
    many chunks came in - this is what avoids ever sending "hundreds of
    pages" to the LLM in one request.
    """
    batches: List[List[Document]] = []
    current: List[Document] = []
    current_len = 0
    for chunk in chunks:
        chunk_len = len(chunk.page_content)
        if current and current_len + chunk_len > budget:
            batches.append(current)
            current, current_len = [], 0
        current.append(chunk)
        current_len += chunk_len
    if current:
        batches.append(current)
    return batches


def _map_notes(chunks: List[Document], question: str) -> List[str]:
    """MAP step: one small, bounded LLM call per batch, extracting only
    what's relevant to the question rather than summarizing indiscriminately."""
    notes = []
    chain = _map_prompt | llm | StrOutputParser()
    for batch in _batch_by_char_budget(chunks, MAP_BATCH_CHAR_BUDGET):
        try:
            notes.append(chain.invoke({"excerpt": _format_docs(batch), "question": question}))
        except Exception as exc:
            # A transient failure on one batch (timeout, rate limit)
            # shouldn't sink the whole summary - skip it and keep going.
            logger.warning("Map step failed for one batch (%s); skipping it.", exc)
    return notes


def _reduce_notes(notes: List[str], question: str, scope_label: str) -> str:
    """
    REDUCE step. If there are more notes than fit comfortably in one
    call, they're condensed in groups first (repeatedly, if needed) -
    a real hierarchical reduce, not a single call hoping everything fits.
    """
    if not notes:
        return "I cannot find enough content to answer that."

    condense_chain = _condense_prompt | llm | StrOutputParser()
    while len(notes) > REDUCE_BATCH_SIZE:
        condensed = []
        for i in range(0, len(notes), REDUCE_BATCH_SIZE):
            group = notes[i : i + REDUCE_BATCH_SIZE]
            try:
                condensed.append(
                    condense_chain.invoke(
                        {"notes": "\n\n".join(group), "question": question, "scope": scope_label}
                    )
                )
            except Exception as exc:
                logger.warning("Condense step failed for one group (%s); skipping it.", exc)
        # If every group in this pass failed, fall back to a plain
        # truncation rather than looping forever on an empty result.
        notes = condensed or notes[:REDUCE_BATCH_SIZE]

    reduce_chain = _reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({"notes": "\n\n".join(notes), "question": question, "scope": scope_label})


def query_rag_system(user_message: str, book_type: str = "coding") -> str:
    """
    Classifies the question, retrieves at a tier-appropriate breadth, and
    generates a cited answer. Replaces Sprint 2's fixed k=30 (Bug 2).

      local  - an ordinary question, including an explicit page lookup.
               Small k, one LLM call - same shape as Sprint 2.
      scoped - a chapter/section-level request. Moderate k via similarity
               search against the question, then map-reduce.
      global - a whole-book request. Every chunk for book_type is fetched,
               capped and evenly sampled if very large, then summarized
               hierarchically so no single call sees the whole book at once.

    All tiers are now scoped to the active book_type - Sprint 2's local
    path previously searched across every indexed book regardless of mode.
    """
    if llm is None:
        return "ERROR: No LLM is configured. Set GEMINI_API_KEY, or run Ollama locally."

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    scope, label = _classify_query(user_message)

    if scope == "global":
        chunks = _fetch_all_chunks_for_book(vector_db, book_type)
        if not chunks:
            return f"No {book_type} document is indexed yet. Please upload one first."
        chunks = _stride_sample(chunks, MAX_CHUNKS_GLOBAL)
        return _reduce_notes(_map_notes(chunks, user_message), user_message, label)

    if scope == "scoped":
        chunks = vector_db.similarity_search(
            user_message, k=RETRIEVAL_K_SCOPED, filter=_build_filter(book_type=book_type)
        )
        if not chunks:
            return f"I cannot find {label} in the {book_type} document."
        chunks.sort(key=lambda d: d.metadata.get("page_number", 0))
        return _reduce_notes(_map_notes(chunks, user_message), user_message, label)

    # scope == "local"
    requested_page = _extract_requested_page(user_message)
    if requested_page is not None:
        results = vector_db.similarity_search(
            user_message, k=8, filter=_build_filter(page_number=requested_page, book_type=book_type)
        )
        if not results:
            # Nothing indexed under that exact page (book too short, or
            # the 0-index -> +1 mapping is off for this PDF). Fall back
            # rather than answering from an empty context.
            results = vector_db.similarity_search(
                user_message, k=RETRIEVAL_K_LOCAL, filter=_build_filter(book_type=book_type)
            )
    else:
        results = vector_db.similarity_search(
            user_message, k=RETRIEVAL_K_LOCAL, filter=_build_filter(book_type=book_type)
        )

    context = _format_docs(results)
    chain = _prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": user_message})
