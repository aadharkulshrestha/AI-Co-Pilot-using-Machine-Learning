"""
Vector Store Management using ChromaDB for Aviation Manuals (FCOM/QRH)
Maintains persistent vector collection 'aviation_manuals' with full metadata and query capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from rag.embeddings import get_embedding_function

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", os.path.join("data", "manuals", "processed", "chroma"))
COLLECTION_NAME = "aviation_manuals"

_CHROMA_CLIENT = None
_COLLECTION = None


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """Returns persistent ChromaDB client instance."""
    global _CHROMA_CLIENT
    directory = persist_directory or DEFAULT_PERSIST_DIR
    os.makedirs(directory, exist_ok=True)
    if _CHROMA_CLIENT is None:
        logger.info(f"Initializing ChromaDB PersistentClient at: {directory}")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=directory)
    return _CHROMA_CLIENT


def get_collection(persist_directory: Optional[str] = None, reset: bool = False):
    """
    Retrieves or creates the 'aviation_manuals' collection with our embedding function.
    """
    global _COLLECTION
    client = get_chroma_client(persist_directory)
    embedding_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}' for rebuild.")
        except Exception as e:
            logger.debug(f"Collection deletion check: {e}")
        _COLLECTION = None

    if _COLLECTION is None or reset:
        _COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to collection '{COLLECTION_NAME}'. Count: {_COLLECTION.count()}")

    return _COLLECTION


def get_indexed_documents(persist_directory: Optional[str] = None) -> List[str]:
    """Returns a unique list of document filenames already indexed in the vector store."""
    try:
        coll = get_collection(persist_directory)
        count = coll.count()
        if count == 0:
            return []
        # Get all metadata (only metadata, without loading embeddings/documents)
        data = coll.get(include=["metadatas"])
        docs = set()
        for meta in data.get("metadatas", []):
            if meta and "document" in meta:
                docs.add(meta["document"])
        return sorted(list(docs))
    except Exception as e:
        logger.error(f"Error querying indexed documents: {e}")
        return []


def is_document_indexed(document_name: str, persist_directory: Optional[str] = None) -> bool:
    """Checks whether a given manual has already been ingested into the collection."""
    indexed = get_indexed_documents(persist_directory)
    return document_name in indexed


def delete_document(document_name: str, persist_directory: Optional[str] = None) -> int:
    """Deletes all chunks associated with a specific document."""
    try:
        coll = get_collection(persist_directory)
        # Find ids for this document
        data = coll.get(where={"document": document_name}, include=["metadatas"])
        ids = data.get("ids", [])
        if ids:
            coll.delete(ids=ids)
            logger.info(f"Removed {len(ids)} chunks for document '{document_name}'.")
            return len(ids)
        return 0
    except Exception as e:
        logger.error(f"Error deleting document '{document_name}': {e}")
        return 0


def add_chunks(chunks: List[Dict[str, Any]], persist_directory: Optional[str] = None) -> int:
    """
    Adds a list of pre-processed manual chunks into the ChromaDB collection.
    Each chunk dict must contain:
    - chunk_id (str)
    - text (str)
    - metadata: { document, page, section, aircraft, ... }
    """
    if not chunks:
        return 0

    coll = get_collection(persist_directory)

    # Batch insertions to prevent payload overflow
    BATCH_SIZE = 250
    total_added = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Ensure all metadata values are strings, ints, floats, or bools for ChromaDB
        sanitized_metadatas = []
        for m in metadatas:
            clean_m = {}
            for k, v in m.items():
                if v is None:
                    clean_m[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            sanitized_metadatas.append(clean_m)

        coll.upsert(
            ids=ids,
            documents=documents,
            metadatas=sanitized_metadatas
        )
        total_added += len(batch)

    logger.info(f"Successfully upserted {total_added} chunks into '{COLLECTION_NAME}'. New total: {coll.count()}")
    return total_added


def query_collection(
    query_text: str,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs vector similarity search against the collection.
    Returns list of matched chunks with metadata and cosine similarity score.
    """
    coll = get_collection(persist_directory)
    if coll.count() == 0:
        return []

    where_clause = None
    if filter_metadata:
        # Filter empty or 'All' values
        active_filters = {k: v for k, v in filter_metadata.items() if v and str(v).lower() != "all"}
        if len(active_filters) == 1:
            k, v = list(active_filters.items())[0]
            where_clause = {k: v}
        elif len(active_filters) > 1:
            where_clause = {"$and": [{k: v} for k, v in active_filters.items()]}

    results = coll.query(
        query_texts=[query_text],
        n_results=min(top_k, coll.count()),
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

        for doc_text, meta, dist, cid in zip(docs, metas, distances, ids):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity (0 is identical, 2 is opposite)
            # Similarity score ~ 1 - (dist / 2) or max(0, 1 - dist)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            formatted_results.append({
                "chunk_id": cid,
                "text": doc_text,
                "document": meta.get("document", "Unknown Document"),
                "page": meta.get("page", 1),
                "section": meta.get("section", "General"),
                "aircraft": meta.get("aircraft", "Unspecified"),
                "score": round(similarity, 4),
                "metadata": meta
            })

    return formatted_results
