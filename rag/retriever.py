"""
Retriever Module for Aviation Flight Manual RAG
Provides semantic search with cosine similarity scoring, metadata filtering (aircraft, document, section),
and structured citation assembly.
"""

import logging
from typing import List, Dict, Any, Optional
from rag.vector_store import query_collection, get_collection

logger = logging.getLogger(__name__)


def retrieve_manual_context(
    query: str,
    top_k: int = 5,
    aircraft: Optional[str] = None,
    document: Optional[str] = None,
    section: Optional[str] = None,
    min_score: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Retrieves the most semantically relevant manual sections for a pilot's query.
    Applies optional metadata filtering for aircraft type, specific manual document, or section.
    """
    filters = {}
    if aircraft and aircraft.upper() not in ["ALL", "ANY", "GENERAL", "UNSPECIFIED"]:
        filters["aircraft"] = aircraft.upper()
    if document and document.upper() not in ["ALL", "ANY"]:
        filters["document"] = document
    if section and section.upper() not in ["ALL", "ANY"]:
        filters["section"] = section

    results = query_collection(
        query_text=query,
        top_k=top_k,
        filter_metadata=filters if filters else None
    )

    # Filter out results below relevance threshold
    filtered_results = [r for r in results if r.get("score", 0.0) >= min_score]

    # If strict filter returned no results, retry without aircraft filter to prevent pilot lockout
    if not filtered_results and filters.get("aircraft"):
        logger.info(f"No results found for aircraft '{aircraft}'. Retrying across all indexed fleet manuals.")
        fallback_results = query_collection(query_text=query, top_k=top_k)
        filtered_results = [r for r in fallback_results if r.get("score", 0.0) >= min_score]

    return filtered_results


def format_citations_for_prompt(sources: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved manual passages into an authoritative context block for the LLM.
    Strictly numbers each passage with [Doc, Page, Section] so the LLM cannot hallucinate sources.
    """
    if not sources:
        return "NO RELEVANT MANUAL EXCERPTS FOUND IN INDEXED DOCUMENTATION."

    context_blocks = []
    for idx, s in enumerate(sources, 1):
        doc = s.get("document", "Manual")
        page = s.get("page", 1)
        sec = s.get("section", "General")
        aircraft = s.get("aircraft", "")
        text = s.get("text", "").strip()

        header = f"[Source {idx} | Document: {doc} | Page: {page} | Section: {sec} | Aircraft: {aircraft}]"
        context_blocks.append(f"{header}\n{text}\n")

    return "\n---\n".join(context_blocks)
