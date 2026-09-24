"""
Aviation FCOM/QRH Prompt Templates and RAG Answer Generator
Implements safety boundaries, exact citation injection, and dual-mode generation
(LLM-based when API key is set, deterministic extractive RAG when offline/no key).
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from rag.retriever import retrieve_manual_context, format_citations_for_prompt

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an aviation information retrieval assistant.
Answer using only the provided retrieved flight-manual context.
Do not invent procedures, limitations, numbers, callouts, checklist steps, or aircraft-specific information.
If the retrieved context does not contain enough information, explicitly say that the required information was not found in the provided manuals.
Every factual statement derived from a manual must have a citation.
Citations must include:
[Document, Page X, Section Y]
Do not present yourself as the aircraft's official onboard system.
For operational emergencies, clearly indicate that the official aircraft documentation and applicable airline/operator procedures take precedence.

SAFETY NOTICE:
Decision-support only. Verify against the applicable aircraft manual, QRH/FCOM, and operator procedures."""


def generate_llm_response(
    query: str,
    retrieved_sources: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Calls an OpenAI-compatible LLM endpoint using retrieved manual context.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()

    if not api_key:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        context_text = format_citations_for_prompt(retrieved_sources)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Add recent conversation memory (last 4 turns)
        if conversation_history:
            for turn in conversation_history[-4:]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        user_content = f"""RETRIEVED FLIGHT MANUAL EXCERPTS:
{context_text}

PILOT QUERY:
{query}

Provide a concise, operational answer citing [Document, Page X, Section Y] for all facts. If not found in the context, explicitly declare that no relevant information was found."""

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=600
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM API call failed ({e}). Reverting to deterministic extractive response.")
        return ""


def generate_extractive_response(query: str, sources: List[Dict[str, Any]]) -> str:
    """
    High-fidelity offline / zero-API extractive RAG synthesizer.
    Pulls exact procedures, limits, and steps from retrieved manual chunks,
    guaranteeing zero hallucinations and strictly cited facts.
    """
    if not sources:
        return (
            "No relevant information was found in the indexed manuals.\n\n"
            "Please check that the relevant FCOM or QRH PDF has been placed in "
            "`data/manuals/pdf/` and indexed via `python -m rag.ingest`."
        )

    # Clean query terms
    q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    stop_words = {"what", "does", "manual", "about", "show", "tell", "pilot", "procedure", "checklist", "with", "this", "the", "for", "and", "how", "should", "crew"}
    keywords = q_words - stop_words

    # Verify if any concept from query is present in retrieved chunks
    has_keyword_match = False
    for src in sources:
        combined_text = (src.get("text", "") + " " + src.get("section", "")).lower()
        if any(k in combined_text for k in keywords):
            has_keyword_match = True
            break

    top_score = sources[0].get("score", 0.0) if sources else 0.0
    if keywords and not has_keyword_match and top_score < 0.70:
        return (
            "No relevant section was found in the indexed manuals.\n\n"
            "Please verify that the applicable aircraft manual or QRH is indexed in `data/manuals/pdf/`."
        )

    matched_points = []
    citations_used = []

    for src in sources:
        doc = src.get("document", "Manual")
        page = src.get("page", 1)
        sec = src.get("section", "General")
        cite_tag = f"[{doc}, Page {page}, Section {sec}]"
        citations_used.append(cite_tag)

        text = src.get("text", "")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:
            line_lower = line.lower()
            # If line matches keywords or is part of numbered action steps
            if any(k in line_lower for k in keywords) or re.match(r"^(\d+\.|\-|\*|CONDITION:|PILOT ACTION|CREW ACTION)", line):
                matched_points.append(f"{line} {cite_tag}")

    if not matched_points:
        top_src = sources[0]
        # If top source score is below relevance threshold and no keywords matched, reject
        if top_src.get("score", 0.0) < 0.62:
            return "No relevant information was found in the indexed manuals."

        # Fallback to top source content if similarity is very high
        text_preview = "\n".join([f"> {l}" for l in top_src.get("text", "").split("\n")[:8] if l.strip()])
        doc = top_src.get("document", "Manual")
        page = top_src.get("page", 1)
        sec = top_src.get("section", "General")
        return (
            f"According to the indexed flight manual documentation [{doc}, Page {page}, Section {sec}]:\n\n"
            f"{text_preview}\n\n"
            f"⚠️ *Decision-support only. Verify against official aircraft documentation, QRH/FCOM, and applicable operator procedures.*"
        )

    # Format into cohesive response
    primary_cite = citations_used[0] if citations_used else ""
    response_lines = [
        f"According to the retrieved manual context {primary_cite}:",
        ""
    ]

    # Limit to top 8 key points to avoid overwhelm in cockpit
    for pt in matched_points[:8]:
        if pt.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "-", "*")):
            response_lines.append(f"{pt}")
        else:
            response_lines.append(f"- {pt}")

    response_lines.append("")
    response_lines.append("⚠️ *Decision-support only. Verify against official aircraft documentation, QRH/FCOM, and applicable operator procedures.*")

    return "\n".join(response_lines)


def generate_rag_answer(
    query: str,
    aircraft: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Main RAG generation coordinator:
    1. Retrieves relevant manual passages.
    2. Attempts LLM response if API key is configured.
    3. Falls back to deterministic extractive synthesizer if no API key or on error.
    4. Formats clean, structured sources for UI rendering.
    """
    sources = retrieve_manual_context(query, top_k=top_k, aircraft=aircraft)

    if not sources:
        return {
            "answer": "No relevant section was found in the indexed manuals.",
            "sources": [],
            "query": query,
            "aircraft": aircraft or "ALL"
        }

    # Try LLM generation first
    llm_output = generate_llm_response(query, sources, conversation_history)
    answer = llm_output if llm_output else generate_extractive_response(query, sources)

    # Format structured sources for frontend cards
    formatted_sources = []
    for s in sources:
        formatted_sources.append({
            "document": s.get("document", "Manual"),
            "page": s.get("page", 1),
            "section": s.get("section", "General"),
            "aircraft": s.get("aircraft", "Fleet"),
            "score": s.get("score", 0.0),
            "text": s.get("text", "").strip()[:400] + ("..." if len(s.get("text", "")) > 400 else "")
        })

    return {
        "answer": answer,
        "sources": formatted_sources,
        "query": query,
        "aircraft": aircraft or "ALL"
    }
