"""
Embedding Provider Module for Aviation FCOM/QRH Manual RAG
Supports local ONNX MiniLM (via ChromaDB Default), Sentence-Transformers, and OpenAI embeddings.
Configurable via environment variables without hardcoded keys.
"""

import os
import logging
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global singleton cache for embedding function
_EMBEDDING_FUNCTION = None

def get_embedding_function():
    """
    Returns an embedding function instance compatible with ChromaDB.
    Defaults to all-MiniLM-L6-v2 (ONNX-accelerated, zero-API-cost local embedding).
    """
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is not None:
        return _EMBEDDING_FUNCTION

    provider = os.getenv("EMBEDDING_PROVIDER", "chroma_default").lower()
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    api_key = os.getenv("LLM_API_KEY", "")

    # Provider option 1: OpenAI embedding if explicitly configured and key present
    if (provider == "openai" or "text-embedding" in model_name) and api_key:
        try:
            from chromadb.utils import embedding_functions
            logger.info(f"Using OpenAI Embedding Function: {model_name}")
            _EMBEDDING_FUNCTION = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name
            )
            return _EMBEDDING_FUNCTION
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings ({e}), falling back to local ONNX MiniLM.")

    # Provider option 2: Sentence-Transformers if explicitly chosen and installed
    if provider == "sentence_transformers":
        try:
            from chromadb.utils import embedding_functions
            logger.info(f"Using SentenceTransformer Embedding Function: {model_name}")
            _EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            return _EMBEDDING_FUNCTION
        except Exception as e:
            logger.warning(f"SentenceTransformers unavailable ({e}), falling back to Chroma default ONNX.")

    # Provider option 3 (Default, fast, self-contained local ONNX all-MiniLM-L6-v2):
    try:
        from chromadb.utils import embedding_functions
        logger.info("Initializing ChromaDB Default ONNX Embedding Function (all-MiniLM-L6-v2)...")
        _EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()
        return _EMBEDDING_FUNCTION
    except Exception as e:
        logger.error(f"Error loading ChromaDB default embedding function: {e}")
        raise e


def generate_embedding(text: str) -> List[float]:
    """Generates embedding vector for a single string."""
    fn = get_embedding_function()
    results = fn([text])
    return results[0]


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a batch of strings."""
    fn = get_embedding_function()
    return fn(texts)
