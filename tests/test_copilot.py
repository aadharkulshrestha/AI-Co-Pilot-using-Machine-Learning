"""
Comprehensive Test Suite for Aviation Conversational Voice AI & FCOM RAG Copilot
Covers all 10 evaluation criteria defined in Part 21.
"""

import os
import unittest
from fastapi.testclient import TestClient

from rag.ingest import (
    create_synthetic_test_manual,
    process_pdf,
    chunk_text_by_paragraphs,
    infer_aircraft,
    detect_section_heading,
    ingest_manuals
)
from rag.vector_store import (
    get_collection,
    add_chunks,
    query_collection,
    get_indexed_documents
)
from rag.retriever import retrieve_manual_context, format_citations_for_prompt
from rag.prompts import generate_rag_answer, generate_extractive_response
from voice.stt import transcribe_audio_bytes
from api.intent import classify_intent, extract_entities, resolve_contextual_query
from api.copilot_api import app


class TestAviationCopilot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure test synthetic manual is generated and indexed."""
        cls.test_dir = os.path.join("data", "manuals", "pdf")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.manual_path = create_synthetic_test_manual(cls.test_dir)
        cls.client = TestClient(app)

    # 1. PDF Ingestion Test
    def test_01_pdf_ingestion(self):
        self.assertTrue(os.path.exists(self.manual_path), "Synthetic test flight manual was not created.")
        chunks = process_pdf(self.manual_path)
        self.assertGreater(len(chunks), 0, "No chunks were extracted from PDF manual.")

    # 2. Chunk Creation Test
    def test_02_chunk_creation(self):
        sample_text = (
            "Line 1: Normal procedures.\n\n"
            "Line 2: Engine start checklist.\n\n"
            "Line 3: Flaps set to takeoff configuration."
        )
        chunks = chunk_text_by_paragraphs(sample_text, target_words=10, overlap_words=2)
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1)

    # 3. Metadata Preservation Test
    def test_03_metadata_preservation(self):
        chunks = process_pdf(self.manual_path)
        first_chunk = chunks[0]
        metadata = first_chunk.get("metadata", {})

        self.assertIn("document", metadata)
        self.assertIn("page", metadata)
        self.assertIn("section", metadata)
        self.assertIn("aircraft", metadata)
        self.assertIn("chunk_id", metadata)
        self.assertEqual(metadata["document"], "TEST_FLIGHT_MANUAL.pdf")
        self.assertEqual(metadata["page"], 1)

    # 4. Vector Database Insertion Test
    def test_04_vector_database_insertion(self):
        res = ingest_manuals(pdf_dir=self.test_dir, force=True)
        self.assertEqual(res["status"], "success")
        coll = get_collection()
        self.assertGreater(coll.count(), 0)
        indexed_docs = get_indexed_documents()
        self.assertIn("TEST_FLIGHT_MANUAL.pdf", indexed_docs)

    # 5. Semantic Retrieval Test
    def test_05_semantic_retrieval(self):
        query = "How should the crew respond to an engine 2 vibration indication?"
        results = retrieve_manual_context(query, top_k=3)
        self.assertGreater(len(results), 0, "Semantic retrieval returned no matches.")
        top_match = results[0]
        self.assertIn("vibration", top_match["text"].lower())
        self.assertIn("document", top_match)
        self.assertIn("page", top_match)
        self.assertIn("score", top_match)

    # 6. Citation Generation Test
    def test_06_citation_generation(self):
        query = "What does the manual say about engine vibration?"
        rag_res = generate_rag_answer(query)
        self.assertIn("answer", rag_res)
        self.assertIn("sources", rag_res)
        self.assertGreater(len(rag_res["sources"]), 0)

        # Check citations format
        first_source = rag_res["sources"][0]
        self.assertEqual(first_source["document"], "TEST_FLIGHT_MANUAL.pdf")
        self.assertIn("Page", rag_res["answer"])

    # 7. Empty Retrieval Handling Test
    def test_07_empty_retrieval(self):
        query = "xyznonexistentqueryaboutnothinginmanuals12345"
        # Using a very high min_score to trigger empty condition or unmatched filter
        results = retrieve_manual_context(query, min_score=0.99)
        self.assertEqual(len(results), 0)

        # Ensure RAG generator handles empty gracefully
        rag_res = generate_rag_answer(query)
        self.assertTrue(
            "no relevant" in rag_res["answer"].lower() or "not found" in rag_res["answer"].lower()
        )

    # 8. Voice Transcription Handling Test
    def test_08_voice_transcription_handling(self):
        # Test STT handler with dummy audio bytes
        dummy_audio = b"RIFF....WAVEfmt ...."
        transcript = transcribe_audio_bytes(dummy_audio, filename="test.wav", provider="auto")
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)

    # 9. Intent Classification Test
    def test_09_intent_classification(self):
        # Manual query
        res1 = classify_intent("What does the FCOM say about hydraulic pressure?")
        self.assertEqual(res1["intent"], "MANUAL_QUERY")

        # Risk analysis query
        res2 = classify_intent("Analyze flight risk of engine vibration during takeoff")
        self.assertEqual(res2["intent"], "RISK_ANALYSIS")

        # General query
        res3 = classify_intent("What is V1 speed?")
        self.assertEqual(res3["intent"], "GENERAL_QUERY")

        # Contextual resolution
        history = [
            {"role": "user", "content": "What does the manual say about engine 2 vibration?"},
            {"role": "assistant", "content": "Retard thrust lever to idle."}
        ]
        follow_up = resolve_contextual_query("What if the indication continues?", history)
        self.assertIn("engine vibration", follow_up.lower())

    # 10. API Response Structure Test
    def test_10_api_response_structure(self):
        # Health endpoint
        health_resp = self.client.get("/api/copilot/health")
        self.assertEqual(health_resp.status_code, 200)
        health_json = health_resp.json()
        self.assertEqual(health_json["status"], "ONLINE")
        self.assertGreaterEqual(health_json["indexed_chunks"], 1)

        # Query endpoint
        payload = {
            "query": "What does the manual say about cabin depressurization?",
            "aircraft": "ALL",
            "conversation_id": "test_session_1"
        }
        query_resp = self.client.post("/api/copilot/query", json=payload)
        self.assertEqual(query_resp.status_code, 200)
        query_json = query_resp.json()

        self.assertIn("answer", query_json)
        self.assertIn("sources", query_json)
        self.assertIn("intent", query_json)
        self.assertIn("aircraft", query_json)
        self.assertIn("topic", query_json)
        self.assertIsInstance(query_json["sources"], list)
        self.assertGreater(len(query_json["sources"]), 0)


if __name__ == "__main__":
    unittest.main()
