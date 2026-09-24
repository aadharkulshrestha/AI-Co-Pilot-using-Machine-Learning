"""
Text-to-Speech (TTS) Abstraction Layer
Supports browser Web SpeechSynthesis API (primary client-side for zero latency and natural voice),
OpenAI TTS API, and local audio streaming.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def synthesize_speech(
    text: str,
    voice: str = "alloy",
    format: str = "mp3"
) -> Optional[bytes]:
    """
    Synthesizes speech on the backend using OpenAI TTS API if API key is provided.
    Returns raw audio bytes or None (allowing frontend Web Speech API to handle speech).
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        logger.debug("No LLM_API_KEY configured for backend TTS. Browser SpeechSynthesis will speak the response.")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Clean text of markdown asterisks and citations for clean audio read-out
        clean_text = text.replace("*", "").replace("#", "")
        # Remove markdown citation tags from audio if too verbose
        import re
        clean_text = re.sub(r"\[.*?\]", "", clean_text)

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=clean_text[:4000]
        )
        return response.content
    except Exception as e:
        logger.warning(f"Backend TTS synthesis failed ({e}). Frontend browser speech will be used.")
        return None
