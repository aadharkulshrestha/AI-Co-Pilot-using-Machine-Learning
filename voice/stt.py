"""
Speech-to-Text (STT) Abstraction Layer
Supports browser-side Web Speech API (primary for ultra-low latency)
and backend audio transcription fallback (Whisper / OpenAI Whisper API).
"""

import os
import io
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def transcribe_audio_bytes(
    audio_data: bytes,
    filename: str = "audio.wav",
    provider: Optional[str] = None
) -> str:
    """
    Transcribes audio data bytes to text using available backend engines:
    1. OpenAI Whisper API (if LLM_API_KEY is present)
    2. Local faster-whisper / whisper (if installed)
    3. Graceful fallback message instructing use of browser Web Speech API
    """
    provider = (provider or os.getenv("STT_PROVIDER", "auto")).lower()
    api_key = os.getenv("LLM_API_KEY", "")

    # Attempt 1: OpenAI Whisper API if key present
    if (provider in ["openai", "whisper_api", "auto"]) and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            audio_file = io.BytesIO(audio_data)
            audio_file.name = filename
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="Aviation cockpit radio query, Boeing, Airbus, FCOM, QRH, flight controls, autopilot, engine."
            )
            return transcript.text.strip()
        except Exception as e:
            logger.warning(f"OpenAI Whisper API transcription failed: {e}")

    # Attempt 2: Local Whisper if installed
    if provider in ["local_whisper", "whisper", "auto"]:
        try:
            import whisper
            model_name = os.getenv("WHISPER_MODEL", "base")
            model = whisper.load_model(model_name)
            # Save temporary file for local whisper
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            try:
                res = model.transcribe(tmp_path)
                return res.get("text", "").strip()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except ImportError:
            logger.debug("Local whisper is not installed. Relying on browser Web Speech API.")
        except Exception as e:
            logger.warning(f"Local whisper transcription failed: {e}")

    return "Audio received. For lowest latency cockpit interaction, browser Web Speech API is active."
