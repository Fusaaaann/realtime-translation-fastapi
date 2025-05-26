# test_local_audio_integration.py

import numpy as np
import time
import base64
import asyncio
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

# Import the necessary items from your app.
from server import (
    app,
    published_data_store,
    admin_config,
    client    # This is the ElevenLabs client in your app.
)


# --- Dummy Implementations for External API Calls ---

def dummy_speech_to_text_convert(model_id, file):
    """
    Dummy replacement for the ElevenLabs client's speech_to_text.convert method.
    Returns a fixed transcription.
    """
    return {"text": "dummy transcription"}

def dummy_text_to_speech(text: str, target_lang: str) -> bytes:
    """
    Dummy text-to-speech that returns fixed dummy audio bytes.
    """
    return b"dummy tts audio"

# Dummy OpenAI client replacement
class DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    class responses:
        @staticmethod
        def create(model, input):
            # Return a dummy response object with an output_text attribute.
            class DummyResponse:
                output_text = "dummy translation"
            return DummyResponse()

# --- Fixture to Prepare a Dummy Local Audio File ---

@pytest.fixture
def setup_local_audio(monkeypatch, request):
    """
    This fixture overrides the get_local_audio_source function so that instead of reading
    from a file, it simulates capturing audio from a microphone. It returns dummy audio
    data (a silent PCM signal) and a sample rate.
    """
    use_dummy_mic = request.config.getoption("--use-dummy-mic")
    if use_dummy_mic:
        return
    def dummy_get_local_audio_source(record_seconds: int = 5, sample_rate: int = 44100) -> Tuple[bytes, int]:
        # Simulate a microphone recording: a silent audio of given duration.
        frames = record_seconds * sample_rate
        # Create a numpy array of zeros (int16 PCM), simulating silence.
        dummy_audio_np = np.zeros((frames, 1), dtype=np.int16)
        # Convert the numpy array to bytes.
        dummy_audio_bytes = dummy_audio_np.tobytes()
        return dummy_audio_bytes, sample_rate

    # Replace the original get_local_audio_source with our dummy version.
    monkeypatch.setattr("server.get_local_audio_source", dummy_get_local_audio_source)

# --- Monkeypatching & Configuration Fixture ---

@pytest.fixture
def configure_testing(monkeypatch):
    """
    This fixture patches external dependencies and modifies configurations for faster testing.
    """
    # Patch ElevenLabs client's speech-to-text convert method.
    monkeypatch.setattr(client.speech_to_text, "convert", dummy_speech_to_text_convert)
    
    # Patch text_to_speech function in the application module to use the dummy implementation.
    monkeypatch.setattr("server.text_to_speech", dummy_text_to_speech)
    
    # Patch OpenAI usage: Override the OpenAI class so that every instantiation
    # returns our dummy, which in turn always returns "dummy translation".
    import openai
    monkeypatch.setattr(openai, "OpenAI", DummyOpenAI)
    
    # Set a short polling interval so that the background task runs quickly.
    admin_config["poll_interval"] = 1
    
    # Optionally, clear the published data store.
    async def clear_published_data_store():
        published_data_store.speaking.clear()
        published_data_store.translated.clear()
        published_data_store.audio.clear()
    asyncio.run(clear_published_data_store())

def test_local_audio_source_integration(setup_local_audio, configure_testing):
    """
    Integration test for local audio source mode using dummy external dependencies.
    The background task should process the dummy local audio file and publish:
      - A dummy transcription.
      - For every target language (other than the source language),
        a dummy translation and dummy TTS audio.
    """
    with TestClient(app) as client_api:
        # Wait enough time for the background task to process the dummy audio.
        time.sleep(3)

        # Check the speaking captions endpoint.
        response = client_api.get("/captions/speaking")
        assert response.status_code == 200
        speaking_data = response.json()
        assert any("dummy transcription" in item["text"] for item in speaking_data), \
            "Dummy transcription was not published."

        # For each target language (as per admin_config), check translated data and TTS audio.
        for lang in admin_config.get("target_languages", []):
            if lang.upper() == admin_config.get("default_source_language", "ENGLISH").upper():
                continue

            # Check translated captions.
            response = client_api.get(f"/captions/translated?lang={lang}")
            assert response.status_code == 200
            translated_data = response.json()
            assert any("dummy translation" in item["text"] for item in translated_data), \
                f"Dummy translation not published for language {lang}."

            # Check translated audio TTS.
            response = client_api.get(f"/audio/translated-voice?lang={lang}")
            assert response.status_code == 200
            audio_data = response.json()
            for entry in audio_data:
                decoded_audio = base64.b64decode(entry["audio"])
                assert decoded_audio == b"dummy tts audio", \
                    f"Dummy TTS audio mismatch for language {lang}."

