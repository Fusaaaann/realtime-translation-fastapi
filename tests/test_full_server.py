# test_full_server.py
import json
import time
import base64
import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

# Import mock functions with correct names
from mock_audio import initialize_audio_recorder, get_audio_frames, continuous_recorder

# Import the necessary items from your app.
from server import app, published_data_store, admin_config, client

logger = logging.getLogger(__name__)

# --- Dummy Implementations for External API Calls ---


def dummy_speech_to_text_convert(model_id, file):
    """
    Dummy replacement for the ElevenLabs client's speech_to_text.convert method.
    Returns a fixed transcription.
    """

    class DummyResult:
        text = "dummy transcription"

    return DummyResult()


def dummy_text_to_speech(text: str, target_lang: str) -> bytes:
    """
    Dummy text-to-speech that returns fixed dummy audio bytes.
    """
    return b"dummy tts audio"


# Dummy OpenAI client replacement
class DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                class DummyChoice:
                    class Message:
                        content = "dummy translation"

                    message = Message()

                class DummyResponse:
                    choices = [DummyChoice()]

                return DummyResponse()


@pytest.fixture
def setup_local_audio(monkeypatch, request):
    """
    This fixture overrides the audio recording functions to use mock implementations
    that generate audio from a test file instead of recording from a real microphone.
    """
    use_real_mic = request.config.getoption("--use-real-mic", default=False)
    if use_real_mic:
        return

    # Replace the audio recording functions with our mock versions
    monkeypatch.setattr("server.initialize_audio_recorder", initialize_audio_recorder)
    monkeypatch.setattr("server.get_audio_frames", get_audio_frames)
    # monkeypatch.setattr("server.ContinuousAudioRecorder", MockContinuousAudioRecorder)

    # Clean up when the test is done
    yield

    if continuous_recorder is not None:
        continuous_recorder.stop()


@pytest.fixture
def configure_testing(monkeypatch, request):
    """
    This fixture patches external dependencies and modifies configurations for faster testing.
    """
    use_real_api = request.config.getoption("--use-real-api", default=False)
    if use_real_api:
        return

    # Patch ElevenLabs client's speech-to-text convert method.
    monkeypatch.setattr(client.speech_to_text, "convert", dummy_speech_to_text_convert)

    # Patch text_to_speech function in the application module to use the dummy implementation.
    monkeypatch.setattr("server.text_to_speech", dummy_text_to_speech)

    # Patch OpenAI usage
    import openai

    monkeypatch.setattr(openai, "OpenAI", DummyOpenAI)

    # Set a short polling interval so that the background task runs quickly.
    admin_config["poll_interval"] = 1

    # Clear the published data store.
    async def clear_published_data_store():
        published_data_store.speaking.clear()
        published_data_store.translated.clear()
        published_data_store.audio.clear()

    asyncio.run(clear_published_data_store())


def test_local_audio_source_integration(setup_local_audio, configure_testing):
    """
    Integration test for local audio source mode using dummy external dependencies.
    """
    with TestClient(app) as client_api:
        # Wait enough time for the background task to process the dummy audio.
        time.sleep(5)

        # Check the speaking captions endpoint.
        response = client_api.get("/captions/speaking")
        assert response.status_code == 200

        # Parse chunked response
        content = response.content.decode().strip()
        chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
        speaking_data = chunks[0] if chunks else []

        assert any("dummy transcription" in item["text"] for item in speaking_data), "Dummy transcription was not published."

        # For each target language, check translated data and TTS audio.
        for lang in admin_config.get("target_languages", []):
            if lang.upper() == admin_config.get("default_source_language", "ENGLISH").upper():
                continue

            # Check translated captions.
            response = client_api.get(f"/captions/translated?lang={lang}")
            assert response.status_code == 200

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            translated_data = chunks[0] if chunks else []

            assert any("dummy translation" in item["text"] for item in translated_data), f"Dummy translation not published for language {lang}."

            # Check translated audio TTS.
            response = client_api.get(f"/audio/translated-voice?lang={lang}")
            assert response.status_code == 200

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            audio_data = chunks[0] if chunks else []

            for entry in audio_data:
                decoded_audio = base64.b64decode(entry["audio"])
                assert decoded_audio == b"dummy tts audio", f"Dummy TTS audio mismatch for language {lang}."


@pytest.mark.asyncio
class TestSpeakingCaptions:
    def test_regular_endpoint(self, setup_local_audio, configure_testing):
        """Test the regular speaking captions endpoint"""
        with TestClient(app) as client_api:
            time.sleep(5)

            response = client_api.get("/captions/speaking")
            assert response.status_code == 200

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            speaking_data = chunks[0] if chunks else []

            has_dummy = any("dummy transcription" in item["text"] for item in speaking_data)
            assert has_dummy, f"Dummy transcription was not published. Response: {speaking_data}"

    def test_streaming_endpoint(self, setup_local_audio, configure_testing):
        """Test the streaming speaking captions endpoint"""
        with TestClient(app) as client_api:
            time.sleep(5)

            with client_api.get("/captions/speaking/stream") as response:
                assert response.status_code == 200

                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8")
                        data = json.loads(data_str)
                        if data:  # If we got actual data (not empty keep-alive)
                            has_dummy = any("dummy transcription" in item["text"] for item in data)
                            assert has_dummy, f"Dummy transcription was not published in streaming. Data: {data}"
                            break


@pytest.mark.asyncio
class TestTranslatedCaptions:
    @pytest.mark.parametrize(
        "lang",
        [lang for lang in admin_config.get("target_languages", []) if lang.upper() != admin_config.get("default_source_language", "ENGLISH").upper()],
    )
    def test_regular_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the regular translated captions endpoint for each target language"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(5)

            # Test regular endpoint
            response = client_api.get(f"/captions/translated?lang={lang}")
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            translated_data = chunks[0] if chunks else []

            # Check for dummy translation
            has_dummy = any("dummy translation" in item["text"] for item in translated_data)
            assert has_dummy, f"Dummy translation not published for language {lang}. Response: {translated_data}"

    @pytest.mark.parametrize(
        "lang",
        [lang for lang in admin_config.get("target_languages", []) if lang.upper() != admin_config.get("default_source_language", "ENGLISH").upper()],
    )
    def test_streaming_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the streaming translated captions endpoint for each target language"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(5)

            # Test streaming endpoint
            with client_api.get(f"/captions/translated/stream?lang={lang}") as response:
                assert response.status_code == 200, f"Status code: {response.status_code}"

                # Read the first event with data
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8")
                        data = json.loads(data_str)
                        if data:
                            has_dummy = any("dummy translation" in item["text"] for item in data)
                            assert has_dummy, f"Dummy translation not published for language {lang} in streaming. Data: {data}"
                            break


@pytest.mark.asyncio
class TestTranslatedAudio:
    @pytest.mark.parametrize(
        "lang",
        [lang for lang in admin_config.get("target_languages", []) if lang.upper() != admin_config.get("default_source_language", "ENGLISH").upper()],
    )
    def test_regular_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the regular translated audio endpoint for each target language"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(5)

            # Test regular endpoint
            response = client_api.get(f"/audio/translated-voice?lang={lang}")
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            audio_data = chunks[0] if chunks else []

            # Check audio content
            assert audio_data, f"No audio data returned for language {lang}"
            for entry in audio_data:
                decoded_audio = base64.b64decode(entry["audio"])
                assert decoded_audio == b"dummy tts audio", f"Dummy TTS audio mismatch for language {lang}. Got: {decoded_audio}"

    @pytest.mark.parametrize(
        "lang",
        [lang for lang in admin_config.get("target_languages", []) if lang.upper() != admin_config.get("default_source_language", "ENGLISH").upper()],
    )
    def test_streaming_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the streaming translated audio endpoint for each target language"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(5)

            # Test streaming endpoint
            with client_api.get(f"/audio/translated-voice/stream?lang={lang}") as response:
                assert response.status_code == 200, f"Status code: {response.status_code}"

                # Read the first event with data
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8")
                        data = json.loads(data_str)
                        if data:
                            assert data, f"No audio data returned for language {lang} in streaming"
                            for entry in data:
                                decoded_audio = base64.b64decode(entry["audio"])
                                assert (
                                    decoded_audio == b"dummy tts audio"
                                ), f"Dummy TTS audio mismatch for language {lang} in streaming. Got: {decoded_audio}"
                            break


@pytest.mark.asyncio
class TestPublishEndpoint:
    def test_upload_mode(self, configure_testing):
        """Test the publish endpoint in upload mode"""
        with TestClient(app) as client_api:
            # Create a dummy audio file
            dummy_audio = b"dummy audio data"
            files = {"audio_file": ("test.wav", dummy_audio)}

            # Set up form data
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": 16000}

            # Send the request
            response = client_api.post("/publish", files=files, data=data)
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Verify the response
            response_data = response.json()
            assert "message" in response_data, f"Response missing 'message' field: {response_data}"

            # Wait for processing to complete
            time.sleep(5)

            # Verify that the data was published by checking one of the endpoints
            response = client_api.get("/captions/speaking")
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            speaking_data = chunks[0] if chunks else []

            # The mock should have published something
            assert speaking_data, "No speaking data was published after upload"
