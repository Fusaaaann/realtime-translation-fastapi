# test_local_audio_integration.py
import json
import numpy as np
import time
import base64
import asyncio
from typing import Tuple
import logging
import os

import pytest
from fastapi.testclient import TestClient
from mock_audio import mock_initialize_audio_recorder, mock_get_audio_frames, mock_continuous_recorder

# Import the necessary items from your app.
from server import app, published_data_store, admin_config, client  # This is the ElevenLabs client in your app.

logger = logging.getLogger(__name__)

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
def dummy_get_local_audio_source(record_seconds: int = 15, sample_rate: int = 16000, device_id: int = None) -> Tuple[bytes, int]:
    """
    Mock implementation that simulates capturing audio from a sound device.
    Instead of recording from microphone, this function loads a predefined audio file
    and adjusts it to match the requested parameters.

    :param record_seconds: Duration of the audio recording in seconds (default is 15 seconds).
    :param sample_rate: Sample rate for recording (default is 16000 Hz).
    :param device_id: ID of the sound device to use. If None, uses the default device.
    :return: Tuple of (audio data bytes, sample rate)
    """
    # Simulate listing available devices for debugging
    logger.info("Available audio devices: [Mock device list]")

    # Simulate device selection logic
    device_info = None
    if device_id is not None:
        try:
            device_info = {"name": f"Mock Device {device_id}"}
            logger.info(f"Using audio device: {device_info['name']} (ID: {device_id})")
        except Exception as e:
            logger.warning(f"Could not use device ID {device_id}: {str(e)}. Falling back to default device.")

    # Simulate default device selection
    if device_info is None:
        logger.info("Using default input device: Mock Default Device")

    logger.info(f"Recording {record_seconds} seconds of audio from the microphone at {sample_rate} Hz...")

    try:
        # Path to the test audio file
        test_audio_path = "./test_audio.wav"

        # Check if the file exists
        if not os.path.exists(test_audio_path):
            logger.warning(f"Test audio file not found at {test_audio_path}. Generating silent audio.")
            # Create a numpy array of zeros (int16 PCM), simulating silence
            recording = np.zeros((int(record_seconds * sample_rate), 1), dtype=np.int16)
            audio_data = recording.tobytes()
            return audio_data, sample_rate

        # Load the audio file using scipy
        import scipy.io.wavfile as wavfile

        file_sample_rate, audio_data = wavfile.read(test_audio_path)

        # Ensure the audio data is in the right format (int16)
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # If the audio is stereo, convert to mono by averaging channels
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)

        # Reshape to ensure it's (samples, 1)
        audio_data = audio_data.reshape(-1, 1)

        # Resample if the file's sample rate doesn't match the requested sample rate
        if file_sample_rate != sample_rate:
            from scipy.signal import resample

            num_samples = int(len(audio_data) * sample_rate / file_sample_rate)
            audio_data = resample(audio_data, num_samples).astype(np.int16)

        # Calculate the target number of samples
        target_samples = int(record_seconds * sample_rate)
        current_samples = len(audio_data)

        # Adjust the audio length to match record_seconds
        if current_samples > target_samples:
            # Trim the audio if it's too long
            audio_data = audio_data[:target_samples]
        elif current_samples < target_samples:
            # Pad with silence if it's too short
            silence_samples = target_samples - current_samples
            silence = np.zeros((silence_samples, 1), dtype=np.int16)
            audio_data = np.vstack((audio_data, silence))

        # Convert the NumPy array to bytes
        audio_data = audio_data.tobytes()
        return audio_data, sample_rate

    except Exception as e:
        logger.error(f"Error loading test audio: {str(e)}")
        # Return empty audio in case of error, exactly as the original function does
        empty_audio = np.zeros((int(record_seconds * sample_rate), 1), dtype=np.int16).tobytes()
        return empty_audio, sample_rate


@pytest.fixture
def setup_local_audio(monkeypatch, request):
    """
    This fixture overrides the audio recording functions to use mock implementations
    that generate silent audio instead of recording from a real microphone.
    """
    use_real_mic = request.config.getoption("--use-real-mic")
    if use_real_mic:
        return

    # Replace the audio recording functions with our mock versions
    monkeypatch.setattr("server.initialize_audio_recorder", mock_initialize_audio_recorder)
    monkeypatch.setattr("server.get_audio_frames", mock_get_audio_frames)

    # Initialize the mock recorder
    mock_initialize_audio_recorder()

    # Clean up when the test is done
    yield

    if mock_continuous_recorder is not None:
        mock_continuous_recorder.stop()


# --- Monkeypatching & Configuration Fixture ---


@pytest.fixture
def configure_testing(monkeypatch, request):
    """
    This fixture patches external dependencies and modifies configurations for faster testing.
    """
    use_real_api = request.config.getoption("--use-real-api")
    if use_real_api:
        return
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
        assert any("dummy transcription" in item["text"] for item in speaking_data), "Dummy transcription was not published."

        # For each target language (as per admin_config), check translated data and TTS audio.
        for lang in admin_config.get("target_languages", []):
            if lang.upper() == admin_config.get("default_source_language", "ENGLISH").upper():
                continue

            # Check translated captions.
            response = client_api.get(f"/captions/translated?lang={lang}")
            assert response.status_code == 200
            translated_data = response.json()
            assert any("dummy translation" in item["text"] for item in translated_data), f"Dummy translation not published for language {lang}."

            # Check translated audio TTS.
            response = client_api.get(f"/audio/translated-voice?lang={lang}")
            assert response.status_code == 200
            audio_data = response.json()
            for entry in audio_data:
                decoded_audio = base64.b64decode(entry["audio"])
                assert decoded_audio == b"dummy tts audio", f"Dummy TTS audio mismatch for language {lang}."


@pytest.mark.asyncio
class TestSpeakingCaptions:
    def test_regular_endpoint(self, setup_local_audio, configure_testing):
        """Test the regular speaking captions endpoint"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(3)

            # Test regular endpoint
            response = client_api.get("/captions/speaking")
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            speaking_data = chunks[0] if chunks else []

            # Check for dummy transcription
            has_dummy = any("dummy transcription" in item["text"] for item in speaking_data)
            assert has_dummy, f"Dummy transcription was not published. Response: {speaking_data}"

    def test_streaming_endpoint(self, setup_local_audio, configure_testing):
        """Test the streaming speaking captions endpoint"""
        with TestClient(app) as client_api:
            # Wait enough time for the background task to process the dummy audio
            time.sleep(3)

            # Test streaming endpoint
            with client_api.get("/captions/speaking/stream") as response:
                assert response.status_code == 200, f"Status code: {response.status_code}"

                # Read the first event with data
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode("utf-8")  # Skip 'data: ' prefix
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
            time.sleep(3)

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
            time.sleep(3)

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
            time.sleep(3)

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
            time.sleep(3)

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
            time.sleep(3)

            # Verify that the data was published by checking one of the endpoints
            response = client_api.get("/captions/speaking")
            assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

            # Parse chunked response
            content = response.content.decode().strip()
            chunks = [json.loads(chunk) for chunk in content.split("\n") if chunk]
            speaking_data = chunks[0] if chunks else []

            # The mock should have published something
            assert speaking_data, "No speaking data was published after upload"
