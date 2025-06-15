# test_full_server.py
import json
import time
import base64
import asyncio
import logging
from typing import List, Dict, Any

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


# --- Helper Functions ---


def parse_response_safely(response) -> List[Dict[str, Any]]:
    """Safely parse chunked or regular JSON response"""
    try:
        content = response.content.decode().strip()
        if not content:
            return []

        # Try parsing as single JSON first
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # Try parsing as chunked response
            chunks = []
            for line in content.split("\n"):
                if line.strip():
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return chunks[0] if chunks else []
    except Exception as e:
        logger.warning(f"Failed to parse response: {e}")
        return []


def wait_for_data(client_api, endpoint: str, max_wait: int = 10, check_interval: float = 0.5) -> List[Dict[str, Any]]:
    """Wait for data to appear with polling"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = client_api.get(endpoint)
            if response.status_code == 200:
                data = parse_response_safely(response)
                if data and any(item for item in data if item):  # Check for non-empty items
                    return data
        except Exception as e:
            logger.warning(f"Error polling {endpoint}: {e}")
        time.sleep(check_interval)
    return []


def validate_text_item(item: Dict[str, Any], item_type: str = "text") -> None:
    """Validate structure of a text item (speaking/translated)"""
    assert isinstance(item, dict), f"{item_type} item should be a dictionary"
    assert "text" in item, f"{item_type} item missing 'text' field"
    assert isinstance(item["text"], str), f"{item_type} text should be string"
    assert len(item["text"].strip()) > 0, f"{item_type} text should not be empty"
    assert len(item["text"]) < 10000, f"{item_type} text suspiciously long"


def validate_audio_item(item: Dict[str, Any]) -> None:
    """Validate structure of an audio item"""
    assert isinstance(item, dict), "Audio item should be a dictionary"
    assert "audio" in item, "Audio item missing 'audio' field"
    assert isinstance(item["audio"], str), "Audio data should be base64 string"

    # Validate base64 encoding
    try:
        decoded_audio = base64.b64decode(item["audio"])
        assert len(decoded_audio) > 0, "Decoded audio data is empty"
        assert isinstance(decoded_audio, bytes), "Decoded audio should be bytes"
        # Reasonable size check (not too small, not too large)
        assert 1 <= len(decoded_audio) <= 1024 * 1024, f"Audio size {len(decoded_audio)} seems unreasonable"
    except Exception as e:
        pytest.fail(f"Failed to decode audio data: {e}")


def get_target_languages() -> List[str]:
    """Get list of target languages excluding source language"""
    target_langs = admin_config.get("target_languages", [])
    source_lang = admin_config.get("default_source_language", "ENGLISH").upper()
    return [lang for lang in target_langs if lang.upper() != source_lang]


def read_streaming_data(response, max_items: int = 1) -> List[Dict[str, Any]]:
    """Read data from streaming response"""
    items_found = 0
    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            try:
                data_str = line[6:].decode("utf-8")
                data = json.loads(data_str)
                if data and any(item for item in data if item):  # Non-empty data
                    items_found += 1
                    if items_found >= max_items:
                        return data
            except json.JSONDecodeError:
                continue
    return []


# --- Fixtures ---


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


# --- Test Cases ---


def test_local_audio_source_integration(setup_local_audio, configure_testing):
    """
    Integration test for local audio source mode using dummy external dependencies.
    """
    with TestClient(app) as client_api:
        # Wait for speaking data to be processed
        speaking_data = wait_for_data(client_api, "/captions/speaking")
        assert speaking_data, "No speaking data was published within timeout"

        # Validate speaking data structure
        for item in speaking_data:
            validate_text_item(item, "speaking")

        # Test each target language
        target_languages = get_target_languages()
        assert target_languages, "No target languages configured for testing"

        for lang in target_languages:
            # Check translated captions
            translated_data = wait_for_data(client_api, f"/captions/translated?lang={lang}")
            assert translated_data, f"No translated data published for language {lang}"

            for item in translated_data:
                validate_text_item(item, f"translated-{lang}")

            # Check translated audio TTS
            audio_data = wait_for_data(client_api, f"/audio/translated-voice?lang={lang}")
            assert audio_data, f"No audio data published for language {lang}"

            for item in audio_data:
                validate_audio_item(item)


@pytest.mark.asyncio
class TestSpeakingCaptions:
    def test_regular_endpoint(self, setup_local_audio, configure_testing):
        """Test the regular speaking captions endpoint"""
        with TestClient(app) as client_api:
            speaking_data = wait_for_data(client_api, "/captions/speaking")

            assert speaking_data, "No speaking data was published"
            assert isinstance(speaking_data, list), "Speaking data should be a list"

            for item in speaking_data:
                validate_text_item(item, "speaking")

    def test_streaming_endpoint(self, setup_local_audio, configure_testing):
        """Test the streaming speaking captions endpoint"""
        with TestClient(app) as client_api:
            # Give some time for data to be available
            time.sleep(2)

            with client_api.get("/captions/speaking/stream") as response:
                assert response.status_code == 200, f"Streaming endpoint failed: {response.status_code}"

                data = read_streaming_data(response)
                assert data, "No data received from streaming endpoint"

                for item in data:
                    validate_text_item(item, "speaking-stream")


@pytest.mark.asyncio
class TestTranslatedCaptions:
    @pytest.mark.parametrize("lang", get_target_languages())
    def test_regular_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the regular translated captions endpoint for each target language"""
        with TestClient(app) as client_api:
            translated_data = wait_for_data(client_api, f"/captions/translated?lang={lang}")

            assert translated_data, f"No translated data published for language {lang}"
            assert isinstance(translated_data, list), f"Translated data should be a list for {lang}"

            for item in translated_data:
                validate_text_item(item, f"translated-{lang}")

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_streaming_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the streaming translated captions endpoint for each target language"""
        with TestClient(app) as client_api:
            # Give some time for data to be available
            time.sleep(2)

            with client_api.get(f"/captions/translated/stream?lang={lang}") as response:
                assert response.status_code == 200, f"Streaming endpoint failed for {lang}: {response.status_code}"

                data = read_streaming_data(response)
                assert data, f"No translated data received from streaming endpoint for {lang}"

                for item in data:
                    validate_text_item(item, f"translated-stream-{lang}")


@pytest.mark.asyncio
class TestTranslatedAudio:
    @pytest.mark.parametrize("lang", get_target_languages())
    def test_regular_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the regular translated audio endpoint for each target language"""
        with TestClient(app) as client_api:
            audio_data = wait_for_data(client_api, f"/audio/translated-voice?lang={lang}")

            assert audio_data, f"No audio data returned for language {lang}"
            assert isinstance(audio_data, list), f"Audio data should be a list for {lang}"

            for item in audio_data:
                validate_audio_item(item)

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_streaming_endpoint(self, setup_local_audio, configure_testing, lang):
        """Test the streaming translated audio endpoint for each target language"""
        with TestClient(app) as client_api:
            # Give some time for data to be available
            time.sleep(2)

            with client_api.get(f"/audio/translated-voice/stream?lang={lang}") as response:
                assert response.status_code == 200, f"Streaming audio endpoint failed for {lang}: {response.status_code}"

                data = read_streaming_data(response)
                assert data, f"No audio data received from streaming endpoint for {lang}"

                for item in data:
                    validate_audio_item(item)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Not run to shorten test time")
class TestPublishEndpoint:
    def test_upload_mode(self, configure_testing):
        """Test the publish endpoint in upload mode"""
        with TestClient(app) as client_api:
            # Create a dummy audio file
            dummy_audio = b"dummy audio data for testing upload"
            files = {"audio_file": ("test.wav", dummy_audio)}

            # Set up form data
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": 16000}

            # Send the request
            response = client_api.post("/publish", files=files, data=data)
            assert response.status_code == 200, f"Upload failed: {response.status_code}, Response: {response.text}"

            # Verify the response structure
            response_data = response.json()
            assert isinstance(response_data, dict), "Response should be a dictionary"
            assert "message" in response_data, f"Response missing 'message' field: {response_data}"
            assert isinstance(response_data["message"], str), "Message should be a string"

            # Wait for processing and verify data was published
            speaking_data = wait_for_data(client_api, "/captions/speaking", max_wait=15)
            assert speaking_data, "No speaking data was published after upload"

            # Validate the published data
            for item in speaking_data:
                validate_text_item(item, "upload-speaking")

    def test_upload_mode_invalid_file(self, configure_testing):
        """Test the publish endpoint with invalid file"""
        with TestClient(app) as client_api:
            # Test with no file
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": 16000}
            response = client_api.post("/publish", data=data)

            # Should handle missing file gracefully
            assert response.status_code in [400, 422], f"Expected error status, got: {response.status_code}"

    def test_upload_mode_invalid_params(self, configure_testing):
        """Test the publish endpoint with invalid parameters"""
        with TestClient(app) as client_api:
            dummy_audio = b"dummy audio data"
            files = {"audio_file": ("test.wav", dummy_audio)}

            # Test with invalid sample rate
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": "invalid"}
            response = client_api.post("/publish", files=files, data=data)

            # Should handle invalid parameters gracefully
            assert response.status_code in [400, 422], f"Expected error status for invalid params, got: {response.status_code}"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Not run to shorten test time")
class TestEndpointAvailability:
    """Test that all endpoints are available and return proper status codes"""

    def test_speaking_endpoints_availability(self, configure_testing):
        """Test that speaking caption endpoints are available"""
        with TestClient(app) as client_api:
            # Test regular endpoint
            response = client_api.get("/captions/speaking")
            assert response.status_code == 200, f"Speaking captions endpoint unavailable: {response.status_code}"

            # Test streaming endpoint
            with client_api.get("/captions/speaking/stream") as stream_response:
                assert stream_response.status_code == 200, f"Speaking captions streaming endpoint unavailable: {stream_response.status_code}"

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_translated_endpoints_availability(self, configure_testing, lang):
        """Test that translated caption endpoints are available for each language"""
        with TestClient(app) as client_api:
            # Test regular endpoint
            response = client_api.get(f"/captions/translated?lang={lang}")
            assert response.status_code == 200, f"Translated captions endpoint unavailable for {lang}: {response.status_code}"

            # Test streaming endpoint
            with client_api.get(f"/captions/translated/stream?lang={lang}") as stream_response:
                assert (
                    stream_response.status_code == 200
                ), f"Translated captions streaming endpoint unavailable for {lang}: {stream_response.status_code}"

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_audio_endpoints_availability(self, configure_testing, lang):
        """Test that audio endpoints are available for each language"""
        with TestClient(app) as client_api:
            # Test regular endpoint
            response = client_api.get(f"/audio/translated-voice?lang={lang}")
            assert response.status_code == 200, f"Audio endpoint unavailable for {lang}: {response.status_code}"

            # Test streaming endpoint
            with client_api.get(f"/audio/translated-voice/stream?lang={lang}") as stream_response:
                assert stream_response.status_code == 200, f"Audio streaming endpoint unavailable for {lang}: {stream_response.status_code}"

    def test_publish_endpoint_availability(self, configure_testing):
        """Test that publish endpoint is available"""
        with TestClient(app) as client_api:
            # Test with minimal valid data
            dummy_audio = b"test audio"
            files = {"audio_file": ("test.wav", dummy_audio)}
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": 16000}

            response = client_api.post("/publish", files=files, data=data)
            assert response.status_code in [200, 201], f"Publish endpoint unavailable: {response.status_code}"

    def test_invalid_language_handling(self, configure_testing):
        """Test how endpoints handle invalid language parameters"""
        with TestClient(app) as client_api:
            invalid_lang = "INVALID_LANGUAGE_CODE"

            # Test translated captions with invalid language
            response = client_api.get(f"/captions/translated?lang={invalid_lang}")
            # Should either return empty data or proper error
            assert response.status_code in [200, 400, 404], f"Unexpected status for invalid language: {response.status_code}"

            # Test audio with invalid language
            response = client_api.get(f"/audio/translated-voice?lang={invalid_lang}")
            assert response.status_code in [200, 400, 404], f"Unexpected status for invalid language in audio: {response.status_code}"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Not run to shorten test time")
class TestDataConsistency:
    """Test data consistency across different endpoints"""

    def test_speaking_data_consistency(self, setup_local_audio, configure_testing):
        """Test that speaking data is consistent between regular and streaming endpoints"""
        with TestClient(app) as client_api:
            # Get data from regular endpoint
            regular_data = wait_for_data(client_api, "/captions/speaking")
            assert regular_data, "No data from regular speaking endpoint"

            # Get data from streaming endpoint
            time.sleep(1)  # Brief pause
            with client_api.get("/captions/speaking/stream") as response:
                streaming_data = read_streaming_data(response)

            # Both should have data
            assert streaming_data, "No data from streaming speaking endpoint"

            # Validate structure consistency
            for item in regular_data:
                validate_text_item(item, "regular-speaking")
            for item in streaming_data:
                validate_text_item(item, "streaming-speaking")

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_translated_data_consistency(self, setup_local_audio, configure_testing, lang):
        """Test that translated data is consistent between regular and streaming endpoints"""
        with TestClient(app) as client_api:
            # Get data from regular endpoint
            regular_data = wait_for_data(client_api, f"/captions/translated?lang={lang}")
            assert regular_data, f"No data from regular translated endpoint for {lang}"

            # Get data from streaming endpoint
            time.sleep(1)  # Brief pause
            with client_api.get(f"/captions/translated/stream?lang={lang}") as response:
                streaming_data = read_streaming_data(response)

            # Both should have data
            assert streaming_data, f"No data from streaming translated endpoint for {lang}"

            # Validate structure consistency
            for item in regular_data:
                validate_text_item(item, f"regular-translated-{lang}")
            for item in streaming_data:
                validate_text_item(item, f"streaming-translated-{lang}")

    @pytest.mark.parametrize("lang", get_target_languages())
    def test_audio_data_consistency(self, setup_local_audio, configure_testing, lang):
        """Test that audio data is consistent between regular and streaming endpoints"""
        with TestClient(app) as client_api:
            # Get data from regular endpoint
            regular_data = wait_for_data(client_api, f"/audio/translated-voice?lang={lang}")
            assert regular_data, f"No data from regular audio endpoint for {lang}"

            # Get data from streaming endpoint
            time.sleep(1)  # Brief pause
            with client_api.get(f"/audio/translated-voice/stream?lang={lang}") as response:
                streaming_data = read_streaming_data(response)

            # Both should have data
            assert streaming_data, f"No data from streaming audio endpoint for {lang}"

            # Validate structure consistency
            for item in regular_data:
                validate_audio_item(item)
            for item in streaming_data:
                validate_audio_item(item)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Not run to shorten test time")
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_endpoints(self, configure_testing):
        """Test endpoints when no data is available"""
        with TestClient(app) as client_api:
            # Clear any existing data by waiting briefly and then testing
            time.sleep(0.5)

            # Test speaking endpoint with no data
            response = client_api.get("/captions/speaking")
            assert response.status_code == 200, "Speaking endpoint should return 200 even with no data"

            data = parse_response_safely(response)
            # Should return empty list or handle gracefully
            assert isinstance(data, list), "Response should be a list even when empty"

    def test_malformed_requests(self, configure_testing):
        """Test handling of malformed requests"""
        with TestClient(app) as client_api:
            # Test publish with malformed data
            response = client_api.post("/publish", data={"invalid": "data"})
            assert response.status_code in [400, 422], f"Should reject malformed request: {response.status_code}"

            # Test with missing required fields
            files = {"audio_file": ("test.wav", b"test")}
            response = client_api.post("/publish", files=files, data={"mode": "upload"})  # Missing required fields
            assert response.status_code in [400, 422], f"Should reject incomplete request: {response.status_code}"

    def test_large_file_handling(self, configure_testing):
        """Test handling of large files"""
        with TestClient(app) as client_api:
            # Create a reasonably large dummy file (1MB)
            large_audio = b"dummy audio data" * 70000  # ~1MB
            files = {"audio_file": ("large_test.wav", large_audio)}
            data = {"mode": "upload", "sourceLanguage": "ENGLISH", "sampleRate": 16000}

            response = client_api.post("/publish", files=files, data=data)
            # Should either accept or reject gracefully
            assert response.status_code in [200, 201, 413, 422], f"Unexpected status for large file: {response.status_code}"

    def test_concurrent_requests(self, setup_local_audio, configure_testing):
        """Test handling of concurrent requests"""
        import threading
        import queue

        results = queue.Queue()

        def make_request():
            try:
                with TestClient(app) as client_api:
                    response = client_api.get("/captions/speaking")
                    results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))

        # Start multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success":
                assert result_value == 200, f"Concurrent request failed with status: {result_value}"
                success_count += 1

        assert success_count > 0, "No concurrent requests succeeded"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Not run to shorten test time")
class TestPerformance:
    """Basic performance and timeout tests"""

    def test_endpoint_response_times(self, configure_testing):
        """Test that endpoints respond within reasonable time"""
        with TestClient(app) as client_api:
            endpoints = [
                "/captions/speaking",
                "/captions/translated?lang=SPANISH" if "SPANISH" in get_target_languages() else None,
                "/audio/translated-voice?lang=SPANISH" if "SPANISH" in get_target_languages() else None,
            ]

            for endpoint in endpoints:
                if endpoint is None:
                    continue

                start_time = time.time()
                response = client_api.get(endpoint)
                response_time = time.time() - start_time

                assert response.status_code == 200, f"Endpoint {endpoint} failed: {response.status_code}"
                assert response_time < 30, f"Endpoint {endpoint} too slow: {response_time:.2f}s"

    def test_streaming_timeout(self, configure_testing):
        """Test that streaming endpoints don't hang indefinitely"""
        with TestClient(app) as client_api:
            start_time = time.time()

            try:
                with client_api.get("/captions/speaking/stream") as response:
                    assert response.status_code == 200

                    # Read a few lines or timeout
                    lines_read = 0
                    for line in response.iter_lines():
                        lines_read += 1
                        if lines_read >= 3 or (time.time() - start_time) > 10:
                            break

                    elapsed = time.time() - start_time
                    assert elapsed < 15, f"Streaming took too long: {elapsed:.2f}s"

            except Exception as e:
                elapsed = time.time() - start_time
                assert elapsed < 15, f"Streaming failed after {elapsed:.2f}s: {e}"


@pytest.mark.asyncio
class TestAdminEndpoints:
    """Test admin endpoints with authentication and basic functionality"""

    @pytest.fixture
    def admin_headers(self):
        """Get headers with admin token"""
        admin_token = admin_config.get("admin_token", "test-admin-token")
        return {"X-Admin-Token": admin_token}

    @pytest.fixture
    def invalid_headers(self):
        """Get headers with invalid admin token"""
        return {"X-Admin-Token": "invalid-token"}

    def test_admin_auth_required(self, configure_testing):
        """Test that admin endpoints require authentication"""
        with TestClient(app) as client_api:
            # Test without token
            response = client_api.get("/admin/config")
            assert response.status_code == 422, "Should require admin token header"

            # Test with invalid token
            response = client_api.get("/admin/config", headers={"X-Admin-Token": "invalid"})
            assert response.status_code == 401, "Should reject invalid admin token"

    def test_get_config(self, configure_testing, admin_headers):
        """Test getting admin configuration"""
        with TestClient(app) as client_api:
            response = client_api.get("/admin/config", headers=admin_headers)
            assert response.status_code == 200, f"Get config failed: {response.status_code}"

            config_data = response.json()
            assert isinstance(config_data, dict), "Config should be a dictionary"

            # Check for expected config keys
            expected_keys = ["admin_token", "publish_mode", "default_source_language", "target_languages"]
            for key in expected_keys:
                assert key in config_data, f"Missing config key: {key}"

    def test_update_config_basic(self, configure_testing, admin_headers):
        """Test basic configuration update"""
        with TestClient(app) as client_api:
            # Get current config
            response = client_api.get("/admin/config", headers=admin_headers)
            original_config = response.json()
            logger.info(f"{original_config=}")

            # Update some basic settings
            update_data = {"default_source_language": "SPANISH", "poll_interval": 5}

            response = client_api.post("/admin/config", headers=admin_headers, json=update_data)
            assert response.status_code == 200, f"Config update failed: {response.status_code}, {response.text}"

            response_data = response.json()
            assert "message" in response_data, "Response should contain message"
            assert "config" in response_data, "Response should contain updated config"

            # Verify the update
            updated_config = response_data["config"]
            assert updated_config["default_source_language"] == "SPANISH", "Source language not updated"
            assert updated_config["poll_interval"] == 5, "Poll interval not updated"

    def test_update_config_admin_token_protection(self, configure_testing, admin_headers):
        """Test that admin_token cannot be updated"""
        with TestClient(app) as client_api:
            # Try to update admin_token
            update_data = {"admin_token": "new-token-should-be-ignored", "poll_interval": 3}

            response = client_api.post("/admin/config", headers=admin_headers, json=update_data)
            assert response.status_code == 200, f"Config update failed: {response.status_code}"

            # Verify admin_token was not changed
            updated_config = response.json()["config"]
            assert updated_config["admin_token"] != "new-token-should-be-ignored", "Admin token should not be updatable"
            assert updated_config["poll_interval"] == 3, "Other settings should still update"

    def test_update_config_invalid_auth(self, configure_testing, invalid_headers):
        """Test config update with invalid authentication"""
        with TestClient(app) as client_api:
            update_data = {"poll_interval": 10}

            response = client_api.post("/admin/config", headers=invalid_headers, json=update_data)
            assert response.status_code == 401, "Should reject invalid admin token"

    def test_get_audio_status(self, configure_testing, admin_headers):
        """Test getting audio status"""
        with TestClient(app) as client_api:
            response = client_api.get("/admin/audio-status", headers=admin_headers)
            assert response.status_code == 200, f"Audio status failed: {response.status_code}"

            status_data = response.json()
            assert isinstance(status_data, dict), "Status should be a dictionary"
            assert "publish_mode" in status_data, "Status should contain publish_mode"
            assert "audio_recorder" in status_data, "Status should contain audio_recorder info"
            assert "audio_processor_task" in status_data, "Status should contain audio_processor_task info"

    def test_audio_control_invalid_action(self, configure_testing, admin_headers):
        """Test audio control with invalid action"""
        with TestClient(app) as client_api:
            response = client_api.post("/admin/audio-control?action=invalid", headers=admin_headers)
            assert response.status_code == 400, "Should reject invalid action"

    def test_audio_control_status(self, configure_testing, admin_headers):
        """Test audio control status action"""
        with TestClient(app) as client_api:
            response = client_api.post("/admin/audio-control?action=status", headers=admin_headers)
            assert response.status_code == 200, f"Audio control status failed: {response.status_code}"

            control_data = response.json()
            assert "action" in control_data, "Response should contain action"
            assert "status" in control_data, "Response should contain status"
            assert control_data["action"] == "status", "Action should match request"

    def test_audio_control_start_stop(self, configure_testing, admin_headers):
        """Test audio control start/stop actions"""
        with TestClient(app) as client_api:
            # Test start action
            response = client_api.post("/admin/audio-control?action=start", headers=admin_headers)
            assert response.status_code == 200, f"Audio control start failed: {response.status_code}"

            start_data = response.json()
            assert "action" in start_data, "Start response should contain action"
            assert "success" in start_data, "Start response should contain success status"
            assert start_data["action"] == "start", "Action should match request"

            # Test stop action
            response = client_api.post("/admin/audio-control?action=stop", headers=admin_headers)
            assert response.status_code == 200, f"Audio control stop failed: {response.status_code}"

            stop_data = response.json()
            assert "action" in stop_data, "Stop response should contain action"
            assert "success" in stop_data, "Stop response should contain success status"
            assert stop_data["action"] == "stop", "Action should match request"

    def test_audio_control_restart(self, configure_testing, admin_headers):
        """Test audio control restart action"""
        with TestClient(app) as client_api:
            response = client_api.post("/admin/audio-control?action=restart", headers=admin_headers)
            assert response.status_code == 200, f"Audio control restart failed: {response.status_code}"

            restart_data = response.json()
            assert "action" in restart_data, "Restart response should contain action"
            assert "success" in restart_data, "Restart response should contain success status"
            assert restart_data["action"] == "restart", "Action should match request"

    def test_audio_control_invalid_auth(self, configure_testing, invalid_headers):
        """Test audio control with invalid authentication"""
        with TestClient(app) as client_api:
            response = client_api.post("/admin/audio-control?action=status", headers=invalid_headers)
            assert response.status_code == 401, "Should reject invalid admin token"

    def test_list_audio_devices(self, configure_testing, admin_headers):
        """Test listing audio devices"""
        with TestClient(app) as client_api:
            response = client_api.get("/admin/audio-devices", headers=admin_headers)
            assert response.status_code == 200, f"List audio devices failed: {response.status_code}"

            devices_data = response.json()
            assert isinstance(devices_data, dict), "Devices response should be a dictionary"
            assert "devices" in devices_data, "Response should contain devices list"
            assert "current_device_id" in devices_data, "Response should contain current device ID"

            devices_list = devices_data["devices"]
            assert isinstance(devices_list, list), "Devices should be a list"

            # Validate device structure if devices exist
            for device in devices_list:
                assert isinstance(device, dict), "Each device should be a dictionary"
                required_fields = ["id", "name", "channels", "default_samplerate"]
                for field in required_fields:
                    assert field in device, f"Device missing required field: {field}"

                # Validate field types
                assert isinstance(device["id"], int), "Device ID should be integer"
                assert isinstance(device["name"], str), "Device name should be string"
                assert isinstance(device["channels"], int), "Device channels should be integer"
                assert device["channels"] > 0, "Input device should have channels > 0"

    def test_list_audio_devices_invalid_auth(self, configure_testing, invalid_headers):
        """Test listing audio devices with invalid authentication"""
        with TestClient(app) as client_api:
            response = client_api.get("/admin/audio-devices", headers=invalid_headers)
            assert response.status_code == 401, "Should reject invalid admin token"

    def test_config_update_publish_mode_change(self, configure_testing, admin_headers):
        """Test configuration update that changes publish mode"""
        with TestClient(app) as client_api:
            # Get current config
            response = client_api.get("/admin/config", headers=admin_headers)
            original_config = response.json()
            original_mode = original_config.get("publish_mode", "upload")

            # Change to different mode
            new_mode = "local-audio-source" if original_mode == "upload" else "upload"
            update_data = {"publish_mode": new_mode}

            response = client_api.post("/admin/config", headers=admin_headers, json=update_data)
            assert response.status_code == 200, f"Publish mode change failed: {response.status_code}"

            response_data = response.json()
            assert response_data["config"]["publish_mode"] == new_mode, "Publish mode not updated"

            # Restore original mode
            restore_data = {"publish_mode": original_mode}
            response = client_api.post("/admin/config", headers=admin_headers, json=restore_data)
            assert response.status_code == 200, "Failed to restore original publish mode"

    def test_config_update_target_languages(self, configure_testing, admin_headers):
        """Test updating target languages configuration"""
        with TestClient(app) as client_api:
            # Get current config
            response = client_api.get("/admin/config", headers=admin_headers)
            original_config = response.json()
            logger.info(f"{original_config=}")

            # Update target languages
            new_languages = ["ENGLISH", "SPANISH", "FRENCH"]
            update_data = {"target_languages": new_languages}

            response = client_api.post("/admin/config", headers=admin_headers, json=update_data)
            assert response.status_code == 200, f"Target languages update failed: {response.status_code}"

            response_data = response.json()
            updated_languages = response_data["config"]["target_languages"]
            assert updated_languages == new_languages, "Target languages not updated correctly"

    def test_admin_endpoints_error_handling(self, configure_testing, admin_headers):
        """Test error handling in admin endpoints"""
        with TestClient(app) as client_api:
            # Test invalid JSON in config update
            response = client_api.post("/admin/config", headers=admin_headers, data="invalid json")
            assert response.status_code in [400, 422], "Should handle invalid JSON gracefully"

            # Test audio control with missing action parameter
            response = client_api.post("/admin/audio-control", headers=admin_headers)
            assert response.status_code in [400, 422], "Should require action parameter"

    def test_admin_config_persistence(self, configure_testing, admin_headers):
        """Test that configuration changes persist"""
        with TestClient(app) as client_api:
            # Update a setting
            test_value = 123
            update_data = {"poll_interval": test_value}

            response = client_api.post("/admin/config", headers=admin_headers, json=update_data)
            assert response.status_code == 200, "Config update failed"

            # Verify the change persists by getting config again
            response = client_api.get("/admin/config", headers=admin_headers)
            assert response.status_code == 200, "Get config failed"

            config_data = response.json()
            assert config_data["poll_interval"] == test_value, "Configuration change did not persist"
