import json
import base64
import time

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app from the server file.
from example_browserclient.server import app, published_data_store, admin_config

client = TestClient(app)

def test_publish_endpoint_valid(monkeypatch):
    """
    Test the POST /publish endpoint with a valid audio file and sample rate.

    What to test:
    - Sending a valid audio file (binary data) with a valid 'sampleRate' form field.
    - Ensuring the endpoint returns a JSON response containing a 'timestamp' and a success message.
    - Verifying the transcription output is stored in the in-memory datastore.
    
    Edge cases:
    - Missing audio file or 'sampleRate' in the request.
    - Invalid 'sampleRate' values (e.g. negative numbers, non-integer values).
    - Corrupted audio file data.
    - Very large audio payloads.
    """
    # Import the server module so we can patch its ElevenLabs client's STT conversion method.
    import example_browserclient.server as server_mod
    # Monkey-patch the speech_to_text.convert method to always return a dummy transcription.
    monkeypatch.setattr(
        server_mod.client.speech_to_text,
        "convert",
        lambda model_id, file: {"text": "dummy transcription"}
    )
    
    # Create a dummy audio file (ensure length is a multiple of 2 for int16 conversion)
    dummy_audio = b'\x00\x00' * 10  # 20 bytes of dummy data
    
    # Send a POST request to the /publish endpoint with the dummy audio and a valid sample rate.
    response = client.post(
        "/publish",
        files={"audio_file": ("dummy.wav", dummy_audio, "audio/wav")},
        data={"sampleRate": "44100"}
    )
    
    assert response.status_code == 200, "Expected status code 200 for valid publish request."
    json_data = response.json()
    assert "timestamp" in json_data, "Response should contain a 'timestamp'."
    assert "Published transcription" in json_data.get("message", ""), "Success message missing in response."
    
    # Verify that the dummy transcription was stored in the in-memory datastore.
    stored_entries = server_mod.published_data_store.speaking
    assert any("dummy transcription" in entry.get("text", "") for entry in stored_entries), \
        "The dummy transcription was not found in the published speaking captions."


def test_get_speaking_captions():
    """
    Test the GET /captions/speaking endpoint.

    What to test:
    - Request without a 'timestamp' parameter returns all published speaking captions.
    - Request with a valid 'timestamp' parameter returns only captions published after that timestamp.
    
    Edge cases:
    - A 'timestamp' set to a value later than any published caption (should return an empty list).
    - Invalid timestamp format (e.g. string or negative number).
    - No speaking captions available.
    """
    # Clear existing speaking captions and create test entries.
    published_data_store.speaking.clear()
    import time
    current_time = time.time()
    old_caption = {"timestamp": current_time - 20, "text": "old caption"}
    new_caption = {"timestamp": current_time + 10, "text": "new caption"}
    published_data_store.speaking.extend([old_caption, new_caption])
    
    # Test: Without a 'timestamp' parameter, both entries should be returned.
    response = client.get("/captions/speaking")
    data = response.json()
    assert isinstance(data, list), "Response should be a list."
    assert any(entry["text"] == "old caption" for entry in data), "Old caption should be present."
    assert any(entry["text"] == "new caption" for entry in data), "New caption should be present."
    
    # Test: With a valid 'timestamp', only entries after that time should be returned.
    response = client.get("/captions/speaking", params={"timestamp": current_time})
    data = response.json()
    assert all(entry["timestamp"] > current_time for entry in data), \
        "All returned captions should have a timestamp after the provided value."
    assert any(entry["text"] == "new caption" for entry in data), "New caption should be present."
    assert not any(entry["text"] == "old caption" for entry in data), "Old caption should not be present."
    
    # Edge case: A 'timestamp' later than any caption should return an empty list.
    future_timestamp = current_time + 100
    response = client.get("/captions/speaking", params={"timestamp": future_timestamp})
    assert response.json() == [], "Expected an empty list for a timestamp beyond any published caption."
    
    # Edge case: Invalid timestamp format should result in a validation error (422).
    response = client.get("/captions/speaking", params={"timestamp": "invalid"})
    assert response.status_code == 422, "Invalid timestamp format should result in a 422 error."


def test_get_translated_captions_valid_lang():
    """
    Test the GET /captions/translated endpoint for a valid target language.

    What to test:
    - Sending a request with a valid 'lang' query parameter (e.g. 'es') should return the corresponding translated captions.
    - Using an optional 'timestamp' query parameter filters the results correctly.
    
    Edge cases:
    - Missing 'lang' parameter (should result in a validation error).
    - Invalid or unsupported 'lang' value.
    - A 'timestamp' filter that excludes all data (returns an empty list).
    """
    import time
    current_time = time.time()
    
    # Set up test translated captions for the 'es' language.
    published_data_store.translated["es"] = [
        {"timestamp": current_time - 50, "text": "[es] Old translation"},
        {"timestamp": current_time + 10, "text": "[es] New translation"}
    ]
    
    # Test: Without a timestamp parameter, both translations should be returned.
    response = client.get("/captions/translated", params={"lang": "es"})
    data = response.json()
    assert isinstance(data, list), "Response should be a list."
    assert len(data) == 2, "Both translated captions should be returned."
    
    # Test: With a valid timestamp, only translations after that time should be returned.
    response = client.get("/captions/translated", params={"lang": "es", "timestamp": current_time})
    data = response.json()
    assert len(data) == 1, "Only one caption should be returned after the given timestamp."
    assert "[es] New translation" in data[0]["text"], "The new translation should be returned."
    
    # Edge case: Missing 'lang' parameter should yield a 422 validation error.
    response = client.get("/captions/translated")
    assert response.status_code == 422, "Missing 'lang' parameter should result in a 422 error."
    
    # Edge case: An unsupported language (e.g., 'fr') should return an empty list.
    response = client.get("/captions/translated", params={"lang": "fr"})
    assert response.status_code == 200, "Request with unsupported language should succeed with an empty result."
    assert response.json() == [], "Unsupported language should return an empty list."

def test_get_translated_voice():
    """
    Test the GET /audio/translated-voice endpoint.

    What to test:
    - Request with a valid 'lang' parameter returns base64 encoded TTS audio corresponding to the translated text.
    - Including the optional 'timestamp' query parameter filters the audio entries as expected.
    
    Edge cases:
    - Missing 'lang' parameter should result in an error.
    - 'timestamp' filtering excludes all audio entries.
    - Corrupted audio data (ensure it is properly encoded/decoded).
    """
    import time, base64
    # Set up two valid audio entries for language 'es'
    current_time = time.time()
    published_data_store.audio["es"] = [
        {"timestamp": current_time - 50, "audio": b'\x00\x01\x02'},
        {"timestamp": current_time + 10, "audio": b'\x03\x04\x05'},
    ]

    # Test: Valid request without timestamp - both entries should be returned.
    response = client.get("/audio/translated-voice", params={"lang": "es"})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, list), "Response should be a list."
    assert len(data) == 2, "Both audio entries should be returned."
    expected_audio_1 = base64.b64encode(b'\x00\x01\x02').decode('utf-8')
    expected_audio_2 = base64.b64encode(b'\x03\x04\x05').decode('utf-8')
    assert data[0]["audio"] == expected_audio_1, "The first audio entry was not properly encoded."
    assert data[1]["audio"] == expected_audio_2, "The second audio entry was not properly encoded."

    # Test: Valid request with a timestamp filter that excludes the older entry.
    response = client.get("/audio/translated-voice", params={"lang": "es", "timestamp": current_time})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert len(data) == 1, "Only the newer audio entry should be returned after the timestamp filter."
    assert data[0]["audio"] == expected_audio_2, "The filtered audio entry does not match the expected encoding."

    # Edge case: Missing 'lang' parameter should yield a 422 error.
    response = client.get("/audio/translated-voice")
    assert response.status_code == 422, "Missing 'lang' parameter should result in a 422 error."

    # Edge case: Timestamp filtering excludes all audio entries.
    response = client.get("/audio/translated-voice", params={"lang": "es", "timestamp": current_time + 100})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert data == [], "A timestamp filter excluding all entries should return an empty list."

    # Edge case: Corrupted audio data.
    # Insert an entry with a non-bytes type to simulate corrupted audio data.
    published_data_store.audio["es"] = [
        {"timestamp": current_time + 30, "audio": "not_bytes"}
    ]
    response = client.get("/audio/translated-voice", params={"lang": "es"}) # TODO: define 500
    assert response.status_code == 500, "Corrupted audio data should result in an internal server error."


def test_update_admin_config():
    """
    Test the POST /admin/config endpoint to update administrator configuration.

    What to test:
    - Sending a valid configuration update (for example, updating target languages) returns the new configuration.
    - The global admin configuration is correctly modified.
    
    Edge cases:
    - Sending an empty or malformed JSON dict.
    - Updating with non-existent keys.
    - Partial updates (e.g., only updating one parameter while others remain unchanged).
    """
    # Reset the global admin configuration to a known state.
    admin_config.clear()
    admin_config.update({"target_languages": ["es"]})

    # Test: Valid configuration update.
    response = client.post("/admin/config", json={"target_languages": ["de", "it"]})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert data.get("target_languages") == ["de", "it"], "The target_languages were not updated correctly."

    # Edge case: Sending an empty JSON dict should not change the configuration.
    response = client.post("/admin/config", json={})
    assert response.status_code == 200, f"Expected status code 200 for empty update, got {response.status_code}"
    data = response.json()
    assert data.get("target_languages") == ["de", "it"], "Empty update should not modify the existing configuration."

    # Edge case: Updating with non-existent keys.
    response = client.post("/admin/config", json={"foo": "bar"})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert data.get("foo") == "bar", "Non-existent key 'foo' should be added to the configuration."
    assert data.get("target_languages") == ["de", "it"], "Existing keys should remain unchanged after adding non-existent keys."

    # Edge case: Partial update.
    response = client.post("/admin/config", json={"target_languages": ["ja"]})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert data.get("target_languages") == ["ja"], "Partial update did not modify target_languages correctly."
    assert data.get("foo") == "bar", "Non-updated keys should persist after a partial update."

def test_websocket_endpoint(monkeypatch):
    """
    Test the WebSocket /ws endpoint.

    What to test:
    - A client can successfully connect to the WebSocket endpoint.
    - Sending a valid binary message:
        * The first 4 bytes provide correct metadata length.
        * The metadata JSON includes valid fields (like sampleRate).
        * The binary payload represents valid audio data.
    - The server processes the message and sends back a JSON response with a transcription.
    
    Edge cases:
    - Incomplete or incorrect metadata length (e.g. length field not matching actual metadata).
    - Malformed metadata JSON.
    - Missing audio data in the message.
    - Invalid sampleRate provided in metadata.
    - Abrupt disconnection from the client.
    """
    import json
    import pytest
    # Patch the speech_to_text.convert method to always return a dummy transcription.
    import example_browserclient.server as server_mod
    monkeypatch.setattr(
        server_mod.client.speech_to_text,
        "convert",
        lambda model_id, file: {"text": "websocket dummy transcription"}
    )

    # Valid WebSocket message test
    with client.websocket_connect("/ws") as websocket:
        metadata = {"sampleRate": 44100}
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        metadata_length = len(metadata_bytes)
        message = metadata_length.to_bytes(4, byteorder="little") + metadata_bytes + (b"\x00\x00" * 10)
        websocket.send_bytes(message)
        response_text = websocket.receive_text()
        response_data = json.loads(response_text)
        assert response_data.get("text") == "websocket dummy transcription", "WebSocket transcription did not match expected dummy transcription."
        assert response_data.get("type") == "fullSentence", "WebSocket response type is not 'fullSentence'."

    # Edge case: Incorrect metadata length (declared length not matching actual metadata)
    with client.websocket_connect("/ws") as websocket:
        metadata = {"sampleRate": 44100}
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        wrong_length = len(metadata_bytes) - 2  # intentionally incorrect length
        message = wrong_length.to_bytes(4, byteorder="little") + metadata_bytes + (b"\x00\x00" * 10)
        websocket.send_bytes(message)
        with pytest.raises(Exception):
            websocket.receive_text()

    # Edge case: Malformed metadata JSON
    with client.websocket_connect("/ws") as websocket:
        bad_metadata = b"not_a_json"
        message = len(bad_metadata).to_bytes(4, byteorder="little") + bad_metadata + (b"\x00\x00" * 10)
        websocket.send_bytes(message)
        with pytest.raises(Exception):
            websocket.receive_text()

    # Edge case: Missing audio data in the message
    with client.websocket_connect("/ws") as websocket:
        metadata = {"sampleRate": 44100}
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        message = len(metadata_bytes).to_bytes(4, byteorder="little") + metadata_bytes  # no audio data appended
        websocket.send_bytes(message)
        with pytest.raises(Exception):
            websocket.receive_text()

    # Edge case: Invalid sampleRate provided in metadata (non-integer value)
    with client.websocket_connect("/ws") as websocket:
        metadata = {"sampleRate": "invalid_rate"}
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        message = len(metadata_bytes).to_bytes(4, byteorder="little") + metadata_bytes + (b"\x00\x00" * 10)
        websocket.send_bytes(message)
        with pytest.raises(Exception):
            websocket.receive_text()

    # Edge case: Abrupt disconnection from the client.
    websocket = client.websocket_connect("/ws")
    websocket.close()  # abruptly close the connection
    with pytest.raises(Exception) as exc_info:
        websocket.send_bytes(b"Irrelevant message")
    assert exc_info.value.status_code == 400, "Expected status code 400 for abrupt disconnection."