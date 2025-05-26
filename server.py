import sounddevice as sd
import asyncio
import base64
from enum import Enum
import json
import logging
import os
import time
from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.signal import resample

from elevenlabs import ElevenLabs
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends
import uvicorn
import openai

###############################################################################
# Admin Configuration Management
###############################################################################

CONFIG_FILE = "admin_config.json"

default_config = {
    "elevenlabs_api_key": "",
    "openai_api_key": "",
    "publish_mode": "local-audio-source",  # or "upload"
    "default_source_language": "ENGLISH",
    "target_languages": ["CHINESE", "VIETNAMESE", "THAI"],
    "tts_model": "eleven_multilingual_v2",
    "stt_model": "scribe_v1",
    "admin_token": ""  # admin token is read-only via config endpoint.
}

def load_admin_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                config = json.load(f)
                # Merge missing keys with default_config
                merged_config = {**default_config, **config}
                return merged_config
            except Exception as e:
                logger.error(f"error reading config file: {str(e)}")
                return default_config.copy()
    else:
        return default_config.copy()

def save_admin_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Load the configuration at startup into a global variable.
admin_config = load_admin_config()
save_admin_config(admin_config)  # Ensure the file exists with default values if needed.

###############################################################################
# Set API keys and models from admin_config
###############################################################################

elevenlabs_api_key = admin_config["elevenlabs_api_key"]
openai_api_key = admin_config["openai_api_key"]
openai.api_key = openai_api_key

TTS_MODEL = admin_config.get("tts_model", "eleven_multilingual_v2")
STT_MODEL = admin_config.get("stt_model", "scribe_v1")
PUBLISH_MODE = admin_config.get("publish_mode", "local-audio-source")
DEFAULT_SOURCE_LANGUAGE = admin_config.get("default_source_language", "ENGLISH")

###############################################################################
# Global Configuration
###############################################################################

class Language(Enum):
    ENGLISH = "ENGLISH"
    CHINESE = "CHINESE"
    VIETNAMESE = "VIETNAMESE"
    THAI = "THAI"

supported_languages = [Language.ENGLISH, Language.CHINESE, Language.VIETNAMESE, Language.THAI]

translation_prompt = {
    (Language.ENGLISH, Language.CHINESE): "Translate the following text from English to Chinese: ${text}",
    (Language.ENGLISH, Language.VIETNAMESE): "Translate the following text from English to Vietnamese: ${text}",
    (Language.ENGLISH, Language.THAI): "Translate the following text from English to Thai: ${text}",
    (Language.CHINESE, Language.ENGLISH): "Translate the following text from Chinese to English: ${text}",
    (Language.CHINESE, Language.VIETNAMESE): "Translate the following text from Chinese to Vietnamese: ${text}",
    (Language.CHINESE, Language.THAI): "Translate the following text from Chinese to Thai: ${text}",
    (Language.VIETNAMESE, Language.ENGLISH): "Translate the following text from Vietnamese to English: ${text}",
    (Language.VIETNAMESE, Language.CHINESE): "Translate the following text from Vietnamese to Chinese: ${text}",
    (Language.VIETNAMESE, Language.THAI): "Translate the following text from Vietnamese to Thai: ${text}",
    (Language.THAI, Language.ENGLISH): "Translate the following text from Thai to English: ${text}",
    (Language.THAI, Language.CHINESE): "Translate the following text from Thai to Chinese: ${text}",
    (Language.THAI, Language.VIETNAMESE): "Translate the following text from Thai to Vietnamese: ${text}",
}


###############################################################################
# Initialize Clients and FastAPI
###############################################################################

client = ElevenLabs(api_key=elevenlabs_api_key)
app = FastAPI()
logger = logging.getLogger(__name__)

###############################################################################
# In-Memory Datastore Implementation
###############################################################################

class PublishedDataStore:
    def __init__(self):
        self.speaking = []
        self.translated = defaultdict(list)
        self.audio = defaultdict(list)
        self.lock = asyncio.Lock()

    async def update_speaking(self, timestamp: float, text: str):
        async with self.lock:
            self.speaking.append({"timestamp": timestamp, "text": text})

    async def update_translated(self, lang: str, timestamp: float, text: str):
        async with self.lock:
            self.translated[lang].append({"timestamp": timestamp, "text": text})

    async def update_audio(self, lang: str, timestamp: float, audio: bytes):
        async with self.lock:
            self.audio[lang].append({"timestamp": timestamp, "audio": audio})

    async def get_speaking(self, timestamp: float = None):
        async with self.lock:
            if timestamp is None:
                return self.speaking.copy()
            return [item for item in self.speaking if item["timestamp"] > timestamp]

    async def get_translated(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.translated.get(lang, []).copy()
            if timestamp is None:
                return items
            return [item for item in items if item["timestamp"] > timestamp]

    async def get_audio(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.audio.get(lang, []).copy()
            if timestamp is None:
                return items
            return [item for item in items if item["timestamp"] > timestamp]

published_data_store = PublishedDataStore()

###############################################################################
# Helper Functions
###############################################################################

def decode_and_resample(audio_data: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    # Ensure the length is a multiple of 2 bytes (16-bit PCM)
    if len(audio_data) % 2 != 0:
        audio_data = audio_data[:-1]
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
    resampled_audio = resample(audio_np, num_target_samples)
    return resampled_audio.astype(np.int16).tobytes()

def translate_text(source_text: str, source_lang: str, target_lang: str) -> str:
    if source_lang.upper() == target_lang.upper():
        return source_text

    key = (Language(source_lang.upper()), Language(target_lang.upper()))
    prompt_template = translation_prompt.get(
        key,
        f"Translate the following text from {source_lang} to {target_lang}: ${'{text}'}"
    )
    prompt = prompt_template.replace("${text}", source_text)

    response = client.responses.create(
        model="gpt-4.1", # TODO: use admin_config to control
        input=prompt
    )
    return response.output_text.strip()

def text_to_speech(text: str, target_lang: str) -> bytes:
    response_stream = client.text_to_speech.stream_with_timestamps(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_44100_128",
        text=text,
        model_id=TTS_MODEL,
    )
    chunks = []
    for chunk in response_stream:
        if "audio_base64" in chunk:
            chunks.append(chunk["audio_base64"])
    complete_base64 = "".join(chunks)
    audio_bytes = base64.b64decode(complete_base64)
    return audio_bytes

def get_local_audio_source(record_seconds: int = 5, sample_rate: int = 44100) -> Tuple[bytes, int]:
    """
    Captures audio from the system microphone for `record_seconds` seconds
    at the specified `sample_rate`. Returns the raw audio bytes (as int16 PCM) and the sample rate.

    :param record_seconds: Duration of the audio recording in seconds (default is 5 seconds).
    :param sample_rate: Sample rate for recording (default is 44100 Hz).
    :return: Tuple of (audio data bytes, sample rate)
    """
    print(f"Recording {record_seconds} seconds of audio from the microphone at {sample_rate} Hz...")
    # Record audio; the recording is a numpy array of shape (samples, channels).
    recording = sd.rec(int(record_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is complete.
    
    # Convert the NumPy array to bytes.
    audio_data = recording.tobytes()
    return audio_data, sample_rate

###############################################################################
# Admin Token Security Dependency
###############################################################################

async def verify_admin_token(x_admin_token: str = Header(...)):
    if x_admin_token != admin_config.get("admin_token"):
        raise HTTPException(status_code=403, detail="Invalid admin token.")

###############################################################################
# Background Task for Local Audio Source Mode
###############################################################################

async def local_audio_source_processor():
    """
    A background task that, when in local-audio-source mode,
    periodically fetches local audio and triggers the publish process.
    """
    while True:
        try:
            audio_data, source_sample_rate = get_local_audio_source()
            logger.info("Local audio source acquired")
            
            # Process the audio:
            resampled_chunk = decode_and_resample(audio_data, source_sample_rate, 16000)
            result = client.speech_to_text.convert(model_id=STT_MODEL, file=resampled_chunk)
            transcribed_text = result.get("text", "")
            timestamp = time.time()
            
            # Publish the transcription
            await published_data_store.update_speaking(timestamp, transcribed_text)
            
            # For every target language (skip if source equals target)
            async def process_language(language: str):
                if DEFAULT_SOURCE_LANGUAGE.upper() == language.upper():
                    logger.info(f"Skipping translation for {language} because source and target are the same.")
                    return
                translated_text = translate_text(transcribed_text, DEFAULT_SOURCE_LANGUAGE, language)
                await published_data_store.update_translated(language, timestamp, translated_text)
                tts_audio = text_to_speech(translated_text, language)
                await published_data_store.update_audio(language, timestamp, tts_audio)
            
            tasks = [process_language(lang) for lang in admin_config.get("target_languages", [])]
            await asyncio.gather(*tasks)
            
            logger.info(f"Audio published at timestamp {timestamp}")
        except Exception as e:
            logger.error(f"error during local audio processing: {str(e)}")
        
        current_interval = admin_config.get("poll_interval", 10)
        logger.info(f"Sleeping for {current_interval} seconds...")
        await asyncio.sleep(current_interval)

@app.on_event("startup")
async def startup_event():
    """
    On startup, if the mode is set to local-audio-source,
    start the background task that periodically fetches and publishes local audio.
    """
    if PUBLISH_MODE == "local-audio-source":
        logger.info(f"Starting background task for local audio source mode with poll_interval={admin_config.get('poll_interval', 10)}...")
        # Launch the background task without a fixed interval.
        asyncio.create_task(local_audio_source_processor())
    else:
        logger.info("Server running in upload mode. Use the /publish endpoint to send audio.")
###############################################################################
# HTTP Endpoints
###############################################################################

@app.post("/publish")
async def publish_audio_upload(
    mode: str = Form(...),  # Should be "upload" for this endpoint.
    sourceLanguage: str = Form(...),
    audio_file: UploadFile = File(None),
    sampleRate: int = Form(None)
):
    """
    This endpoint is intended for upload mode only.
    In local-audio-source mode, publishing is done via the background task.
    """
    if mode != "upload":
        raise HTTPException(status_code=400, detail="This endpoint accepts only upload mode requests.")

    if audio_file is None or sampleRate is None:
        raise HTTPException(status_code=400, detail="audio_file and sampleRate are required in upload mode.")
    
    audio_data = await audio_file.read()
    source_sample_rate = sampleRate

    resampled_chunk = decode_and_resample(audio_data, source_sample_rate, 16000)
    result = client.speech_to_text.convert(model_id=STT_MODEL, file=resampled_chunk)
    transcribed_text = result.get("text", "")
    timestamp = time.time()
    await published_data_store.update_speaking(timestamp, transcribed_text)

    async def process_language(language: str):
        if sourceLanguage.upper() == language.upper():
            logger.info(f"Skipping translation for {language} because source and target are the same.")
            return
        translated_text = translate_text(transcribed_text, sourceLanguage, language)
        await published_data_store.update_translated(language, timestamp, translated_text)
        tts_audio = text_to_speech(translated_text, language)
        await published_data_store.update_audio(language, timestamp, tts_audio)

    tasks = [process_language(lang) for lang in admin_config.get("target_languages", [])]
    await asyncio.gather(*tasks)

    return {"timestamp": timestamp, "message": "Published transcription, translation, and TTS audio (upload mode)."}

@app.get("/captions/speaking")
async def get_speaking_captions(timestamp: float = None):
    return await published_data_store.get_speaking(timestamp)

@app.get("/captions/translated")
async def get_translated_captions(lang: str, timestamp: float = None):
    return await published_data_store.get_translated(lang, timestamp)

@app.get("/audio/translated-voice")
async def get_translated_voice(lang: str, timestamp: float = None):
    try:
        audio_entries = await published_data_store.get_audio(lang, timestamp)
        result = []
        for entry in audio_entries:
            encoded_audio = base64.b64encode(entry["audio"]).decode("utf-8")
            result.append({
                "timestamp": entry["timestamp"],
                "audio": encoded_audio
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/admin/config", dependencies=[Depends(verify_admin_token)])
async def update_config(config: dict):
    """
    Update the admin configuration.
    Example payload: {
        "elevenlabs_api_key": "new-key",
        "openai_api_key": "new-key",
        "default_source_language": "CHINESE",
        "target_languages": ["ENGLISH", "THAI"],
        "publish_mode": "upload"
    }
    Note: The "admin_token" value is read-only and cannot be updated via this endpoint.
    """
    global admin_config, PUBLISH_MODE, DEFAULT_SOURCE_LANGUAGE, TTS_MODEL, STT_MODEL
    # Ensure admin_token is not updated
    if "admin_token" in config:
        del config["admin_token"]

    # Update the config values
    admin_config.update(config)
    save_admin_config(admin_config)
    
    # Reinitialize API keys and other constants if they are updated.
    if "openai_api_key" in config:
        openai.api_key = admin_config["openai_api_key"]
    if "elevenlabs_api_key" in config:
        logger.warning("ElevenLabs API key update detected, but client reinitialization is not implemented. "
                    "The new API key will not take effect until server restart.")
        pass
    PUBLISH_MODE = admin_config.get("publish_mode", PUBLISH_MODE)
    DEFAULT_SOURCE_LANGUAGE = admin_config.get("default_source_language", DEFAULT_SOURCE_LANGUAGE)
    TTS_MODEL = admin_config.get("tts_model", TTS_MODEL)
    STT_MODEL = admin_config.get("stt_model", STT_MODEL)
    return admin_config

###############################################################################
# Main Application Runner
###############################################################################

if __name__ == "__main__":
    logger.info("Starting FastAPI server, please wait...")
    uvicorn.run(app, host="localhost", port=9001)
