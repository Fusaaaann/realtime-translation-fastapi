import io
import wave
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
from scipy.io.wavfile import write

from elevenlabs import ElevenLabs
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends
from fastapi.responses import  StreamingResponse
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
    "audio_device_id": None,  # ID of the audio device to use, or None for default
    "admin_token": "",  # admin token is read-only via config endpoint.

    "poll_interval": 10,  # seconds between audio recordings
    "first_record_seconds": 5,  # actual recording duration for better interactivity
    "min_audio_duration": 15  # minimum duration required by the STT API
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

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=openai_api_key)

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
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d:%(funcName)s] - %(message)s',
    level=logging.DEBUG
)
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
        self.speaking_events = []
        self.translated_events = defaultdict(list)
        self.audio_events = defaultdict(list)
        logger.info("Initialized PublishedDataStore")

    async def update_speaking(self, timestamp: float, text: str):
        async with self.lock:
            self.speaking.append({"timestamp": timestamp, "text": text})
            logger.debug(f"Added speaking entry: {text[:50]}... at {timestamp}")
            
            active_events = len(self.speaking_events)
            for event in self.speaking_events:
                event.set()
            self.speaking_events = [e for e in self.speaking_events if not e.is_set()]
            logger.debug(f"Notified {active_events} speaking listeners, {len(self.speaking_events)} remaining")

    async def update_translated(self, lang: str, timestamp: float, text: str):
        async with self.lock:
            self.translated[lang].append({"timestamp": timestamp, "text": text})
            logger.debug(f"Added translation [{lang}]: {text[:50]}... at {timestamp}")
            
            active_events = len(self.translated_events.get(lang, []))
            for event in self.translated_events.get(lang, []):
                event.set()
            self.translated_events[lang] = [e for e in self.translated_events[lang] if not e.is_set()]
            logger.debug(f"Notified {active_events} translation listeners for {lang}, {len(self.translated_events[lang])} remaining")

    async def update_audio(self, lang: str, timestamp: float, audio: bytes):
        async with self.lock:
            audio_size = len(audio)
            self.audio[lang].append({"timestamp": timestamp, "audio": audio})
            logger.debug(f"Added audio [{lang}]: {audio_size} bytes at {timestamp}")
            
            active_events = len(self.audio_events.get(lang, []))
            for event in self.audio_events.get(lang, []):
                event.set()
            self.audio_events[lang] = [e for e in self.audio_events[lang] if not e.is_set()]
            logger.debug(f"Notified {active_events} audio listeners for {lang}, {len(self.audio_events[lang])} remaining")

    async def get_speaking(self, timestamp: float = None):
        async with self.lock:
            if timestamp is None:
                result = self.speaking.copy()
                logger.debug(f"Retrieved all {len(result)} speaking entries")
                return result
            result = [item for item in self.speaking if item["timestamp"] > timestamp]
            logger.debug(f"Retrieved {len(result)} speaking entries after timestamp {timestamp}")
            return result

    async def get_translated(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.translated.get(lang, []).copy()
            if timestamp is None:
                logger.debug(f"Retrieved all {len(items)} translations for {lang}")
                return items
            result = [item for item in items if item["timestamp"] > timestamp]
            logger.debug(f"Retrieved {len(result)} translations for {lang} after timestamp {timestamp}")
            return result

    async def get_audio(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.audio.get(lang, []).copy()
            if timestamp is None:
                logger.debug(f"Retrieved all {len(items)} audio entries for {lang}")
                return items
            result = [item for item in items if item["timestamp"] > timestamp]
            logger.debug(f"Retrieved {len(result)} audio entries for {lang} after timestamp {timestamp}")
            return result

    async def wait_for_speaking_update(self, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.speaking_events.append(event)
            logger.debug(f"Added speaking listener (total: {len(self.speaking_events)})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.debug("Speaking listener notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.debug(f"Speaking listener timed out after {timeout}s")
            return False

    async def wait_for_translated_update(self, lang: str, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.translated_events[lang].append(event)
            logger.debug(f"Added translation listener for {lang} (total: {len(self.translated_events[lang])})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.debug(f"Translation listener for {lang} notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.debug(f"Translation listener for {lang} timed out after {timeout}s")
            return False

    async def wait_for_audio_update(self, lang: str, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.audio_events[lang].append(event)
            logger.debug(f"Added audio listener for {lang} (total: {len(self.audio_events[lang])})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.debug(f"Audio listener for {lang} notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.debug(f"Audio listener for {lang} timed out after {timeout}s")
            return False

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
    return resampled_audio.astype('<i2').astype(np.int16).tobytes()

# TODO: keep conversation for each language until explicit clear for better context understanding and noise word reduction, trim sliding window while retaining 
def translate_text(source_text: str, source_lang: str, target_lang: str) -> str:
    if source_lang.upper() == target_lang.upper():
        return source_text

    key = (Language(source_lang.upper()), Language(target_lang.upper()))
    prompt_template = translation_prompt.get(
        key,
        f"Translate the following text from {source_lang} to {target_lang}: ${'{text}'}"
    )
    prompt = prompt_template.replace("${text}", source_text)

    # Use the initialized OpenAI client
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",  # Use the model specified in admin_config
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


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

# TODO: make this a background job, no sleep
def get_local_audio_source(record_seconds: int = 15, sample_rate: int = 16000, device_id: int = None) -> Tuple[bytes, int]:
    """
    Captures audio from the specified sound device for `record_seconds` seconds
    at the specified `sample_rate`. Returns the raw audio bytes (as int16 PCM) and the sample rate.

    :param record_seconds: Duration of the audio recording in seconds (default is 5 seconds).
    :param sample_rate: Sample rate for recording (default is 16000 Hz).
    :param device_id: ID of the sound device to use. If None, uses the default device.
    :return: Tuple of (audio data bytes, sample rate)
    """
    # List available devices for debugging
    devices = sd.query_devices()
    logger.info(f"Available audio devices: {devices}")
    
    device_info = None
    if device_id is not None:
        try:
            device_info = sd.query_devices(device_id)
            logger.info(f"Using audio device: {device_info['name']} (ID: {device_id})")
        except Exception as e:
            logger.warning(f"Could not use device ID {device_id}: {str(e)}. Falling back to default device.")
    
    # If device_id is None or invalid, use the default input device
    if device_info is None:
        default_device = sd.query_devices(kind='input')
        logger.info(f"Using default input device: {default_device['name']}")
    
    logger.info(f"Recording {record_seconds} seconds of audio from the microphone at {sample_rate} Hz...")
    
    try:
        # Record audio; the recording is a numpy array of shape (samples, channels).
        recording = sd.rec(
            int(record_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device_id
        )
        sd.wait()  # Wait until recording is complete.
        
        # Convert the NumPy array to bytes.
        audio_data = recording.tobytes()
        return audio_data, sample_rate
    
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        # Return empty audio in case of error
        empty_audio = np.zeros((int(record_seconds * sample_rate), 1), dtype=np.int16).tobytes()
        return empty_audio, sample_rate

def pad_audio_with_silence(audio_data: bytes, current_sample_rate: int, 
                          current_duration_seconds: float, 
                          target_duration_seconds: float = 15.0) -> bytes:
    """
    Prepends silence to audio data to reach a target duration.
    This helps meet minimum length requirements for STT APIs while keeping recording time short.
    
    :param audio_data: Raw audio bytes (int16 PCM)
    :param current_sample_rate: Sample rate of the audio in Hz
    :param current_duration_seconds: Current duration of the audio in seconds
    :param target_duration_seconds: Target duration in seconds (default 15s for ElevenLabs STT)
    :return: Padded audio bytes
    """
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    # Reshape if needed (assuming mono audio)
    if len(audio_np.shape) == 1:
        audio_np = audio_np.reshape(-1, 1)
    
    # Calculate how many samples of silence to prepend
    current_samples = len(audio_np)
    target_samples = int(target_duration_seconds * current_sample_rate)
    
    # Only pad if the current duration is less than the target
    if current_samples < target_samples:
        silence_samples = target_samples - current_samples
        logger.info(f"Padding audio with {silence_samples/current_sample_rate:.2f}s of silence to reach {target_duration_seconds}s")
        
        # Create silence array (very quiet, not absolute zero to avoid processing artifacts)
        silence = np.zeros((silence_samples, audio_np.shape[1]), dtype=np.int16)
        
        # Prepend silence to the audio
        padded_audio = np.vstack((silence, audio_np))
        
        return padded_audio.tobytes()
    else:
        logger.info(f"Audio already meets minimum duration ({current_samples/current_sample_rate:.2f}s >= {target_duration_seconds}s)")
        return audio_data


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
    The recording duration is aligned with the poll interval for consistent timing.
    """
    # First-time initialization
    first_run = True
    
    while True:
        try:
            # Use a semaphore to limit concurrent processing
            async with asyncio.Semaphore(3):  # Limit to 3 concurrent audio processing tasks
                # Get configuration parameters
                device_id = admin_config.get("audio_device_id")
                sample_rate = admin_config.get("sample_rate", 44100)
                poll_interval = admin_config.get("poll_interval", 10)
                min_audio_duration = admin_config.get("min_audio_duration", 15)
                
                # On first run, use a shorter recording time for quick startup
                # On subsequent runs, align recording time with poll interval
                if first_run:
                    record_seconds = admin_config.get("first_record_seconds", 5)
                    first_run = False
                    logger.info(f"First recording: using shorter duration of {record_seconds}s")
                else:
                    record_seconds = poll_interval
                    logger.info(f"Subsequent recording: using poll interval duration of {record_seconds}s")
                
                # Record audio
                start_time = time.time()
                # the audio data is obtained in wav
                audio_data, source_sample_rate = get_local_audio_source(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    device_id=device_id
                )
                recording_duration = time.time() - start_time
                logger.info(f"Local audio source acquired in {recording_duration:.2f}s")
                
                # Process the audio
                resampled_chunk = decode_and_resample(audio_data, source_sample_rate, 16000)
                
                # Pad audio if needed to meet minimum duration
                padded_audio = pad_audio_with_silence(
                    audio_data=resampled_chunk,
                    current_sample_rate=16000,  # Resampled rate
                    current_duration_seconds=record_seconds,
                    target_duration_seconds=min_audio_duration
                )
                # Save the padded audio as a WAV file for debugging
                
                timestamp=int(time.time())

                # Create file-like object for API
                
                audio_buffer = io.BytesIO()
                print(type(padded_audio))
                
                write(audio_buffer, 16000, np.frombuffer(padded_audio, dtype=np.int16))  # Write WAV format
                # TODO: decrease time of conversion between np and bytes buffer
                audio_buffer.seek(0)  # Rewind to beginning
                
                # Send to STT API
                stt_start_time = time.time()
                result = client.speech_to_text.convert(
                    model_id=STT_MODEL, 
                    file=(f"input-file-{timestamp}.wav", audio_buffer, "audio/wav"),
                    # language_code="",
                    timestamps_granularity="word",
                )
                stt_duration = time.time() - stt_start_time
                logger.info(f"Speech-to-text completed in {stt_duration:.2f}s")
                
                transcribed_text = result.text
                timestamp = time.time()
                
                # Only process if there's actual transcribed text
                if transcribed_text.strip():
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
                    
                    # Process languages in parallel but with a limit
                    semaphore = asyncio.Semaphore(2)  # Process at most 2 languages at once
                    
                    async def bounded_process_language(language: str):
                        async with semaphore:
                            await process_language(language)
                    
                    tasks = [bounded_process_language(lang) for lang in admin_config.get("target_languages", [])]
                    await asyncio.gather(*tasks)
                    
                    logger.info(f"Audio published at timestamp {timestamp}")
                else:
                    logger.info("No speech detected in the recording, skipping processing")
                
                # Calculate remaining time in the poll interval
                elapsed_time = time.time() - start_time
                remaining_time = max(0, poll_interval - elapsed_time)
                
                if remaining_time > 0:
                    logger.info(f"Completed processing in {elapsed_time:.2f}s. Waiting {remaining_time:.2f}s until next recording.")
                    await asyncio.sleep(remaining_time)
                else:
                    logger.warning(f"Processing took {elapsed_time:.2f}s, which exceeds the poll interval of {poll_interval}s. Starting next recording immediately.")
        
        except Exception as e:
            logger.error(f"Error during local audio processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Wait the poll interval before trying again after an error
            await asyncio.sleep(admin_config.get("poll_interval", 10))

@app.on_event("startup")
async def startup_event():
    """
    On startup, if the mode is set to local-audio-source,
    start the background task that periodically fetches and publishes local audio.
    """
    logger.info("server startup")
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
    transcribed_text = result.text
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

@app.get("/captions/speaking/stream")
async def stream_speaking_captions(timestamp: float = None):
    """Stream speaking captions as they become available"""
    async def event_generator():
        current_timestamp = timestamp if timestamp is not None else 0
        while True:
            # Get any new captions since the last timestamp
            captions = await published_data_store.get_speaking(current_timestamp)
            
            if captions:
                # Update the timestamp to the latest one
                current_timestamp = max(caption["timestamp"] for caption in captions)
                # Yield the new captions as a JSON string with SSE format
                yield f"data: {json.dumps(captions)}\n\n"
            else:
                # Wait for new data
                update_occurred = await published_data_store.wait_for_speaking_update()
                if not update_occurred:
                    # Send a keep-alive message if no updates
                    yield f"data: {json.dumps([])}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/captions/translated/stream")
async def stream_translated_captions(lang: str, timestamp: float = None):
    """Stream translated captions as they become available"""
    async def event_generator():
        current_timestamp = timestamp if timestamp is not None else 0
        while True:
            # Get any new translations since the last timestamp
            translations = await published_data_store.get_translated(lang, current_timestamp)
            
            if translations:
                # Update the timestamp to the latest one
                current_timestamp = max(translation["timestamp"] for translation in translations)
                # Yield the new translations as a JSON string with SSE format
                yield f"data: {json.dumps(translations)}\n\n"
            else:
                # Wait for new data
                update_occurred = await published_data_store.wait_for_translated_update(lang)
                if not update_occurred:
                    # Send a keep-alive message if no updates
                    yield f"data: {json.dumps([])}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/audio/translated-voice/stream")
async def stream_translated_voice(lang: str, timestamp: float = None):
    """Stream translated audio as it becomes available"""
    async def event_generator():
        current_timestamp = timestamp if timestamp is not None else 0
        while True:
            # Get any new audio since the last timestamp
            audio_entries = await published_data_store.get_audio(lang, current_timestamp)
            
            if audio_entries:
                # Update the timestamp to the latest one
                current_timestamp = max(entry["timestamp"] for entry in audio_entries)
                
                # Process audio entries
                result = []
                for entry in audio_entries:
                    encoded_audio = base64.b64encode(entry["audio"]).decode("utf-8")
                    result.append({
                        "timestamp": entry["timestamp"],
                        "audio": encoded_audio
                    })
                
                # Yield the new audio as a JSON string with SSE format
                yield f"data: {json.dumps(result)}\n\n"
            else:
                # Wait for new data
                update_occurred = await published_data_store.wait_for_audio_update(lang)
                if not update_occurred:
                    # Send a keep-alive message if no updates
                    yield f"data: {json.dumps([])}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/captions/speaking")
async def get_speaking_captions(timestamp: float = None):
    """Get speaking captions with chunked response to prevent starvation"""
    async def generate_chunks():
        # Get data in chunks to prevent blocking
        captions = await published_data_store.get_speaking(timestamp)
        chunk_size = 5  # Adjust based on expected data size
        
        for i in range(0, len(captions), chunk_size):
            chunk = captions[i:i+chunk_size]
            yield json.dumps(chunk).encode() + b"\n"
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
    
    return StreamingResponse(
        generate_chunks(),
        media_type="application/json",
        headers={"Content-Disposition": "inline"}
    )

@app.get("/captions/translated")
async def get_translated_captions(lang: str, timestamp: float = None):
    """Get translated captions with chunked response to prevent starvation"""
    async def generate_chunks():
        # Get data in chunks to prevent blocking
        translations = await published_data_store.get_translated(lang, timestamp)
        chunk_size = 5  # Adjust based on expected data size
        
        for i in range(0, len(translations), chunk_size):
            chunk = translations[i:i+chunk_size]
            yield json.dumps(chunk).encode() + b"\n"
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
    
    return StreamingResponse(
        generate_chunks(),
        media_type="application/json",
        headers={"Content-Disposition": "inline"}
    )

@app.get("/audio/translated-voice")
async def get_translated_voice(lang: str, timestamp: float = None):
    """Get translated voice with chunked response to prevent starvation"""
    try:
        async def generate_chunks():
            # Get data in chunks to prevent blocking
            audio_entries = await published_data_store.get_audio(lang, timestamp)
            chunk_size = 2  # Smaller chunk size for audio due to size
            
            for i in range(0, len(audio_entries), chunk_size):
                chunk = audio_entries[i:i+chunk_size]
                result = []
                for entry in chunk:
                    encoded_audio = base64.b64encode(entry["audio"]).decode("utf-8")
                    result.append({
                        "timestamp": entry["timestamp"],
                        "audio": encoded_audio
                    })
                yield json.dumps(result).encode() + b"\n"
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.02)  # Slightly longer delay for audio
        
        return StreamingResponse(
            generate_chunks(),
            media_type="application/json",
            headers={"Content-Disposition": "inline"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/admin/config", dependencies=[Depends(verify_admin_token)])
async def update_config():
    return admin_config




###############################################################################
# Admin Helpers
###############################################################################
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
    global admin_config, PUBLISH_MODE, DEFAULT_SOURCE_LANGUAGE, TTS_MODEL, STT_MODEL, openai_client
    # Ensure admin_token is not updated
    if "admin_token" in config:
        del config["admin_token"]

    # Update the config values
    admin_config.update(config)
    save_admin_config(admin_config)
    
    # Reinitialize API keys and other constants if they are updated.
    if "openai_api_key" in config:
        openai_client = openai.OpenAI(api_key=admin_config["openai_api_key"])
        
    if "elevenlabs_api_key" in config:
        logger.warning("ElevenLabs API key update detected, but client reinitialization is not implemented. "
                    "The new API key will not take effect until server restart.")
        pass
    PUBLISH_MODE = admin_config.get("publish_mode", PUBLISH_MODE)
    DEFAULT_SOURCE_LANGUAGE = admin_config.get("default_source_language", DEFAULT_SOURCE_LANGUAGE)
    TTS_MODEL = admin_config.get("tts_model", TTS_MODEL)
    STT_MODEL = admin_config.get("stt_model", STT_MODEL)
    return admin_config

@app.get("/admin/audio-devices", dependencies=[Depends(verify_admin_token)])
async def list_audio_devices():
    """
    List all available audio input devices.
    """
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                input_devices.append({
                    "id": i,
                    "name": device.get('name', f"Device {i}"),
                    "channels": device.get('max_input_channels'),
                    "default_samplerate": device.get('default_samplerate'),
                    "is_default": device.get('default_input', False)
                })
        
        return {
            "devices": input_devices,
            "current_device_id": admin_config.get("audio_device_id")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing audio devices: {str(e)}")


###############################################################################
# Main Application Runner
###############################################################################

if __name__ == "__main__":
    logger.info("Starting FastAPI server, please wait...")
    uvicorn.run(app, host="localhost", port=9001)
