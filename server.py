import io
from contextlib import contextmanager
import secrets
from scipy import signal
import threading
import wave
import asyncio
import base64
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import resample

from elevenlabs import ElevenLabs
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends, Request, status, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import sounddevice as sd
import uvicorn
import openai

from i18n import DEFAULT_MESSAGES, LOCALIZED_MESSAGES, Language, translation_prompt

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
    "min_audio_duration": 15,  # minimum duration required by the STT API
}

API_KEY = secrets.token_urlsafe(12)[:16]


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
MAIN_CONSUMER_ID = "translation-consumer"
###############################################################################
# Global Configuration
###############################################################################


###############################################################################
# Initialize Clients and FastAPI
###############################################################################

client = ElevenLabs(api_key=elevenlabs_api_key)
app = FastAPI()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d:%(funcName)s] - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
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
        self.max_history_items = 100
        logger.info("Initialized PublishedDataStore")

    async def update_speaking(self, timestamp: float, text: str):
        async with self.lock:
            self.speaking.append({"timestamp": timestamp, "text": text})
            logger.info(f"Added speaking entry: {text[:50]}... at {timestamp}")
            if len(self.speaking) > self.max_history_items:
                self.speaking = self.speaking[-self.max_history_items :]

            active_events = len(self.speaking_events)
            for event in self.speaking_events:
                event.set()
            self.speaking_events = [e for e in self.speaking_events if not e.is_set()]
            logger.info(f"Notified {active_events} speaking listeners, {len(self.speaking_events)} remaining")

    async def update_translated(self, lang: str, timestamp: float, text: str):
        async with self.lock:
            self.translated[lang].append({"timestamp": timestamp, "text": text})
            logger.info(f"Added translation [{lang}]: {text[:50]}... at {timestamp}")

            active_events = len(self.translated_events.get(lang, []))
            for event in self.translated_events.get(lang, []):
                event.set()
            self.translated_events[lang] = [e for e in self.translated_events[lang] if not e.is_set()]
            logger.info(f"Notified {active_events} translation listeners for {lang}, {len(self.translated_events[lang])} remaining")

    async def update_audio(self, lang: str, timestamp: float, audio: bytes):
        async with self.lock:
            audio_size = len(audio)
            self.audio[lang].append({"timestamp": timestamp, "audio": audio})
            logger.info(f"Added audio [{lang}]: {audio_size} bytes at {timestamp}")

            active_events = len(self.audio_events.get(lang, []))
            for event in self.audio_events.get(lang, []):
                event.set()
            self.audio_events[lang] = [e for e in self.audio_events[lang] if not e.is_set()]
            logger.info(f"Notified {active_events} audio listeners for {lang}, {len(self.audio_events[lang])} remaining")

    async def get_speaking(self, timestamp: float = None):
        async with self.lock:
            if timestamp is None:
                result = self.speaking.copy()
                logger.info(f"Retrieved all {len(result)} speaking entries")
                return result
            result = [item for item in self.speaking if item["timestamp"] > timestamp]
            logger.info(f"Retrieved {len(result)} speaking entries after timestamp {timestamp}")
            return result

    async def get_translated(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.translated.get(lang, []).copy()
            if timestamp is None:
                logger.info(f"Retrieved all {len(items)} translations for {lang}")
                return items
            result = [item for item in items if item["timestamp"] > timestamp]
            logger.info(f"Retrieved {len(result)} translations for {lang} after timestamp {timestamp}")
            return result

    async def get_audio(self, lang: str, timestamp: float = None):
        async with self.lock:
            items = self.audio.get(lang, []).copy()
            if timestamp is None:
                logger.info(f"Retrieved all {len(items)} audio entries for {lang}")
                return items
            result = [item for item in items if item["timestamp"] > timestamp]
            logger.info(f"Retrieved {len(result)} audio entries for {lang} after timestamp {timestamp}")
            return result

    async def wait_for_speaking_update(self, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.speaking_events.append(event)
            logger.info(f"Added speaking listener (total: {len(self.speaking_events)})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.info("Speaking listener notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.info(f"Speaking listener timed out after {timeout}s")
            return False

    async def wait_for_translated_update(self, lang: str, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.translated_events[lang].append(event)
            logger.info(f"Added translation listener for {lang} (total: {len(self.translated_events[lang])})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.info(f"Translation listener for {lang} notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.info(f"Translation listener for {lang} timed out after {timeout}s")
            return False

    async def wait_for_audio_update(self, lang: str, timeout=30):
        event = asyncio.Event()
        async with self.lock:
            self.audio_events[lang].append(event)
            logger.info(f"Added audio listener for {lang} (total: {len(self.audio_events[lang])})")
        try:
            await asyncio.wait_for(event.wait(), timeout)
            logger.info(f"Audio listener for {lang} notified successfully")
            return True
        except asyncio.TimeoutError:
            logger.info(f"Audio listener for {lang} timed out after {timeout}s")
            return False


published_data_store = PublishedDataStore()

###############################################################################
# Helper Functions
###############################################################################


def resample_audio(audio_np: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """
    Resample audio data from original sample rate to target sample rate.

    :param audio_np: Audio data as numpy array
    :param original_sample_rate: Original sample rate of the audio
    :param target_sample_rate: Target sample rate
    :return: Resampled audio as numpy array
    """
    # Ensure the audio is properly shaped
    if len(audio_np.shape) == 1:
        audio_np = audio_np.reshape(-1, 1)

    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)

    # Resample each channel separately if multi-channel
    if audio_np.shape[1] > 1:
        resampled_channels = []
        for channel in range(audio_np.shape[1]):
            channel_data = audio_np[:, channel]
            resampled_channel = resample(channel_data, num_target_samples)
            resampled_channels.append(resampled_channel)
        resampled_audio = np.column_stack(resampled_channels)
    else:
        # Single channel resampling
        resampled_audio = resample(audio_np.flatten(), num_target_samples).reshape(-1, 1)

    return resampled_audio.astype(np.int16)


def translate_text(source_text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text from source language to target language, maintaining conversation context
    for better quality and noise word reduction.
    """
    # Initialize conversation history if it doesn't exist
    if not hasattr(translate_text, "conversation_history"):
        translate_text.conversation_history = defaultdict(list)

    # Create a key for this language pair
    lang_pair_key = f"{source_lang.upper()}-{target_lang.upper()}"

    # If same language, return original text
    if source_lang.upper() == target_lang.upper():
        return source_text

    # Get the prompt template for this language pair
    key = (Language(source_lang.upper()), Language(target_lang.upper()))
    prompt_template = translation_prompt.get(key, f"Translate the following text from {source_lang} to {target_lang}: ${{text}}")

    # Construct messages for the API call
    messages = [{"role": "system", "content": "You are a professional translator. Maintain consistency with previous translations."}]

    # Add conversation history as context in the messages
    if translate_text.conversation_history[lang_pair_key]:
        history_context = "Previous conversation context:\n"
        for i, prev_text in enumerate(translate_text.conversation_history[lang_pair_key][-5:]):
            history_context += f"[{i+1}] {prev_text}\n"
        messages.append({"role": "user", "content": history_context})
        messages.append({"role": "assistant", "content": "I'll maintain consistency with these previous translations."})

    # Add the current translation request
    translation_request = prompt_template.replace("${text}", source_text)
    messages.append({"role": "user", "content": translation_request})

    response = openai_client.chat.completions.create(model="gpt-4.1", messages=messages)

    translated_text = response.choices[0].message.content.strip()

    # Update conversation history (keep last 10 exchanges)
    translate_text.conversation_history[lang_pair_key].append(f"Source: {source_text}\nTranslation: {translated_text}")
    if len(translate_text.conversation_history[lang_pair_key]) > 10:
        translate_text.conversation_history[lang_pair_key].pop(0)

    return translated_text


def text_to_speech(text: str, target_lang: str) -> bytes:
    response_stream = client.text_to_speech.stream_with_timestamps(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_44100_128",
        text=text,
        model_id=TTS_MODEL,
        # previous_text="", # TODO: support previous_text by storing previous translated clip
    )
    chunks = []
    for chunk in response_stream:
        if "audio_base64" in chunk:
            chunks.append(chunk["audio_base64"])
    complete_base64 = "".join(chunks)
    audio_bytes = base64.b64decode(complete_base64)
    return audio_bytes


class ContinuousAudioRecorder:
    """
    A service that continuously records audio in the background and maintains
    a buffer of recent audio data that can be accessed on demand.
    """

    def __init__(self, sample_rate=16000, channels=1, device_id=None, buffer_seconds=30):
        """
        Initialize the continuous audio recorder.

        :param sample_rate: Sample rate for recording (Hz)
        :param channels: Number of audio channels (1 for mono, 2 for stereo)
        :param device_id: ID of the audio device to use, or None for default
        :param buffer_seconds: Maximum duration of audio to keep in the buffer (seconds)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.buffer_seconds = buffer_seconds

        # Calculate buffer size in frames
        self.buffer_size = int(buffer_seconds * sample_rate)

        # Create a circular buffer to store audio frames
        self.buffer = np.zeros((self.buffer_size, channels), dtype=np.int16)
        self.buffer_index = 0  # Current position in the buffer

        # Thread synchronization and resource management
        self.lock = threading.RLock()
        self._state_lock = threading.RLock()  # Separate lock for state management
        self._running = False
        self._starting = False
        self._stopping = False
        self.thread = None
        self.stream = None

        # Audio update tracking
        self.audio_update_event = None  # Will be set when event loop is available
        self.current_sequence = 0
        self.last_processed_sequences = {}  # Dict to track processing by consumer
        self.audio_update_queue = None  # Will be initialized when needed
        self.last_update_timestamp = time.time()

        # Consumer management
        self.consumers = set()

        logger.info(f"Initialized ContinuousAudioRecorder with {buffer_seconds}s buffer at {sample_rate}Hz")

    @property
    def running(self) -> bool:
        """Thread-safe property to check if recorder is running."""
        with self._state_lock:
            return self._running

    @contextmanager
    def _resource_lock(self):
        """Context manager for resource operations."""
        with self._state_lock:
            yield

    def start(self) -> bool:
        """
        Start the continuous recording thread.

        :return: True if started successfully, False if already running or failed to start
        """
        with self._resource_lock():
            if self._running:
                logger.warning("Continuous audio recorder is already running")
                return False

            if self._starting:
                logger.warning("Continuous audio recorder is already starting")
                return False

            if self._stopping:
                logger.warning("Continuous audio recorder is currently stopping, please wait")
                return False

            try:
                self._starting = True
                self._initialize_async_components()

                # Start the recording thread
                self.thread = threading.Thread(target=self._record_thread, daemon=True)
                self.thread.start()

                # Wait longer for complex audio setup - increased from 0.1 to 0.5
                time.sleep(0.5)

                if self.thread.is_alive():
                    self._running = True
                    logger.info("Started continuous audio recording thread")
                    return True
                else:
                    logger.error("Failed to start audio recording thread")
                    return False

            except Exception as e:
                logger.error(f"Error starting continuous audio recorder: {e}")
                return False
            finally:
                self._starting = False

    def stop(self) -> bool:
        """
        Stop the continuous recording thread and clean up resources.

        :return: True if stopped successfully, False if not running or failed to stop
        """
        with self._resource_lock():
            if not self._running:
                logger.warning("Continuous audio recorder is not running")
                return False

            if self._stopping:
                logger.warning("Continuous audio recorder is already stopping")
                return False

            try:
                self._stopping = True
                self._running = False

                # Close the audio stream first
                if self.stream:
                    try:
                        self.stream.close()
                        self.stream = None
                    except Exception as e:
                        logger.warning(f"Error closing audio stream: {e}")

                # Wait for thread to finish
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=3.0)
                    if self.thread.is_alive():
                        logger.warning("Audio recording thread did not terminate cleanly within timeout")
                        return False

                self.thread = None

                # Clean up async resources
                self._cleanup_async_components()

                logger.info("Stopped continuous audio recording thread")
                return True

            except Exception as e:
                logger.error(f"Error stopping continuous audio recorder: {e}")
                return False
            finally:
                self._stopping = False

    def restart(self) -> bool:
        """
        Restart the continuous recording with current settings.

        :return: True if restarted successfully
        """
        logger.info("Restarting continuous audio recorder")
        if not self.stop():
            logger.error("Failed to stop recorder for restart")
            return False

        # Wait a moment between stop and start
        time.sleep(0.2)

        return self.start()

    def _initialize_async_components(self):
        """Initialize async components if event loop is available."""
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                if self.audio_update_event is None:
                    self.audio_update_event = asyncio.Event()
                if self.audio_update_queue is None:
                    self.audio_update_queue = asyncio.Queue(maxsize=100)
        except RuntimeError:
            # No event loop available, will initialize later
            logger.debug("No event loop available for async components initialization")

    def _cleanup_async_components(self):
        """Clean up async components."""
        # Clear consumers
        self.consumers.clear()
        self.last_processed_sequences.clear()

        # Note: We don't explicitly close asyncio objects as they'll be garbage collected
        # and we might not have access to the event loop during cleanup

    def _record_thread(self):
        """Background thread that continuously records audio into the circular buffer."""
        try:
            logger.info("Starting audio recording thread initialization")
            # List available devices for debugging
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {len(devices)} devices found")

            self._init_noise_filter()
            device_info = None
            if self.device_id is not None:
                try:
                    device_info = sd.query_devices(self.device_id)
                    logger.info(f"Using audio device: {device_info['name']} (ID: {self.device_id})")
                except Exception as e:
                    logger.warning(f"Could not use device ID {self.device_id}: {str(e)}. Falling back to default device.")

            # If device_id is None or invalid, use the default input device
            if device_info is None:
                default_device = sd.query_devices(kind="input")
                logger.info(f"Using default input device: {default_device['name']}")

            # Wait for the _running flag to be set by start() method
            startup_timeout = 10.0  # 10 second timeout for audio setup
            startup_start = time.time()
            while not self._running and not self._stopping:
                if time.time() - startup_start > startup_timeout:
                    logger.error("Audio recording thread startup timeout - _running flag not set")
                    return
                time.sleep(0.01)

            if self._stopping:
                logger.info("Audio recording thread stopping before stream creation")
                return

            # Pre-allocate variables to avoid allocations in callback
            overflow_count = 0
            last_overflow_log = 0

            # Define optimized callback function for the stream
            def callback(indata, frames, time_info, status):
                nonlocal overflow_count, last_overflow_log

                # Handle status warnings with rate limiting
                if status:
                    current_time = time.time()
                    if "input overflow" in str(status).lower():
                        overflow_count += 1
                        # Only log every 10 overflows or every 5 seconds
                        if overflow_count % 10 == 1 or (current_time - last_overflow_log) > 5.0:
                            logger.warning(f"Audio input overflow detected (count: {overflow_count})")
                            last_overflow_log = current_time
                    else:
                        logger.warning(f"Audio callback status: {status}")

                # Quick exit check without any processing
                if not self._running:
                    return

                try:
                    # Apply noise filtering - do this as efficiently as possible
                    filtered_audio = self._apply_noise_filter(indata)

                    # Use non-blocking lock with immediate fallback
                    if self.lock.acquire(blocking=False):
                        try:
                            # Double-check running state
                            if not self._running:
                                return

                            # Optimized circular buffer write
                            start_idx = self.buffer_index
                            end_idx = start_idx + frames

                            if end_idx <= self.buffer_size:
                                # Simple case: no wraparound
                                self.buffer[start_idx:end_idx] = filtered_audio
                                self.buffer_index = end_idx % self.buffer_size
                            else:
                                # Wraparound case
                                first_chunk = self.buffer_size - start_idx
                                self.buffer[start_idx:] = filtered_audio[:first_chunk]
                                self.buffer[: frames - first_chunk] = filtered_audio[first_chunk:]
                                self.buffer_index = frames - first_chunk

                        finally:
                            self.lock.release()
                    else:
                        # Don't log in callback - just drop this chunk silently
                        # Logging in audio callback can cause more delays
                        pass

                except Exception:
                    # Minimal error handling in callback
                    pass

                # Simplified async signaling - don't wait for result
                if self.audio_update_event is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop and not loop.is_closed():
                            # Fire and forget - don't add callback handlers
                            asyncio.run_coroutine_threadsafe(self._signal_new_audio(), loop)
                    except (RuntimeError, AttributeError):
                        pass

            # Calculate optimal blocksize based on sample rate
            # Larger blocksize = fewer callbacks = less overhead
            optimal_blocksize = int(self.sample_rate * 0.2)  # 200ms chunks instead of 100ms

            # Start the input stream with optimized parameters
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=callback,
                blocksize=optimal_blocksize,  # Increased from 100ms to 200ms
                device=self.device_id,
                latency="low",  # Request low latency mode
                extra_settings=sd.AsioSettings(channel_selectors=[0]) if hasattr(sd, "AsioSettings") else None,
            ) as stream:
                self.stream = stream
                logger.info(f"Started continuous audio stream at {self.sample_rate}Hz with {optimal_blocksize} sample blocks")

                # Keep the stream running until self._running becomes False
                while self._running:
                    sd.sleep(200)  # Increased sleep time to reduce thread overhead

        except Exception as e:
            logger.error(f"Error in audio recording thread: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            with self._resource_lock():
                self._running = False
                self.stream = None
            logger.info("Audio recording thread finished")

    async def _signal_new_audio(self):
        """Signal that new audio is available with sequence tracking."""
        self.current_sequence += 1
        timestamp = time.time()

        # Create audio update object
        audio_update = {
            "sequence": self.current_sequence,
            "timestamp": timestamp,
            "buffer_index": self.buffer_index,
        }

        # Add to queue for guaranteed delivery
        if self.audio_update_queue:
            try:
                await self.audio_update_queue.put(audio_update)
            except asyncio.QueueFull:
                logger.warning("Audio update queue full - dropping oldest update")
                try:
                    self.audio_update_queue.get_nowait()
                    await self.audio_update_queue.put(audio_update)
                except Exception as e:
                    logger.error(f"Error managing audio queue: {e}")

        # Set event to notify waiting consumers
        if self.audio_update_event:
            self.audio_update_event.set()

        self.last_update_timestamp = timestamp

    async def register_consumer(self, consumer_id: str):
        """Register a new audio consumer."""
        lock_timeout = 0.1  # Shorter timeout for real-time audio
        try:
            if self.lock.acquire(timeout=lock_timeout):
                try:
                    self.consumers.add(consumer_id)
                    self.last_processed_sequences[consumer_id] = 0
                finally:
                    self.lock.release()
                logger.info(f"Registered audio consumer: {consumer_id}")
            else:
                logger.warning(f"Failed to register consumer {consumer_id} - lock timeout")
        except Exception as e:
            logger.error(f"Error registering consumer {consumer_id}: {e}")

    async def unregister_consumer(self, consumer_id: str):
        """Unregister an audio consumer."""
        lock_timeout = 0.1
        try:
            if self.lock.acquire(timeout=lock_timeout):
                try:
                    self.consumers.discard(consumer_id)
                    self.last_processed_sequences.pop(consumer_id, None)
                finally:
                    self.lock.release()
                logger.info(f"Unregistered audio consumer: {consumer_id}")
            else:
                logger.warning(f"Failed to unregister consumer {consumer_id} - lock timeout")
        except Exception as e:
            logger.error(f"Error unregistering consumer {consumer_id}: {e}")

    async def wait_for_new_audio(self, consumer_id: str, timeout=None):
        """
        Wait for new audio with guaranteed delivery.

        :param consumer_id: Unique identifier for the consumer
        :param timeout: Maximum time to wait in seconds
        :return: Audio update info if available, None if timeout
        """
        # Initialize async components if needed
        if self.audio_update_event is None or self.audio_update_queue is None:
            self._initialize_async_components()

        try:
            if consumer_id not in self.last_processed_sequences:
                await self.register_consumer(consumer_id)

            # Check queue first for any missed updates
            while not self.audio_update_queue.empty():
                update = self.audio_update_queue.get_nowait()
                if update["sequence"] > self.last_processed_sequences[consumer_id]:
                    self.last_processed_sequences[consumer_id] = update["sequence"]
                    return update

            # Wait for new updates if queue is empty
            if timeout is not None:
                await asyncio.wait_for(self.audio_update_event.wait(), timeout)
            else:
                await self.audio_update_event.wait()

            # Clear event only if all consumers have processed
            with self.lock:
                if all(seq >= self.current_sequence for seq in self.last_processed_sequences.values()):
                    self.audio_update_event.clear()

            # Get latest update
            if not self.audio_update_queue.empty():
                update = await self.audio_update_queue.get()
                self.last_processed_sequences[consumer_id] = update["sequence"]
                return update

            return None

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error waiting for new audio: {e}")
            return None

    async def get_audio_updates(self, consumer_id: str, batch_size=10):
        """
        Get multiple audio updates at once for batch processing.

        :param consumer_id: Unique identifier for the consumer
        :param batch_size: Maximum number of updates to retrieve
        :return: List of audio updates
        """
        updates = []
        try:
            if consumer_id not in self.last_processed_sequences:
                await self.register_consumer(consumer_id)

            last_seq = self.last_processed_sequences[consumer_id]

            # Collect available updates up to batch_size
            while len(updates) < batch_size and not self.audio_update_queue.empty():
                update = await self.audio_update_queue.get()
                if update["sequence"] > last_seq:
                    updates.append(update)
                    last_seq = update["sequence"]

            if updates:
                self.last_processed_sequences[consumer_id] = updates[-1]["sequence"]

            return updates

        except Exception as e:
            logger.error(f"Error getting audio updates: {e}")
            return []

    def get_audio(self, duration_seconds: float, update_info: dict = None) -> np.ndarray:
        """
        Get the most recent audio of the specified duration.

        :param duration_seconds: Duration of audio to retrieve in seconds
        :param update_info: Optional update info for consistency validation
        :return: Numpy array of audio data
        """
        frames = int(duration_seconds * self.sample_rate)

        # Use timeout to prevent indefinite blocking in real-time audio
        lock_timeout = 0.5  # 500ms timeout for audio access
        try:
            if not self.lock.acquire(timeout=lock_timeout):
                logger.warning(f"Failed to acquire audio lock within {lock_timeout}s timeout")
                # Return silence as fallback for real-time applications
                return np.zeros((frames, self.channels), dtype=np.int16)

            try:
                if not self._running:
                    logger.warning("Attempting to get audio from stopped recorder")
                    return np.zeros((frames, self.channels), dtype=np.int16)

                # Validate update info if provided
                if update_info and update_info["buffer_index"] != self.buffer_index:
                    logger.warning("Buffer index mismatch - audio may be inconsistent")

                # Ensure we don't request more frames than we have
                if frames > self.buffer_size:
                    logger.warning(f"Requested {duration_seconds}s of audio but buffer only holds {self.buffer_size/self.sample_rate}s. Truncating.")
                    frames = self.buffer_size

                # Calculate the start index for the requested duration
                start_index = (self.buffer_index - frames) % self.buffer_size

                # Create a new array for the result
                result = np.zeros((frames, self.channels), dtype=np.int16)

                # Copy data from the circular buffer to the result array
                if start_index < self.buffer_index:
                    result[:] = self.buffer[start_index : self.buffer_index]
                else:
                    frames_from_end = self.buffer_size - start_index
                    result[:frames_from_end] = self.buffer[start_index:]
                    result[frames_from_end:] = self.buffer[: self.buffer_index]

                return result

            finally:
                self.lock.release()

        except Exception as e:
            logger.error(f"Error in get_audio: {e}")
            return np.zeros((frames, self.channels), dtype=np.int16)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the audio recorder.

        :return: Dictionary containing status information
        """
        with self._resource_lock():
            return {
                "running": self._running,
                "starting": self._starting,
                "stopping": self._stopping,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "device_id": self.device_id,
                "buffer_seconds": self.buffer_seconds,
                "buffer_size": self.buffer_size,
                "current_sequence": self.current_sequence,
                "consumers": list(self.consumers),
                "last_update_timestamp": self.last_update_timestamp,
                "thread_alive": self.thread.is_alive() if self.thread else False,
                "stream_active": self.stream is not None,
            }

    def update_settings(self, sample_rate=None, device_id=None, buffer_seconds=None):
        """
        Update recorder settings. Requires restart to take effect.

        :param sample_rate: New sample rate
        :param device_id: New device ID
        :param buffer_seconds: New buffer duration
        :return: True if settings were updated
        """
        with self._resource_lock():
            settings_changed = False

            if sample_rate is not None and sample_rate != self.sample_rate:
                self.sample_rate = sample_rate
                settings_changed = True
                logger.info(f"Updated sample rate to {sample_rate}Hz")

            if device_id != self.device_id:  # Allow None values
                self.device_id = device_id
                settings_changed = True
                logger.info(f"Updated device ID to {device_id}")

            if buffer_seconds is not None and buffer_seconds != self.buffer_seconds:
                self.buffer_seconds = buffer_seconds
                self.buffer_size = int(buffer_seconds * self.sample_rate)
                # Recreate buffer with new size
                self.buffer = np.zeros((self.buffer_size, self.channels), dtype=np.int16)
                self.buffer_index = 0
                settings_changed = True
                logger.info(f"Updated buffer duration to {buffer_seconds}s")

            return settings_changed

    def _init_noise_filter(self):
        """Initialize noise filtering components."""
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        hp_cutoff = 80  # Hz
        self.hp_b, self.hp_a = signal.butter(2, hp_cutoff / nyquist, btype="high")
        self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)

        # Low-pass filter to remove high-frequency noise
        lp_cutoff = 8000  # Hz - adjust based on your needs
        self.lp_b, self.lp_a = signal.butter(2, lp_cutoff / nyquist, btype="low")
        self.lp_zi = signal.lfilter_zi(self.lp_b, self.lp_a)

        # Noise gate parameters
        self.noise_threshold = 300
        self.gate_ratio = 0.05

    def _apply_noise_filter(self, audio_data):
        """Apply noise filtering to audio data."""
        # Convert to float for processing
        audio_float = audio_data.astype(np.float32)

        # Apply high-pass filter (remove low-frequency noise like AC hum)
        filtered_audio, self.hp_zi = signal.lfilter(self.hp_b, self.hp_a, audio_float.flatten(), zi=self.hp_zi)

        # Apply low-pass filter (remove high-frequency noise)
        filtered_audio, self.lp_zi = signal.lfilter(self.lp_b, self.lp_a, filtered_audio, zi=self.lp_zi)

        # Reshape back to original shape
        filtered_audio = filtered_audio.reshape(audio_data.shape)

        # Apply noise gate
        rms = np.sqrt(np.mean(filtered_audio**2, axis=1, keepdims=True))
        quiet_mask = rms < self.noise_threshold
        filtered_audio[quiet_mask] *= self.gate_ratio

        # Convert back to int16
        return np.clip(filtered_audio, -32768, 32767).astype(np.int16)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if self._running:
                self.stop()
        except Exception:
            pass  # Ignore errors during cleanup


continuous_recorder = None
audio_processor_task = None
CONSUMER_ID = "main_processor"  # Unique ID for the main audio processing consumer


def initialize_audio_recorder(sample_rate=16000, device_id=None, buffer_seconds=30):
    """
    Initialize the global continuous audio recorder.

    :param sample_rate: Sample rate for recording (Hz)
    :param device_id: ID of the audio device to use, or None for default
    :param buffer_seconds: Maximum duration of audio to keep in the buffer (seconds)
    """
    global continuous_recorder
    logger.info("called server.initialize_audio_recorder")

    # Stop existing recorder if running
    if continuous_recorder is not None:
        logger.info("Stopping existing continuous audio recorder")
        if not continuous_recorder.stop():
            logger.warning("Failed to stop existing recorder cleanly")
        continuous_recorder = None

    # Create and start new recorder
    continuous_recorder = ContinuousAudioRecorder(
        sample_rate=sample_rate, channels=1, device_id=device_id, buffer_seconds=buffer_seconds  # Mono audio
    )

    if continuous_recorder.start():
        logger.info(f"Initialized global continuous audio recorder with {buffer_seconds}s buffer at {sample_rate}Hz")
        return True
    else:
        logger.error("Failed to start continuous audio recorder")
        continuous_recorder = None
        return False


async def get_audio_frames(duration_seconds: float, update_info: dict = None) -> Tuple[np.ndarray, int]:
    """
    Get the most recent audio frames of the specified duration.

    :param duration_seconds: Duration of audio to retrieve in seconds
    :param update_info: Optional update info from wait_for_new_audio for consistency
    :return: Tuple of (audio data as numpy array, sample rate)
    """
    global continuous_recorder

    if continuous_recorder is None or not continuous_recorder.running:
        logger.error("Continuous audio recorder is not initialized or not running")
        # Return empty audio in case of error
        empty_audio = np.zeros((int(duration_seconds * 16000), 1), dtype=np.int16)
        return empty_audio, 16000

    # Get the audio data from the recorder with optional update validation
    audio_data = continuous_recorder.get_audio(duration_seconds, update_info)
    return audio_data, continuous_recorder.sample_rate


def pad_audio_with_silence(
    audio_np: np.ndarray, current_sample_rate: int, current_duration_seconds: float, target_duration_seconds: float = 15.0
) -> np.ndarray:
    """
    Prepends silence to audio data to reach a target duration.
    This helps meet minimum length requirements for STT APIs while keeping recording time short.

    :param audio_np: Audio data as numpy array
    :param current_sample_rate: Sample rate of the audio in Hz
    :param current_duration_seconds: Current duration of the audio in seconds
    :param target_duration_seconds: Target duration in seconds (default 15s for ElevenLabs STT)
    :return: Padded audio as numpy array
    """
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

        return padded_audio
    else:
        logger.info(f"Audio already meets minimum duration ({current_samples/current_sample_rate:.2f}s >= {target_duration_seconds}s)")
        return audio_np


###############################################################################
# Background Task for Local Audio Source Mode
###############################################################################


async def local_audio_source_processor():
    """
    A background task that, when in local-audio-source mode,
    periodically fetches local audio and triggers the publish process.
    The recording duration is aligned with the poll interval for consistent timing.
    """
    global continuous_recorder

    try:
        # Initialize the continuous audio recorder
        device_id = admin_config.get("audio_device_id")
        sample_rate = admin_config.get("sample_rate", 44100)
        buffer_seconds = max(60, admin_config.get("poll_interval", 10) * 3)  # Buffer at least 3x the poll interval

        if not initialize_audio_recorder(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds):
            logger.error("Failed to initialize audio recorder, stopping processor")
            return

        # Register as a consumer
        await continuous_recorder.register_consumer(MAIN_CONSUMER_ID)

        # First-time initialization
        first_run = True

        while True:
            try:
                # Check if recorder is still running
                if not continuous_recorder or not continuous_recorder.running:
                    logger.error("Audio recorder stopped unexpectedly, attempting restart")
                    if not initialize_audio_recorder(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds):
                        logger.error("Failed to restart audio recorder, waiting before retry")
                        await asyncio.sleep(5)
                        continue
                    await continuous_recorder.register_consumer(MAIN_CONSUMER_ID)

                # Use a semaphore to limit concurrent processing
                async with asyncio.Semaphore(3):  # Limit to 3 concurrent audio processing tasks
                    # Get configuration parameters
                    poll_interval = admin_config.get("poll_interval", 10)
                    min_audio_duration = admin_config.get("min_audio_duration", 15)

                    # On first run, use a shorter recording time for quick startup
                    # On subsequent runs, align recording time with poll interval
                    if first_run:
                        record_seconds = admin_config.get("first_record_seconds", 5)
                        first_run = False
                        logger.info(f"First processing: using shorter duration of {record_seconds}s")
                    else:
                        record_seconds = poll_interval
                        logger.info(f"Subsequent processing: using poll interval duration of {record_seconds}s")

                    # Wait for new audio to be available with sequence tracking
                    start_time = time.time()

                    # Calculate timeout based on poll interval to ensure we don't wait too long
                    wait_timeout = min(poll_interval * 0.8, 5.0)  # Max 5 seconds or 80% of poll interval

                    update_info = await continuous_recorder.wait_for_new_audio(consumer_id=MAIN_CONSUMER_ID, timeout=wait_timeout)

                    if update_info is None:
                        logger.warning(f"No new audio available within {wait_timeout}s timeout")
                        # Still try to get audio, but without update validation
                        audio_np, source_sample_rate = await get_audio_frames(record_seconds)
                    else:
                        logger.debug(f"New audio available (sequence: {update_info['sequence']}, timestamp: {update_info['timestamp']})")
                        # Fetch the audio frames with update validation
                        audio_np, source_sample_rate = await get_audio_frames(record_seconds, update_info)

                    recording_duration = time.time() - start_time
                    logger.info(f"Audio frames fetched in {recording_duration:.2f}s")

                    # Process the audio - stay in numpy format
                    resampled_audio_np = resample_audio(audio_np, source_sample_rate, 16000)

                    # Pad audio if needed to meet minimum duration - still in numpy format
                    current_duration = len(resampled_audio_np) / 16000  # in seconds
                    if current_duration < min_audio_duration:
                        padded_audio_np = pad_audio_with_silence(
                            audio_np=resampled_audio_np,
                            current_sample_rate=16000,
                            current_duration_seconds=current_duration,
                            target_duration_seconds=min_audio_duration,
                        )
                    else:
                        padded_audio_np = resampled_audio_np

                    timestamp = int(time.time())

                    # Create WAV file in memory for API - only convert to bytes at the last moment
                    audio_buffer = io.BytesIO()
                    with wave.open(audio_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # 16-bit audio
                        wav_file.setframerate(16000)
                        wav_file.writeframes(padded_audio_np.tobytes())  # Single conversion to bytes

                    audio_buffer.seek(0)  # Rewind to beginning

                    # Send to STT API
                    stt_start_time = time.time()
                    result = client.speech_to_text.convert(
                        model_id=STT_MODEL,
                        file=(f"input-file-{timestamp}.wav", audio_buffer, "audio/wav"),
                        timestamps_granularity="word",
                    )
                    stt_duration = time.time() - stt_start_time
                    logger.info(f"Speech-to-text completed in {stt_duration:.2f}s")
                    logger.debug(f"{result=}")

                    transcribed_text = result.text
                    timestamp = time.time()

                    # Only process if there's actual transcribed text
                    if transcribed_text.strip():
                        # Publish the transcription
                        await published_data_store.update_speaking(timestamp, transcribed_text)

                        # For every target language (skip if source equals target)
                        async def process_language(language: str):
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
                        logger.info(f"Completed processing in {elapsed_time:.2f}s. Waiting {remaining_time:.2f}s until next processing.")
                        await asyncio.sleep(remaining_time)
                    else:
                        logger.warning(
                            f"Processing took {elapsed_time:.2f}s, "
                            f"which exceeds the poll interval of {poll_interval}s. "
                            "Starting next processing immediately."
                        )

            except Exception as e:
                logger.error(f"Error during local audio processing: {str(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

                # Wait the poll interval before trying again after an error
                await asyncio.sleep(admin_config.get("poll_interval", 10))

    except Exception as e:
        logger.error(f"Fatal error in local audio source processor: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

    finally:
        # Clean up consumer registration
        if continuous_recorder:
            try:
                await continuous_recorder.unregister_consumer(MAIN_CONSUMER_ID)
                logger.info("Unregistered main audio consumer")
            except Exception as e:
                logger.warning(f"Error unregistering consumer: {e}")


@app.on_event("startup")
async def startup_event():
    """
    On startup, if the mode is set to local-audio-source,
    start the background task that periodically fetches and publishes local audio.
    """
    global audio_processor_task

    logger.info("Starting FastAPI server, please wait...")
    if PUBLISH_MODE == "local-audio-source":
        logger.info(f"Starting background task for local audio source mode with poll_interval={admin_config.get('poll_interval', 10)}...")
        # Launch the background task
        audio_processor_task = asyncio.create_task(local_audio_source_processor())
    else:
        logger.info("Server running in upload mode. Use the /publish endpoint to send audio.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown."""
    global continuous_recorder, audio_processor_task

    logger.info("Server shutting down")

    # Cancel audio processor task
    if audio_processor_task and not audio_processor_task.done():
        logger.info("Cancelling audio processor task")
        audio_processor_task.cancel()
        try:
            await audio_processor_task
        except asyncio.CancelledError:
            logger.info("Audio processor task cancelled")
        except Exception as e:
            logger.warning(f"Error during audio processor task cancellation: {e}")

    # Stop continuous recorder
    if continuous_recorder is not None:
        logger.info("Stopping continuous audio recorder")
        if continuous_recorder.stop():
            logger.info("Continuous audio recorder stopped successfully")
        else:
            logger.warning("Failed to stop continuous audio recorder cleanly")
        continuous_recorder = None

    logger.info("Server shutdown complete")


###############################################################################
# HTTP Endpoints
###############################################################################
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.post("/publish", dependencies=[Depends(verify_api_key)])
async def publish_audio_upload(
    mode: str = Form(...),
    sourceLanguage: str = Form(...),
    audio_file: UploadFile = File(None),
    sampleRate: int = Form(None),
):
    """
    This endpoint is intended for upload mode only.
    In local-audio-source mode, publishing is done via the background task.
    """
    if mode != "upload":
        raise HTTPException(status_code=400, detail="This endpoint accepts only upload mode requests.")

    if audio_file is None or sampleRate is None:
        raise HTTPException(status_code=400, detail="audio_file and sampleRate are required in upload mode.")

    try:
        # Read the uploaded audio file
        audio_data = await audio_file.read()
        source_sample_rate = sampleRate

        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_np.shape) == 1:
            audio_np = audio_np.reshape(-1, 1)

        # Resample the audio to 16kHz
        resampled_audio_np = resample_audio(audio_np, source_sample_rate, 16000)

        # Check if we need to pad the audio to meet minimum duration
        min_audio_duration = admin_config.get("min_audio_duration", 15)
        current_duration = len(resampled_audio_np) / 16000  # in seconds

        if current_duration < min_audio_duration:
            padded_audio_np = pad_audio_with_silence(
                audio_np=resampled_audio_np,
                current_sample_rate=16000,
                current_duration_seconds=current_duration,
                target_duration_seconds=min_audio_duration,
            )
        else:
            padded_audio_np = resampled_audio_np

        # Create WAV file in memory for API
        timestamp = int(time.time())
        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(16000)
            wav_file.writeframes(padded_audio_np.tobytes())  # Single conversion to bytes

        audio_buffer.seek(0)  # Rewind to beginning

        # Send to STT API
        stt_start_time = time.time()
        result = client.speech_to_text.convert(
            model_id=STT_MODEL,
            file=(f"input-file-{timestamp}.wav", audio_buffer, "audio/wav"),
            timestamps_granularity="word",
        )
        stt_duration = time.time() - stt_start_time
        logger.info(f"Speech-to-text completed in {stt_duration:.2f}s")

        transcribed_text = result.text
        timestamp = time.time()

        # Only process if there's actual transcribed text
        if not transcribed_text.strip():
            logger.info("No speech detected in the uploaded audio, skipping processing")
            return {"timestamp": timestamp, "message": "No speech detected in the uploaded audio"}

        # Publish the transcription
        await published_data_store.update_speaking(timestamp, transcribed_text)

        # Process each target language in parallel
        async def process_language(language: str):
            if sourceLanguage.upper() == language.upper():
                logger.info(f"Skipping translation for {language} because source and target are the same.")
                return
            translated_text = translate_text(transcribed_text, sourceLanguage, language)
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

        return {
            "timestamp": timestamp,
            "message": "Published transcription, translation, and TTS audio (upload mode).",
            "transcribed_text": transcribed_text,
        }

    except Exception as e:
        logger.error(f"Error processing uploaded audio: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


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

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


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

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


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
                    result.append({"timestamp": entry["timestamp"], "audio": encoded_audio})

                # Yield the new audio as a JSON string with SSE format
                yield f"data: {json.dumps(result)}\n\n"
            else:
                # Wait for new data
                update_occurred = await published_data_store.wait_for_audio_update(lang)
                if not update_occurred:
                    # Send a keep-alive message if no updates
                    yield f"data: {json.dumps([])}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/captions/speaking")
async def get_speaking_captions(timestamp: float = None):
    """Get speaking captions with chunked response to prevent starvation"""

    async def generate_chunks():
        # Get data in chunks to prevent blocking
        captions = await published_data_store.get_speaking(timestamp)
        chunk_size = 5  # Adjust based on expected data size

        for i in range(0, len(captions), chunk_size):
            chunk = captions[i : i + chunk_size]
            yield json.dumps(chunk).encode() + b"\n"
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

    return StreamingResponse(generate_chunks(), media_type="application/json", headers={"Content-Disposition": "inline"})


@app.get("/captions/translated")
async def get_translated_captions(lang: str, timestamp: float = None):
    """Get translated captions with chunked response to prevent starvation"""

    async def generate_chunks():
        # Get data in chunks to prevent blocking
        translations = await published_data_store.get_translated(lang, timestamp)
        chunk_size = 5  # Adjust based on expected data size

        for i in range(0, len(translations), chunk_size):
            chunk = translations[i : i + chunk_size]
            yield json.dumps(chunk).encode() + b"\n"
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

    return StreamingResponse(generate_chunks(), media_type="application/json", headers={"Content-Disposition": "inline"})


@app.get("/audio/translated-voice")
async def get_translated_voice(lang: str, timestamp: float = None):
    """Get translated voice with chunked response to prevent starvation"""
    try:

        async def generate_chunks():
            # Get data in chunks to prevent blocking
            audio_entries = await published_data_store.get_audio(lang, timestamp)
            chunk_size = 2  # Smaller chunk size for audio due to size

            for i in range(0, len(audio_entries), chunk_size):
                chunk = audio_entries[i : i + chunk_size]
                result = []
                for entry in chunk:
                    encoded_audio = base64.b64encode(entry["audio"]).decode("utf-8")
                    result.append({"timestamp": entry["timestamp"], "audio": encoded_audio})
                yield json.dumps(result).encode() + b"\n"
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.02)  # Slightly longer delay for audio

        return StreamingResponse(generate_chunks(), media_type="application/json", headers={"Content-Disposition": "inline"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


###############################################################################
# Admin Helpers
###############################################################################


async def verify_admin_token(x_admin_token: str = Header(...)):
    if x_admin_token != admin_config.get("admin_token"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/admin/config", dependencies=[Depends(verify_admin_token)])
async def get_config():
    return admin_config


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
    global admin_config, PUBLISH_MODE, DEFAULT_SOURCE_LANGUAGE, TTS_MODEL, STT_MODEL, openai_client, continuous_recorder, audio_processor_task

    # Ensure admin_token is not updated
    if "admin_token" in config:
        del config["admin_token"]

    # Check if publish_mode is changing
    publish_mode_changing = "publish_mode" in config and config["publish_mode"] != admin_config.get("publish_mode")
    old_publish_mode = admin_config.get("publish_mode")

    # Check if audio-related settings are changing
    audio_settings_changing = any(key in config for key in ["audio_device_id", "sample_rate", "buffer_seconds", "poll_interval"])

    # Update the config values
    admin_config.update(config)
    save_admin_config(admin_config)

    # Reinitialize API keys and other constants if they are updated.
    if "openai_api_key" in config:
        # Assuming openai.OpenAI is imported and available
        import openai

        openai_client = openai.OpenAI(api_key=admin_config["openai_api_key"])
        logger.info("Updated OpenAI client with new API key")

    if "elevenlabs_api_key" in config:
        logger.warning(
            "ElevenLabs API key update detected, but client reinitialization is not implemented. "
            "The new API key will not take effect until server restart."
        )
        pass

    # Handle publish mode changes
    if publish_mode_changing:
        PUBLISH_MODE = admin_config.get("publish_mode")
        logger.info(f"Publish mode changed from {old_publish_mode} to {PUBLISH_MODE}")

        # Cancel existing audio processor task if running
        if audio_processor_task and not audio_processor_task.done():
            logger.info("Cancelling existing audio processor task")
            audio_processor_task.cancel()
            try:
                await audio_processor_task
            except asyncio.CancelledError:
                logger.info("Audio processor task cancelled successfully")
            except Exception as e:
                logger.warning(f"Error cancelling audio processor task: {e}")
            audio_processor_task = None

        # Stop the continuous recorder if it's running
        if continuous_recorder is not None:
            logger.info("Stopping continuous audio recorder due to publish mode change")
            if continuous_recorder.stop():
                logger.info("Continuous audio recorder stopped successfully")
            else:
                logger.warning("Failed to stop continuous audio recorder cleanly")
            continuous_recorder = None

        # Start the local audio source processor if switching to that mode
        if PUBLISH_MODE == "local-audio-source":
            logger.info("Starting local audio source processor due to publish mode change")
            audio_processor_task = asyncio.create_task(local_audio_source_processor())
        else:
            logger.info("Switched to upload mode. Use the /publish endpoint to send audio.")

    # Handle audio settings changes while in local-audio-source mode
    elif PUBLISH_MODE == "local-audio-source" and audio_settings_changing:
        logger.info("Audio settings changed while in local-audio-source mode")

        # Get new settings
        device_id = admin_config.get("audio_device_id")
        sample_rate = admin_config.get("sample_rate", 44100)
        buffer_seconds = max(60, admin_config.get("poll_interval", 10) * 3)

        if continuous_recorder is not None:
            # Check if we need to restart the recorder
            current_status = continuous_recorder.get_status()
            needs_restart = (
                current_status["device_id"] != device_id
                or current_status["sample_rate"] != sample_rate
                or current_status["buffer_seconds"] != buffer_seconds
            )

            if needs_restart:
                logger.info("Audio recorder settings changed significantly, restarting recorder")

                # Update settings first
                settings_updated = continuous_recorder.update_settings(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds)

                if settings_updated:
                    # Restart the recorder to apply new settings
                    if continuous_recorder.restart():
                        logger.info("Audio recorder restarted successfully with new settings")
                    else:
                        logger.error("Failed to restart audio recorder, reinitializing")
                        if not initialize_audio_recorder(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds):
                            logger.error("Failed to reinitialize audio recorder")
                else:
                    logger.info("Audio settings unchanged, no restart needed")
            else:
                logger.info("Audio settings changed but don't require recorder restart")
        else:
            # No recorder running, initialize one
            logger.info("No audio recorder running, initializing new one")
            initialize_audio_recorder(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds)
    else:
        # If not changing publish mode, just update the variable
        PUBLISH_MODE = admin_config.get("publish_mode", PUBLISH_MODE)

    # Update other configuration variables
    DEFAULT_SOURCE_LANGUAGE = admin_config.get("default_source_language", DEFAULT_SOURCE_LANGUAGE)
    TTS_MODEL = admin_config.get("tts_model", TTS_MODEL)
    STT_MODEL = admin_config.get("stt_model", STT_MODEL)

    # Log the current status
    if continuous_recorder:
        status = continuous_recorder.get_status()
        logger.info(
            f"Audio recorder status: running={status['running']}, "
            f"device_id={status['device_id']}, sample_rate={status['sample_rate']}Hz, "
            f"consumers={len(status['consumers'])}"
        )

    return {
        "message": "Configuration updated successfully",
        "config": admin_config,
        "audio_recorder_status": continuous_recorder.get_status() if continuous_recorder else None,
    }


@app.get("/admin/audio-status", dependencies=[Depends(verify_admin_token)])
async def get_audio_status():
    """Get the current status of the audio recorder."""
    global continuous_recorder, audio_processor_task

    status = {"publish_mode": PUBLISH_MODE, "audio_recorder": None, "audio_processor_task": None}

    if continuous_recorder:
        status["audio_recorder"] = continuous_recorder.get_status()

    if audio_processor_task:
        status["audio_processor_task"] = {
            "running": not audio_processor_task.done(),
            "cancelled": audio_processor_task.cancelled(),
            "exception": str(audio_processor_task.exception()) if audio_processor_task.done() and audio_processor_task.exception() else None,
        }

    return status


@app.post("/admin/audio-control", dependencies=[Depends(verify_admin_token)])
async def control_audio_recorder(action: str):
    """
    Control the audio recorder.
    Actions: start, stop, restart, status
    """
    global continuous_recorder

    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(status_code=400, detail="Invalid action. Use: start, stop, restart, or status")

    if action == "status":
        return {"action": action, "status": continuous_recorder.get_status() if continuous_recorder else None}

    if continuous_recorder is None:
        if action in ["stop", "restart"]:
            return {"action": action, "message": "No audio recorder to control", "success": False}
        elif action == "start":
            # Initialize with current config
            device_id = admin_config.get("audio_device_id")
            sample_rate = admin_config.get("sample_rate", 44100)
            buffer_seconds = max(60, admin_config.get("poll_interval", 10) * 3)

            success = initialize_audio_recorder(sample_rate=sample_rate, device_id=device_id, buffer_seconds=buffer_seconds)
            return {"action": action, "success": success, "status": continuous_recorder.get_status() if continuous_recorder else None}

    # Recorder exists, perform action
    if action == "start":
        success = continuous_recorder.start()
    elif action == "stop":
        success = continuous_recorder.stop()
    elif action == "restart":
        success = continuous_recorder.restart()

    return {"action": action, "success": success, "status": continuous_recorder.get_status()}


@app.get("/admin/audio-devices", dependencies=[Depends(verify_admin_token)])
async def list_audio_devices():
    """
    List all available audio input devices.
    """
    try:
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                input_devices.append(
                    {
                        "id": i,
                        "name": device.get("name", f"Device {i}"),
                        "channels": device.get("max_input_channels"),
                        "default_samplerate": device.get("default_samplerate"),
                        "is_default": device.get("default_input", False),
                    }
                )

        return {"devices": input_devices, "current_device_id": admin_config.get("audio_device_id")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing audio devices: {str(e)}")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index(request: Request, streaming: bool = False):
    """Serve the main index.html page or streaming-frontend.html if streaming is enabled."""
    base_url = str(request.base_url).rstrip("/")
    target_languages = admin_config.get("target_languages")
    languages = [
        {
            "code": lang.value,
            "name": lang.value.title(),
            "native_name": lang.native_name,
            "messages": LOCALIZED_MESSAGES.get(lang.value, DEFAULT_MESSAGES),
        }
        for lang in Language
        if lang in target_languages
    ]
    # Prepare template context
    context = {
        "request": request,
        "api_base_url": base_url,
        "languages": languages,
    }

    if streaming:
        return templates.TemplateResponse("streaming-frontend.html", context)
    else:
        return templates.TemplateResponse("index.html", context)


@app.get("/admin")
async def serve_admin(request: Request, admin_token: str):
    """Serve the admin frontend page with current configuration."""
    # Get the current server base URL
    base_url = str(request.base_url).rstrip("/")
    if admin_token != admin_config.get("admin_token"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    # Prepare template context
    context = {
        "request": request,
        "config": admin_config,
        "api_base_url": base_url,
        "admin_token": admin_token,
    }

    return templates.TemplateResponse("admin-frontend.html", context)


@app.get("/upload")
async def serve_upload(
    request: Request,
    api_key: str = Query(None, include_in_schema=False),
):
    """Serve the main index.html page or streaming-frontend.html if streaming is enabled."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    base_url = str(request.base_url).rstrip("/")
    target_languages = admin_config.get("target_languages")
    languages = [
        {
            "code": lang.value,
            "name": lang.value.title(),
            "native_name": lang.native_name,
            "messages": LOCALIZED_MESSAGES.get(lang.value, DEFAULT_MESSAGES),
        }
        for lang in Language
        if lang in target_languages
    ]
    context = {
        "request": request,
        "api_base_url": base_url,
        "languages": languages,
        "api_key": api_key,
    }
    return templates.TemplateResponse("upload-frontend.html", context)


###############################################################################
# Main Application Runner
###############################################################################

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9001)
