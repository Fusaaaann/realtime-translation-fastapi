import wave
import numpy as np
import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class MockContinuousAudioRecorder:
    """
    A mock implementation of ContinuousAudioRecorder that loads audio from a WAV file
    instead of recording from a real device. Maintains the same API as the real recorder.
    """

    def __init__(self, sample_rate=16000, channels=1, device_id=None, buffer_seconds=30, test_file="test_audio.wav"):
        """
        Initialize the mock audio recorder.

        :param sample_rate: Sample rate for recording (Hz)
        :param channels: Number of audio channels (1 for mono, 2 for stereo)
        :param device_id: ID of the audio device to use (ignored in mock)
        :param buffer_seconds: Maximum duration of audio to keep in the buffer (seconds)
        :param test_file: Path to the WAV file to load
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.buffer_seconds = buffer_seconds
        self.test_file = test_file

        # Calculate buffer size in frames
        self.buffer_size = int(buffer_seconds * sample_rate)

        # Load test audio and create circular buffer
        self.test_audio_data = self._load_test_audio()
        self.buffer = np.zeros((self.buffer_size, channels), dtype=np.int16)
        self.buffer_index = 0
        self._fill_buffer_with_test_audio()

        # Thread synchronization and resource management
        self.lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._running = False
        self._starting = False
        self._stopping = False
        self.thread = None

        # Audio update tracking
        self.audio_update_event = None
        self.current_sequence = 0
        self.last_processed_sequences = {}
        self.audio_update_queue = None
        self.last_update_timestamp = time.time()

        # Consumer management
        self.consumers = set()

        # Mock-specific: position in test audio for cycling
        self.test_audio_position = 0

        logger.info(f"Initialized MockContinuousAudioRecorder with test file: {test_file}")

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

    def _load_test_audio(self) -> np.ndarray:
        """
        Load audio data from the test WAV file.

        :return: Numpy array of audio data
        """
        try:
            with wave.open(self.test_file, "rb") as wav_file:
                # Get file properties
                file_channels = wav_file.getnchannels()
                file_sample_width = wav_file.getsampwidth()
                file_sample_rate = wav_file.getframerate()
                file_frames = wav_file.getnframes()

                # Read all frames
                audio_bytes = wav_file.readframes(file_frames)

                # Convert to numpy array based on sample width
                if file_sample_width == 2:  # 16-bit audio
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                elif file_sample_width == 1:  # 8-bit audio
                    audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) * 256 - 32768
                elif file_sample_width == 4:  # 32-bit audio
                    audio_np = (np.frombuffer(audio_bytes, dtype=np.int32) / 65536).astype(np.int16)
                else:
                    raise ValueError(f"Unsupported sample width: {file_sample_width}")

                # Reshape for channels
                if file_channels > 1:
                    audio_np = audio_np.reshape(-1, file_channels)
                    # Convert to mono if needed
                    if self.channels == 1 and file_channels > 1:
                        audio_np = np.mean(audio_np, axis=1, dtype=np.int16).reshape(-1, 1)
                    elif file_channels > self.channels:
                        audio_np = audio_np[:, : self.channels]
                else:
                    audio_np = audio_np.reshape(-1, 1)

                # Resample if needed
                if file_sample_rate != self.sample_rate:
                    from scipy.signal import resample

                    num_samples = int(len(audio_np) * self.sample_rate / file_sample_rate)
                    if audio_np.shape[1] > 1:
                        # Resample each channel separately
                        resampled_channels = []
                        for ch in range(audio_np.shape[1]):
                            resampled_ch = resample(audio_np[:, ch], num_samples)
                            resampled_channels.append(resampled_ch)
                        audio_np = np.column_stack(resampled_channels).astype(np.int16)
                    else:
                        audio_np = resample(audio_np.flatten(), num_samples).reshape(-1, 1).astype(np.int16)

                logger.info(f"Loaded test audio: {len(audio_np)} frames, {audio_np.shape[1]} channels, {file_sample_rate}Hz -> {self.sample_rate}Hz")
                return audio_np

        except Exception as e:
            logger.error(f"Error loading test audio file {self.test_file}: {str(e)}")
            # Return 10 seconds of low-level pink noise as fallback
            duration_samples = self.sample_rate * 10
            # Generate pink noise (more natural than white noise)
            white_noise = np.random.normal(0, 1000, duration_samples)
            # Simple pink noise approximation
            pink_noise = white_noise.copy()
            for i in range(1, len(pink_noise)):
                pink_noise[i] = 0.99 * pink_noise[i - 1] + 0.01 * white_noise[i]
            return np.clip(pink_noise, -32768, 32767).astype(np.int16).reshape(-1, 1)

    def _fill_buffer_with_test_audio(self):
        """Fill the circular buffer with the test audio, repeating if necessary."""
        test_audio_length = len(self.test_audio_data)

        if test_audio_length == 0:
            logger.warning("Test audio data is empty, filling buffer with silence")
            return

        # Fill buffer by cycling through the test audio
        for i in range(self.buffer_size):
            source_idx = i % test_audio_length
            self.buffer[i] = self.test_audio_data[source_idx]

    def start(self) -> bool:
        """Start the mock recording thread."""
        with self._resource_lock():
            if self._running:
                logger.warning("Mock audio recorder is already running")
                return False

            if self._starting:
                logger.warning("Mock audio recorder is already starting")
                return False

            if self._stopping:
                logger.warning("Mock audio recorder is currently stopping, please wait")
                return False

            try:
                self._starting = True
                self._initialize_async_components()

                # Start the mock recording thread
                self.thread = threading.Thread(target=self._mock_record_thread, daemon=True)
                self.thread.start()

                # Wait a moment to ensure thread starts properly
                time.sleep(0.1)

                if self.thread.is_alive():
                    self._running = True
                    logger.info("Started mock audio recording thread")
                    return True
                else:
                    logger.error("Failed to start mock audio recording thread")
                    return False

            except Exception as e:
                logger.error(f"Error starting mock audio recorder: {e}")
                return False
            finally:
                self._starting = False

    def stop(self) -> bool:
        """Stop the mock recording thread and clean up resources."""
        with self._resource_lock():
            if not self._running:
                logger.warning("Mock audio recorder is not running")
                return False

            if self._stopping:
                logger.warning("Mock audio recorder is already stopping")
                return False

            try:
                self._stopping = True
                self._running = False

                # Wait for thread to finish
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=3.0)
                    if self.thread.is_alive():
                        logger.warning("Mock audio recording thread did not terminate cleanly within timeout")
                        return False

                self.thread = None
                self._cleanup_async_components()
                logger.info("Stopped mock audio recording thread")
                return True

            except Exception as e:
                logger.error(f"Error stopping mock audio recorder: {e}")
                return False
            finally:
                self._stopping = False

    def restart(self) -> bool:
        """Restart the mock recording with current settings."""
        logger.info("Restarting mock audio recorder")
        if not self.stop():
            logger.error("Failed to stop mock recorder for restart")
            return False

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
            logger.debug("No event loop available for async components initialization")

    def _cleanup_async_components(self):
        """Clean up async components."""
        self.consumers.clear()
        self.last_processed_sequences.clear()

    def _mock_record_thread(self):
        """Background thread that simulates continuous audio recording by cycling through test audio."""
        try:
            logger.info("Started mock audio recording simulation")

            # Calculate chunk size (simulate 100ms chunks like real recorder)
            chunk_frames = int(self.sample_rate * 0.1)  # 100ms

            while self._running:
                with self.lock:
                    if not self._running:
                        logger.info("Mock recording thread breaking: _running flag set to False")
                        break

                    # Get next chunk from test audio, cycling through it
                    chunk_data = self._get_next_chunk(chunk_frames)

                    # Add chunk to circular buffer
                    frames_to_end = self.buffer_size - self.buffer_index
                    if chunk_frames <= frames_to_end:
                        # Can write all frames without wrapping
                        self.buffer[self.buffer_index : self.buffer_index + chunk_frames] = chunk_data
                        self.buffer_index = (self.buffer_index + chunk_frames) % self.buffer_size
                    else:
                        # Need to wrap around the buffer
                        self.buffer[self.buffer_index :] = chunk_data[:frames_to_end]
                        self.buffer[0 : chunk_frames - frames_to_end] = chunk_data[frames_to_end:]
                        self.buffer_index = chunk_frames - frames_to_end

                # Signal that new audio is available (if async components are ready)
                if self.audio_update_event is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop and not loop.is_closed():
                            future = asyncio.run_coroutine_threadsafe(self._signal_new_audio(), loop)
                            future.add_done_callback(lambda f: logger.error(f"Signal error: {f.exception()}") if f.exception() else None)
                    except RuntimeError:
                        pass

                # Sleep to simulate real-time audio streaming
                time.sleep(0.1)  # 100ms chunks

        except Exception as e:
            logger.error(f"Error in mock audio recording thread: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            with self._resource_lock():
                self._running = False

    def _get_next_chunk(self, chunk_frames: int) -> np.ndarray:
        """Get the next chunk of audio data from the test audio, cycling through it."""
        test_audio_length = len(self.test_audio_data)
        if test_audio_length == 0:
            return np.zeros((chunk_frames, self.channels), dtype=np.int16)

        chunk_data = np.zeros((chunk_frames, self.channels), dtype=np.int16)

        for i in range(chunk_frames):
            source_idx = (self.test_audio_position + i) % test_audio_length
            chunk_data[i] = self.test_audio_data[source_idx]

        # Advance position in test audio
        self.test_audio_position = (self.test_audio_position + chunk_frames) % test_audio_length

        return chunk_data

    async def _signal_new_audio(self):
        """Signal that new audio is available with sequence tracking."""
        self.current_sequence += 1
        timestamp = time.time()

        audio_update = {
            "sequence": self.current_sequence,
            "timestamp": timestamp,
            "buffer_index": self.buffer_index,
        }

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

        if self.audio_update_event:
            self.audio_update_event.set()

        self.last_update_timestamp = timestamp

    async def register_consumer(self, consumer_id: str):
        """Register a new audio consumer."""
        with self.lock:
            self.consumers.add(consumer_id)
            self.last_processed_sequences[consumer_id] = 0
        logger.info(f"Registered mock audio consumer: {consumer_id}")

    async def unregister_consumer(self, consumer_id: str):
        """Unregister an audio consumer."""
        with self.lock:
            self.consumers.discard(consumer_id)
            self.last_processed_sequences.pop(consumer_id, None)
        logger.info(f"Unregistered mock audio consumer: {consumer_id}")

    async def wait_for_new_audio(self, consumer_id: str, timeout=None):
        """Wait for new audio with guaranteed delivery."""
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
        """Get multiple audio updates at once for batch processing."""
        updates = []
        try:
            if consumer_id not in self.last_processed_sequences:
                await self.register_consumer(consumer_id)

            last_seq = self.last_processed_sequences[consumer_id]

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
        """Get the most recent audio of the specified duration."""
        frames = int(duration_seconds * self.sample_rate)

        with self.lock:
            if not self._running:
                logger.warning("Attempting to get audio from stopped mock recorder")
                # Return test audio instead of silence for testing purposes
                if len(self.test_audio_data) > 0:
                    test_frames = min(frames, len(self.test_audio_data))
                    if test_frames < frames:
                        # Repeat the test audio to fill the requested duration
                        repeats = (frames // test_frames) + 1
                        repeated_audio = np.tile(self.test_audio_data[:test_frames], (repeats, 1))
                        return repeated_audio[:frames]
                    else:
                        return self.test_audio_data[:frames]
                else:
                    return np.zeros((frames, self.channels), dtype=np.int16)

            # Validate update info if provided
            if update_info and update_info["buffer_index"] != self.buffer_index:
                logger.debug("Buffer index mismatch in mock recorder - this is expected")

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
                # No wrap-around needed
                result[:] = self.buffer[start_index : self.buffer_index]
            else:
                # Need to wrap around the buffer
                frames_from_end = self.buffer_size - start_index
                result[:frames_from_end] = self.buffer[start_index:]
                result[frames_from_end:] = self.buffer[: self.buffer_index]

            return result

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the mock audio recorder."""
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
                "stream_active": self._running,
                "test_file": self.test_file,
                "test_audio_frames": len(self.test_audio_data) if hasattr(self, "test_audio_data") else 0,
                "test_audio_position": self.test_audio_position,
            }

    def update_settings(self, sample_rate=None, device_id=None, buffer_seconds=None):
        """Update recorder settings. Requires restart to take effect."""
        with self._resource_lock():
            settings_changed = False

            if sample_rate is not None and sample_rate != self.sample_rate:
                self.sample_rate = sample_rate
                settings_changed = True
                logger.info(f"Updated mock sample rate to {sample_rate}Hz")

            if device_id != self.device_id:
                self.device_id = device_id
                settings_changed = True
                logger.info(f"Updated mock device ID to {device_id} (ignored)")

            if buffer_seconds is not None and buffer_seconds != self.buffer_seconds:
                self.buffer_seconds = buffer_seconds
                self.buffer_size = int(buffer_seconds * self.sample_rate)
                self.buffer = np.zeros((self.buffer_size, self.channels), dtype=np.int16)
                self.buffer_index = 0
                self._fill_buffer_with_test_audio()
                settings_changed = True
                logger.info(f"Updated mock buffer duration to {buffer_seconds}s")

            return settings_changed

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
            pass


# Global mock recorder - matches the server's global variable name
continuous_recorder = None


def initialize_audio_recorder(sample_rate=16000, device_id=None, buffer_seconds=30):
    """
    Initialize the global continuous audio recorder.
    Mock version that uses test audio file instead of real microphone.

    :param sample_rate: Sample rate for recording (Hz)
    :param device_id: ID of the audio device to use (ignored in mock)
    :param buffer_seconds: Maximum duration of audio to keep in the buffer (seconds)
    """
    import server  # Import the server module

    logger.info("called mock_audio.initialize_audio_recorder")
    # Stop existing recorder if running
    if server.continuous_recorder is not None:
        logger.info(
            f"Stopping existing continuous_recorder: {type(server.continuous_recorder).__name__} "
            f"(running: {getattr(server.continuous_recorder, 'running', 'unknown')}, "
            f"id: {id(server.continuous_recorder)})"
        )
        if not server.continuous_recorder.stop():
            logger.warning("Failed to stop existing mock recorder cleanly")
        server.continuous_recorder = None

    # Create and start new mock recorder
    server.continuous_recorder = MockContinuousAudioRecorder(
        sample_rate=sample_rate,
        channels=1,  # Mono audio
        device_id=device_id,
        buffer_seconds=buffer_seconds,
        test_file="test_audio.wav",
    )

    if server.continuous_recorder.start():
        logger.info("Initialized global mock continuous audio recorder with test file")
        logger.info(f"{server.continuous_recorder.running=}")
        return True
    else:
        logger.error("Failed to start mock continuous audio recorder")
        server.continuous_recorder = None
        return False


async def get_audio_frames(duration_seconds: float, update_info: dict = None) -> Tuple[np.ndarray, int]:
    """
    Get the most recent audio frames of the specified duration from mock recorder.

    :param duration_seconds: Duration of audio to retrieve in seconds
    :param update_info: Optional update info from wait_for_new_audio for consistency
    :return: Tuple of (audio data as numpy array, sample rate)
    """
    import server
    import wave
    import os

    # Show the result of each expression for debugging
    recorder_exists = server.continuous_recorder is not None
    recorder_running = server.continuous_recorder.running if recorder_exists else False

    logger.info(f"get_audio_frames called: duration={duration_seconds}s, " f"recorder_exists={recorder_exists}, recorder_running={recorder_running}")

    if recorder_exists:
        logger.info(f"Recorder type: {type(server.continuous_recorder).__name__}, " f"id: {id(server.continuous_recorder)}")

    if not recorder_exists or not recorder_running:
        logger.error(f"Mock continuous audio recorder check failed: " f"recorder_exists={recorder_exists}, recorder_running={recorder_running}")
        # Return test audio or empty audio in case of error
        empty_audio = np.zeros((int(duration_seconds * 16000), 1), dtype=np.int16)
        return empty_audio, 16000

    # Get the audio data from the mock recorder with optional update validation
    audio_data = server.continuous_recorder.get_audio(duration_seconds, update_info)
    sample_rate = server.continuous_recorder.sample_rate

    # Save audio to debug_{timestamp}.wav for debugging
    try:
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        debug_filename = f"debug_{timestamp}.wav"

        # Ensure audio_data is properly shaped
        if len(audio_data.shape) == 1:
            audio_data_to_save = audio_data.reshape(-1, 1)
        else:
            audio_data_to_save = audio_data

        # Create debug directory if it doesn't exist
        debug_dir = "debug_audio"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, debug_filename)

        with wave.open(debug_path, "wb") as wav_file:
            wav_file.setnchannels(audio_data_to_save.shape[1])  # Number of channels
            wav_file.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data_to_save.tobytes())

        logger.info(f"Saved debug audio: {debug_path} " f"({len(audio_data)} frames, {audio_data_to_save.shape[1]} channels, {sample_rate}Hz)")

    except Exception as e:
        logger.warning(f"Failed to save debug audio: {e}")

    return audio_data, sample_rate
