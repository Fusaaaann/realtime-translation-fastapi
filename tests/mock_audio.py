import wave
import numpy as np
import asyncio
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class MockContinuousAudioRecorder:
    """
    A mock implementation of ContinuousAudioRecorder that loads audio from a WAV file
    instead of recording from a real device. Useful for testing.
    """

    def __init__(self, sample_rate=16000, channels=1, device_id=None, buffer_seconds=30, test_file="test_audio.wav"):
        """
        Initialize the mock audio recorder.

        :param sample_rate: Sample rate for the mock recorder
        :param channels: Number of audio channels
        :param device_id: Ignored in mock implementation
        :param buffer_seconds: Maximum duration of audio to keep in buffer
        :param test_file: Path to the WAV file to load
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.test_file = test_file

        # Load the test audio file
        self.audio_data = self._load_test_audio()

        # Calculate buffer size in frames
        self.buffer_size = int(buffer_seconds * sample_rate)

        # Create a circular buffer filled with the test audio
        self.buffer = np.zeros((self.buffer_size, channels), dtype=np.int16)
        self._fill_buffer_with_test_audio()

        # Current position in the buffer
        self.buffer_index = 0

        # Thread synchronization (not actually used in mock, but kept for API compatibility)
        self.running = False

        # For tracking when new audio is available
        self.last_read_index = 0
        self.new_audio_event = asyncio.Event()

        logger.info(f"Initialized MockContinuousAudioRecorder with test file: {test_file}")

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

                # Convert to numpy array
                if file_sample_width == 2:  # 16-bit audio
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                elif file_sample_width == 1:  # 8-bit audio
                    audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16) * 256
                else:
                    raise ValueError(f"Unsupported sample width: {file_sample_width}")

                # Reshape for channels
                if file_channels > 1:
                    audio_np = audio_np.reshape(-1, file_channels)
                    # If test file has more channels than we need, take only the first channel
                    if file_channels > self.channels:
                        audio_np = audio_np[:, : self.channels]
                else:
                    audio_np = audio_np.reshape(-1, 1)

                # Resample if needed
                if file_sample_rate != self.sample_rate:
                    from scipy.signal import resample

                    num_samples = int(len(audio_np) * self.sample_rate / file_sample_rate)
                    audio_np = resample(audio_np, num_samples)

                logger.info(f"Loaded test audio: {len(audio_np)} frames, {file_channels} channels, {file_sample_rate}Hz")
                return audio_np

        except Exception as e:
            logger.error(f"Error loading test audio file {self.test_file}: {str(e)}")
            # Return 5 seconds of silence as fallback
            return np.zeros((self.sample_rate * 5, self.channels), dtype=np.int16)

    def _fill_buffer_with_test_audio(self):
        """Fill the circular buffer with the test audio, repeating if necessary."""
        test_audio_length = len(self.audio_data)

        # If test audio is shorter than buffer, repeat it
        if test_audio_length < self.buffer_size:
            repeats_needed = self.buffer_size // test_audio_length + 1
            for i in range(repeats_needed):
                start_idx = i * test_audio_length
                end_idx = min(start_idx + test_audio_length, self.buffer_size)
                copy_length = end_idx - start_idx
                self.buffer[start_idx:end_idx] = self.audio_data[:copy_length]
        else:
            # If test audio is longer than buffer, use the first part
            self.buffer[:] = self.audio_data[: self.buffer_size]

    def start(self):
        """Start the mock recorder."""
        self.running = True
        # Start a background task to periodically signal new audio
        asyncio.create_task(self._simulate_audio_stream())
        logger.info("Started mock audio recorder")

    def stop(self):
        """Stop the mock recorder."""
        self.running = False
        logger.info("Stopped mock audio recorder")

    async def _simulate_audio_stream(self):
        """Simulate an audio stream by periodically signaling new audio."""
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        while self.running:
            # Advance the buffer index
            self.buffer_index = (self.buffer_index + chunk_size) % self.buffer_size

            # Signal that new audio is available
            self.new_audio_event.set()
            self.new_audio_event = asyncio.Event()

            # Wait before next update
            await asyncio.sleep(0.1)  # 100ms

    async def wait_for_new_audio(self, timeout=None):
        """
        Wait for new audio to become available.

        :param timeout: Maximum time to wait in seconds, or None to wait indefinitely
        :return: True if new audio is available, False if timeout occurred
        """
        if not self.running:
            return False

        try:
            await asyncio.wait_for(self.new_audio_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get the most recent audio of the specified duration.

        :param duration_seconds: Duration of audio to retrieve in seconds
        :return: Numpy array of audio data
        """
        frames = int(duration_seconds * self.sample_rate)

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

        # Update the last read index
        self.last_read_index = self.buffer_index

        return result


# Mock functions to replace the real ones in server.py

mock_continuous_recorder = None


def mock_initialize_audio_recorder(sample_rate=16000, device_id=None, buffer_seconds=30):
    """
    Initialize the global mock continuous audio recorder.

    :param sample_rate: Sample rate for recording (Hz)
    :param device_id: ID of the audio device to use (ignored in mock)
    :param buffer_seconds: Maximum duration of audio to keep in the buffer (seconds)
    """
    global mock_continuous_recorder

    if mock_continuous_recorder is not None:
        mock_continuous_recorder.stop()

    mock_continuous_recorder = MockContinuousAudioRecorder(
        sample_rate=sample_rate,
        channels=1,  # Mono audio
        device_id=device_id,
        buffer_seconds=buffer_seconds,
        test_file="test_audio.wav",  # Path to test audio file
    )
    mock_continuous_recorder.start()
    logger.info("Initialized global mock audio recorder with test file")


async def mock_get_audio_frames(duration_seconds: float) -> Tuple[np.ndarray, int]:
    """
    Get audio frames from the mock recorder.

    :param duration_seconds: Duration of audio to retrieve in seconds
    :return: Tuple of (audio data as numpy array, sample rate)
    """
    global mock_continuous_recorder

    if mock_continuous_recorder is None or not mock_continuous_recorder.running:
        logger.error("Mock audio recorder is not initialized or not running")
        # Return empty audio in case of error
        empty_audio = np.zeros((int(duration_seconds * 16000), 1), dtype=np.int16)
        return empty_audio, 16000

    # Get the audio data from the mock recorder
    audio_data = mock_continuous_recorder.get_audio(duration_seconds)
    return audio_data, mock_continuous_recorder.sample_rate
