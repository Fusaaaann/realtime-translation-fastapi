import json
import os
import time
import logging
import threading


logger = logging.getLogger(__name__)


class ConversationCache:
    """
    Manages conversation context for better TTS continuity and translation caching.
    """

    def __init__(self, max_cache_size: int = 1000):
        self.translation_cache = {}  # {speech_id: {lang: text}}
        self.conversation_history = {}  # {speech_id: [{"timestamp": float, "text": str, "lang": str}]}
        self.max_cache_size = max_cache_size
        self.lock = threading.RLock()
        logger.info(f"Initialized ConversationCache with max size {max_cache_size}")

    def cache_translation(self, speech_id: str, language: str, translated_text: str):
        """Cache a translation for a specific speech ID and language."""
        with self.lock:
            if speech_id not in self.translation_cache:
                self.translation_cache[speech_id] = {}
            self.translation_cache[speech_id][language] = translated_text

            # Maintain cache size
            if len(self.translation_cache) > self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.translation_cache.keys())[: -self.max_cache_size]
                for key in oldest_keys:
                    del self.translation_cache[key]

            logger.debug(f"Cached translation for {speech_id}[{language}]: {translated_text[:50]}...")

    def get_previous_translation(self, speech_id: str, language: str) -> str:
        """Get the previous translation for context in TTS."""
        with self.lock:
            if speech_id and speech_id in self.translation_cache:
                return self.translation_cache[speech_id].get(language, "")
            return ""

    def add_conversation_entry(self, speech_id: str, text: str, language: str, timestamp: float):
        """Add an entry to conversation history."""
        with self.lock:
            if speech_id not in self.conversation_history:
                self.conversation_history[speech_id] = []

            self.conversation_history[speech_id].append({"timestamp": timestamp, "text": text, "language": language})

            # Keep only recent entries per conversation
            max_history_per_conversation = 50
            if len(self.conversation_history[speech_id]) > max_history_per_conversation:
                self.conversation_history[speech_id] = self.conversation_history[speech_id][-max_history_per_conversation:]

    def get_conversation_context(self, speech_id: str, language: str = None, max_entries: int = 5) -> list:
        """Get recent conversation context for a speech ID."""
        with self.lock:
            if speech_id not in self.conversation_history:
                return []

            history = self.conversation_history[speech_id]
            if language:
                history = [entry for entry in history if entry["language"] == language]

            return history[-max_entries:] if history else []

    def clear_conversation(self, speech_id: str):
        """Clear conversation history for a specific speech ID."""
        with self.lock:
            self.translation_cache.pop(speech_id, None)
            self.conversation_history.pop(speech_id, None)
            logger.info(f"Cleared conversation cache for {speech_id}")


class SessionArchiver:
    """
    Archives session data for dataset compilation and analysis.
    """

    def __init__(self, archive_dir: str = "session_archives"):
        self.archive_dir = archive_dir
        self.session_id = f"session_{int(time.time())}"
        self.session_data = []
        self.lock = threading.RLock()

        # Create archive directory and subdirectories
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "tts_audio"), exist_ok=True)
        logger.info(f"Initialized SessionArchiver with session ID: {self.session_id}")

    async def archive_session_data(
        self,
        speech_id: str,
        transcribed_text: str,
        source_language: str,
        timestamp: float,
        admin_config: dict,
        audio_data: bytes = None,
        translations: dict = None,
        tts_audio_data: dict = None,
    ):
        """Archive a complete session entry with all related data including TTS audio."""
        with self.lock:
            entry = {
                "speech_id": speech_id,
                "session_id": self.session_id,
                "timestamp": timestamp,
                "transcribed_text": transcribed_text,
                "source_language": source_language,
                "translations": translations or {},
                "audio_size": len(audio_data) if audio_data else 0,
                "tts_audio_files": {},
                "created_at": time.time(),
            }

            self.session_data.append(entry)
            # TODO: save whole speece, not each piece
            # Save original audio file if provided
            if audio_data:
                audio_filename = f"{self.session_id}_{speech_id}_{int(timestamp)}_original.wav"  # TODO: match original format, not assuming it as wav
                audio_path = os.path.join(self.archive_dir, "audio", audio_filename)
                try:
                    with open(audio_path, "wb") as f:
                        f.write(audio_data)
                    entry["audio_file"] = f"audio/{audio_filename}"
                    logger.debug(f"Archived original audio file: {audio_filename}")
                except Exception as e:
                    logger.error(f"Failed to archive original audio file: {e}")

            # Save TTS audio files if provided
            if tts_audio_data:
                for language, tts_audio_bytes in tts_audio_data.items():
                    if tts_audio_bytes:
                        tts_filename = f"{self.session_id}_{speech_id}_{int(timestamp)}_{language}_tts.mp3"
                        tts_path = os.path.join(self.archive_dir, "tts_audio", tts_filename)
                        try:
                            with open(tts_path, "wb") as f:
                                f.write(tts_audio_bytes)
                            entry["tts_audio_files"][language] = f"tts_audio/{tts_filename}"
                            logger.debug(f"Archived TTS audio file for {language}: {tts_filename}")
                        except Exception as e:
                            logger.error(f"Failed to archive TTS audio file for {language}: {e}")

            # Periodically save session data to JSON
            if len(self.session_data) % 10 == 0:  # Save every 10 entries
                await self._save_session_data(admin_config)

    async def _save_session_data(self, admin_config):
        """Save session data to JSON file."""
        try:
            session_file = os.path.join(self.archive_dir, f"{self.session_id}.json")
            session_summary = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "total_entries": len(self.session_data),
                "config_snapshot": admin_config.copy(),
                "entries": self.session_data,
            }

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved session data: {len(self.session_data)} entries to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")

    async def finalize_session(self, admin_config):
        """Finalize and save the complete session data."""
        await self._save_session_data(admin_config)
        logger.info(f"Finalized session {self.session_id} with {len(self.session_data)} entries")

    def get_session_stats(self) -> dict:
        """Get statistics about the current session."""
        with self.lock:
            if not self.session_data:
                return {"session_id": self.session_id, "total_entries": 0}

            languages = set()
            total_audio_size = 0
            tts_files_count = 0

            for entry in self.session_data:
                languages.add(entry["source_language"])
                languages.update(entry["translations"].keys())
                total_audio_size += entry.get("audio_size", 0)
                tts_files_count += len(entry.get("tts_audio_files", {}))

            return {
                "session_id": self.session_id,
                "total_entries": len(self.session_data),
                "languages_used": list(languages),
                "total_audio_size_bytes": total_audio_size,
                "tts_files_count": tts_files_count,
                "session_duration_seconds": time.time() - (self.session_data[0]["timestamp"] if self.session_data else time.time()),
            }
