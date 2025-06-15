# Real-Time Translation Server

A FastAPI-based real-time translation server that captures audio, transcribes speech, translates to multiple languages, and generates text-to-speech audio using ElevenLabs and OpenAI APIs.

## Features

- **Real-time Audio Processing**: Continuous audio recording and processing
- **Speech-to-Text**: Transcribe audio using ElevenLabs STT API
- **Multi-language Translation**: Translate text using OpenAI GPT-4 with context awareness
- **Text-to-Speech**: Generate audio in multiple languages using ElevenLabs TTS
- **Streaming APIs**: Real-time streaming of captions and audio
- **Web Interface**: Multiple frontend interfaces for different use cases
- **Admin Panel**: Configuration management and system control
- **Session Archiving**: Automatic archiving of conversations and audio

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd translation-server
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install system dependencies**
```bash
# For audio processing (Ubuntu/Debian)
sudo apt-get install portaudio19-dev python3-pyaudio

# For macOS
brew install portaudio
```

4. **Set up configuration**
Create `admin_config.json` with your API keys:
```json
{
    "elevenlabs_api_key": "your_elevenlabs_api_key",
    "openai_api_key": "your_openai_api_key",
    "admin_token": "your_admin_token",
    "publish_mode": "local-audio-source",
    "default_source_language": "ENGLISH",
    "target_languages": ["CHINESE", "VIETNAMESE", "THAI"]
}
```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:9001`

### Operating Modes

#### 1. Local Audio Source Mode
- Automatically captures audio from your microphone
- Processes audio at regular intervals
- Ideal for live presentations or meetings

#### 2. Upload Mode
- Accepts audio uploads via API
- Manual control over when audio is processed
- Suitable for batch processing or custom integrations

### Web Interfaces

1. **Main Interface**: `http://localhost:9001/` - View live translations
2. **Streaming Interface**: `http://localhost:9001/?streaming=true` - Real-time streaming view
3. **Upload Interface**: `http://localhost:9001/upload?api_key=YOUR_API_KEY` - Manual audio upload
4. **Admin Panel**: `http://localhost:9001/admin?admin_token=YOUR_TOKEN` - System configuration

## API Endpoints

### Public Endpoints

- `GET /` - Main web interface
- `GET /captions/speaking` - Get speaking captions
- `GET /captions/translated?lang={language}` - Get translated captions
- `GET /audio/translated-voice?lang={language}` - Get translated audio
- `GET /captions/speaking/stream` - Stream speaking captions (SSE)
- `GET /captions/translated/stream?lang={language}` - Stream translated captions (SSE)
- `GET /audio/translated-voice/stream?lang={language}` - Stream translated audio (SSE)

### Protected Endpoints (require API key)

- `POST /publish` - Upload audio for processing

### Admin Endpoints (require admin token)

- `GET /admin/config` - Get current configuration
- `POST /admin/config` - Update configuration
- `GET /admin/audio-status` - Get audio recorder status
- `POST /admin/audio-control` - Control audio recorder
- `GET /admin/audio-devices` - List available audio devices

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `elevenlabs_api_key` | ElevenLabs API key | "" |
| `openai_api_key` | OpenAI API key | "" |
| `publish_mode` | "local-audio-source" or "upload" | "local-audio-source" |
| `default_source_language` | Source language for translation | "ENGLISH" |
| `target_languages` | List of target languages | ["CHINESE", "VIETNAMESE", "THAI"] |
| `tts_model` | ElevenLabs TTS model | "eleven_multilingual_v2" |
| `stt_model` | ElevenLabs STT model | "scribe_v1" |
| `audio_device_id` | Audio input device ID | null (default) |
| `poll_interval` | Seconds between recordings | 10 |
| `first_record_seconds` | Initial recording duration | 5 |
| `min_audio_duration` | Minimum audio duration for STT | 15 |

## Supported Languages

- English
- Chinese (Simplified)
- Vietnamese
- Thai
- Spanish
- French
- German
- Japanese
- Korean
- Portuguese
- Russian
- Arabic
- Hindi
- Italian
- Dutch

## Architecture

### Core Components

1. **ContinuousAudioRecorder**: Manages real-time audio capture
2. **PublishedDataStore**: In-memory storage for real-time data
3. **ConversationCache**: Maintains translation context
4. **SessionArchiver**: Archives conversations and audio files

### Audio Processing Pipeline

1. **Audio Capture** → Continuous recording with noise filtering
2. **Speech-to-Text** → ElevenLabs STT API
3. **Translation** → OpenAI GPT-4 with context awareness
4. **Text-to-Speech** → ElevenLabs TTS API
5. **Distribution** → Real-time streaming to clients

### Translation Features

- **Two-pass translation**: Surface translation + refinement
- **Context awareness**: Maintains conversation history
- **Special terms**: Custom terminology preservation
- **Consistency**: Cross-reference with previous translations

## Development

### Project Structure

```
├── app.py                 # Main application
├── i18n.py               # Internationalization and translation prompts
├── utils.py              # Utility classes (ConversationCache, SessionArchiver)
├── templates/            # HTML templates
│   ├── index.html
│   ├── streaming-frontend.html
│   ├── upload-frontend.html
│   └── admin-frontend.html
├── static/               # Static assets
└── admin_config.json     # Configuration file
```

### Key Classes

- `ContinuousAudioRecorder`: Thread-safe audio recording with circular buffer
- `PublishedDataStore`: Async data store with event-driven updates
- `ConversationCache`: Translation context management
- `SessionArchiver`: Automatic session archiving

### Audio Processing

- **Noise filtering**: High-pass and low-pass filters with noise gate
- **Resampling**: Automatic sample rate conversion
- **Format detection**: Support for multiple audio formats
- **Circular buffering**: Efficient memory usage for continuous recording

## Troubleshooting

### Common Issues

1. **Audio device not found**
   - Check available devices: `GET /admin/audio-devices`
   - Update `audio_device_id` in configuration

2. **API key errors**
   - Verify ElevenLabs and OpenAI API keys
   - Check API quotas and limits

3. **Audio quality issues**
   - Adjust noise filtering parameters
   - Check microphone settings
   - Verify sample rate compatibility

4. **Translation context issues**
   - Clear conversation cache via admin panel
   - Adjust translation history length

### Logging

The application provides detailed logging for debugging:
- Audio processing events
- API call timing and results
- Translation context and history
- Error handling and recovery

