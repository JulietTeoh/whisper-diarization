# Whisper Diarization Server

An OpenAI-compatible API server for speech transcription with speaker diarization, built with FastAPI. It supports long audio files, GPU acceleration, and dynamic workflow configuration per request.

## ✨ Features

### Core Functionality
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint.
- **Speaker Diarization**: Identifies and labels different speakers using Pyannote.audio 3.1.
- **Word-Level Timestamps**: Provides accurate timestamps for each word.
- **Long Audio Support**: Handles large audio files through intelligent chunking.
- **Multiple Response Formats**: Supports JSON, text, SRT, VTT, and verbose JSON.

### Performance & Efficiency
- **Singleton Model Management**: All models are loaded once on startup for optimal performance and memory usage, managed by a central `ModelManager`.
- **GPU Acceleration**: Supports CUDA for all major operations (Whisper, Demucs, Pyannote).
- **Fast Response Times**: Eliminates cold starts for individual requests by pre-loading models.
- **Concurrent Processing**: Handles multiple requests simultaneously and processes audio chunks in parallel.

### Advanced & Configurable Features
- **Vocal Separation (Demucs)**: Optional vocal extraction to improve transcription accuracy on noisy audio.
- **Automatic Punctuation**: Restores punctuation for supported languages.
- **Dynamic Workflows**: Enable or disable features like vocal separation and punctuation on a per-request basis.
- **Modular Architecture**: Clean separation of services for audio, transcription, and diarization.

## 🚀 Quick Start

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **System Dependencies**:
    ```bash
    # Install ffmpeg (required for audio processing)
    sudo apt install ffmpeg  # Ubuntu/Debian
    brew install ffmpeg      # macOS
    ```

3.  **Environment Configuration**:
    ```bash
    cp .env.example .env
    # Edit .env with your settings (see Configuration section)
    ```

4.  **Start the Server**:
    ```bash
    python start_server.py
    ```
    The server will start on `http://localhost:8000` by default. Startup may take a minute as models are loaded.

## API Usage

### Transcribe Audio

Send a POST request to `/v1/audio/transcriptions`.

**Example cURL Request**:
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "enable_vocal_separation=true" \
  -F "enable_punctuation=true"
```

### Supported Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | `UploadFile` | **Required**. The audio file to transcribe. |
| `model` | `str` | **Required**. Must be `"whisper-1"`. |
| `language` | `str` | Optional. The language of the audio. If omitted, it will be auto-detected. |
| `prompt` | `str` | Optional. A prompt to guide the model's transcription. |
| `response_format`| `str` | Optional. The format of the response. Default: `json`. Options: `json`, `text`, `srt`, `vtt`, `verbose_json`. |
| `temperature` | `float` | Optional. The sampling temperature. Default: `0.0`. |
| `enable_vocal_separation` | `bool` | Optional. If `true`, separates vocals from background noise using Demucs. Overrides server config. |
| `enable_punctuation` | `bool` | Optional. If `true`, restores punctuation. Overrides server config. |
| `timestamp_granularities[]` | `List[str]` | Optional. Specify `segment` or `word` to include timestamps. |

## 🛠️ Configuration

Settings are managed via a `.env` file.

```ini
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Model Configuration
WHISPER_MODEL=medium.en   # Whisper model size (e.g., tiny, base, small, medium, large-v3)
DEVICE=auto               # auto, cuda, or cpu
HUGGINGFACE_TOKEN=your_hf_token_here # Required for Pyannote 3.1

# Workflow Configuration (Global Defaults)
ENABLE_VOCAL_SEPARATION=true   # Default for vocal separation (can be overridden per request)
ENABLE_SPEAKER_DIARIZATION=true # Default for speaker diarization
ENABLE_PUNCTUATION_RESTORATION=true # Default for punctuation (can be overridden per request)

# Audio Processing
BATCH_SIZE=8
CHUNK_LENGTH_MINUTES=10
MAX_AUDIO_DURATION_HOURS=24

# Temporary Files
TEMP_DIR=./tmp
CLEANUP_TEMP_FILES=true
```

## 🏛️ Architecture

The server is built with a modular and efficient architecture:

- **`server.py`**: The main FastAPI application, defining endpoints and managing request lifecycle.
- **`services/`**: Contains the core logic for each processing step.
  - `transcription.py`: The main orchestrator that coordinates the transcription workflow.
  - `audio_service.py`: Handles audio loading, validation, vocal separation, and chunking.
  - `whisper_service.py`: Manages transcription using the FasterWhisper model.
  - `diarization_service.py`: Performs speaker diarization and word alignment.
  - `chunk_processor.py`: Processes long audio files by managing chunks concurrently.
- **`utils/`**: Provides shared utilities.
  - `model_manager.py`: A singleton that loads and provides access to all models, preventing redundant memory usage.
  - `response_formatter.py`: Formats the final output into JSON, SRT, VTT, etc.
  - `temp_manager.py`: Manages temporary directories for audio files and chunks.
- **`config.py`**: Loads and validates configuration from the `.env` file.
- **`models.py`**: Defines Pydantic models for API request and response validation.

## ⚙️ Performance

- **Memory**: The `ModelManager` ensures that each model is loaded only once, significantly reducing memory footprint. With a `medium.en` model, expect ~2-4GB of GPU memory usage.
- **Speed**: By pre-loading models, request processing is fast. Performance depends on the hardware (GPU is highly recommended) and the selected workflow. Disabling vocal separation can speed up processing by ~3x.
- **Concurrency**: The server can handle multiple requests in parallel, with configurable limits for concurrent chunk processing to manage resources effectively.

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: 
   - Reduce `BATCH_SIZE` in `.env`
   - Disable `ENABLE_VOCAL_SEPARATION` 
   - Use a smaller `WHISPER_MODEL` (e.g., `base`, `small`).

2. **Slow Server Startup**: 
   - This is normal, as models are loaded into memory upfront to ensure fast request processing.
   - To speed up startup for development, you can disable features in your `.env` file.

3. **Audio Format Not Supported**: 
   - Ensure `ffmpeg` is installed and accessible in your system's PATH.

4. **HuggingFace Authentication**:
   - A HuggingFace token (`HUGGINGFACE_TOKEN`) is required in your `.env` file to use the `pyannote/speaker-diarization-3.1` model.

### Debugging:
- Server logs are printed to standard output.
- Set `LOG_LEVEL=DEBUG` in your `.env` file for more detailed logs.
- Monitor GPU memory usage with `nvidia-smi`.

## License

This project is available under the MIT License.
