# Whisper Diarization - Enhanced Fork

A powerful speaker diarization and transcription solution built upon OpenAI's Whisper, featuring both batch processing and an OpenAI-compatible API server.

## 🔗 Original Project

This project is a fork of [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization.git), enhanced with additional features including:

- **OpenAI-Compatible API Server**: Drop-in replacement for OpenAI's transcription API
- **Pyannote.audio 3.1 Integration**: Advanced speaker diarization capabilities
- **Enhanced Performance**: Singleton model management and GPU optimization
- **Extended Functionality**: Configurable workflows and improved architecture

## ✨ Features

### Core Capabilities
- **Speaker Diarization**: Identifies and labels different speakers in audio files
- **High-Quality Transcription**: Leverages OpenAI Whisper for accurate speech-to-text
- **Word-Level Timestamps**: Precise timing information for each word
- **Multiple Output Formats**: JSON, SRT, VTT, and plain text support
- **Long Audio Support**: Handles large files through intelligent chunking

### Performance & Optimization
- **GPU Acceleration**: CUDA support for all major operations
- **Parallel Processing**: Concurrent chunk processing for large files
- **Model Caching**: Singleton pattern prevents redundant model loading
- **Fast Response Times**: Pre-loaded models eliminate cold starts

### Advanced Features
- **Vocal Separation**: Optional background noise removal using Demucs
- **Automatic Punctuation**: Multilingual punctuation restoration
- **Configurable Workflows**: Enable/disable features per request
- **OpenAI API Compatibility**: Seamless integration with existing tools

## 🚀 Quick Start

### Installation

1. **Prerequisites**:
   ```bash
   # Install system dependencies
   sudo apt install ffmpeg cython3  # Ubuntu/Debian
   brew install ffmpeg              # macOS
   
   # Install Python dependencies
   pip install -c constraints.txt -r requirements.txt
   ```

2. **Configuration** (for server mode):
   ```bash
   cp .env.example .env
   # Edit .env with your settings (HuggingFace token required)
   ```

### Usage Options

#### Option 1: Batch Processing (Original Method)
```bash
# Basic usage
python diarize.py -a your_audio_file.mp3

# Advanced options
python diarize.py -a audio.mp3 --whisper-model large-v3 --device cuda --language en
```

#### Option 2: API Server (Enhanced)
```bash
# Start the server
python start_server.py

# Use with cURL
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json" \
  -F "enable_vocal_separation=true"
```

## 📖 API Documentation

### Endpoints

- **POST** `/v1/audio/transcriptions` - OpenAI-compatible transcription endpoint
- **GET** `/v1/models` - List available models
- **GET** `/health` - Health check

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File | **Required**. Audio file to transcribe |
| `model` | String | **Required**. Must be "whisper-1" |
| `language` | String | Audio language (auto-detected if omitted) |
| `response_format` | String | Output format: json, text, srt, vtt, verbose_json |
| `enable_vocal_separation` | Boolean | Use Demucs for vocal extraction |
| `enable_punctuation` | Boolean | Restore punctuation |
| `timestamp_granularities` | Array | Include segment/word timestamps |

## ⚙️ Configuration

Create a `.env` file for server configuration:

```ini
# Server Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Configuration
WHISPER_MODEL=medium.en
DEVICE=auto
HUGGINGFACE_TOKEN=your_token_here

# Workflow Defaults
ENABLE_VOCAL_SEPARATION=true
ENABLE_SPEAKER_DIARIZATION=true
ENABLE_PUNCTUATION_RESTORATION=true

# Performance
BATCH_SIZE=8
CHUNK_LENGTH_MINUTES=10
MAX_AUDIO_DURATION_HOURS=24
```

## 🏗️ Architecture

```
├── server.py              # FastAPI application
├── services/              # Core processing logic
│   ├── transcription.py   # Main workflow orchestrator
│   ├── audio_service.py   # Audio processing and chunking
│   ├── whisper_service.py # Whisper transcription
│   └── diarization_service.py # Speaker diarization
├── utils/                 # Shared utilities
│   ├── model_manager.py   # Singleton model management
│   ├── response_formatter.py # Output formatting
│   └── temp_manager.py    # Temporary file handling
├── diarize.py            # Original batch processing script
└── config.py             # Configuration management
```

## 🔧 Command Line Options

### Batch Processing
- `-a AUDIO_FILE`: Audio file to process
- `--whisper-model`: Whisper model size (tiny, base, small, medium, large-v3)
- `--device`: Processing device (auto, cuda, cpu)
- `--language`: Audio language (auto-detected if omitted)
- `--batch-size`: Batch size for inference
- `--no-stem`: Disable vocal separation
- `--suppress_numerals`: Convert numbers to words

## 🚧 Known Limitations

- Overlapping speakers not yet fully supported
- Very long audio files may require chunking adjustments
- GPU memory requirements scale with model size and batch size

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- **Original Project**: [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization.git)
- **OpenAI Whisper**: [openai/whisper](https://github.com/openai/whisper)
- **Faster Whisper**: [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
- **Pyannote.audio**: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **NVIDIA NeMo**: [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
- **Facebook Demucs**: [facebookresearch/demucs](https://github.com/facebookresearch/demucs)

Special thanks to [@adamjonas](https://github.com/adamjonas) for supporting the original whisper diarization project.

## 📄 License

This project is licensed under the same terms as the original project. See [LICENSE](LICENSE) for details.

---

For more detailed documentation, see:
- [Server Documentation](SERVER_README.md)
- [OpenAI API Compatibility](OPENAI_API.md)
- [Future Plans](FUTURE.md)