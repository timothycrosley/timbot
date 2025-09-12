# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Timbot is a multimodal AI assistant that communicates through voice, vision, and text. The system uses a modular architecture with specialized components:

- **brain.py**: Core LLM communication using Ollama and Gemma model for natural language processing
- **ears.py**: Audio input capture and sound detection using sounddevice and numpy
- **eyes.py**: Webcam screenshot capture at regular intervals using OpenCV
- **mouth.py**: Text-to-speech output using Piper TTS engine
- **main.py**: CLI interface and orchestration using Typer, manages multimodal interaction loops
- **async_downloader.py**: Handles concurrent downloading of required models and assets

## Key Dependencies & Models

- **Ollama**: Local LLM inference (Gemma model)
- **Piper TTS**: Voice synthesis (en_US-lessac-medium model)
- **OpenCV**: Camera capture and image processing
- **sounddevice**: Real-time audio I/O
- **Faster Whisper**: Speech-to-text transcription (base model)
- Models are downloaded on-demand to `/models` directory

## Common Development Commands

### Setup and Model Installation
```bash
python -m timbot download-models  # Downloads all required AI models
```

### Running the Application
```bash
python -m timbot start  # Starts interactive multimodal mode
```

### Linting
```bash
ruff check src/  # Code linting
```

## Development Notes

- Models directory (`/models`) contains downloaded AI models and is gitignored except for README.md
- Audio files are temporarily stored in `/tmp/` and automatically cleaned up
- Screenshots are taken every 10 seconds and stored in a circular buffer (max 4)
- The system requires Ollama to be installed and running for brain functionality
- Audio processing uses 44.1kHz sample rate with int16 format
- Speech input is transcribed to text using Whisper before being sent to the brain
- All input (typed and spoken) is unified as text when processed by the brain
- Image capture is downscaled to 320x240 for performance

## System Dependencies

Must be installed on the system:
- Ollama (auto-installed by download-models command)
- Working microphone and camera
- Audio output device

The application handles graceful degradation if components are unavailable.