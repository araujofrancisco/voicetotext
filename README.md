# Voice-to-Text

Transcribe audio files using OpenAI's Whisper model. Outputs one sentence or segment per line, with optional timestamps.

## Requirements

- **Python** 3.12+
- **ffmpeg** on PATH (Whisper uses ffmpeg to decode audio)
- **NVIDIA GPU** with CUDA support (transcription is GPU-only; exits with code 1 if no GPU detected)
- ~1.5 GB disk per model download (varies by size)

## Setup

A virtual environment is already set up at `voicetotext/`. Activate it:

```bash
source voicetotext/bin/activate
```

Or run directly without activating:

```bash
voicetotext/bin/python app.py <audio_file>
```

### Installing from scratch

```bash
python3 -m venv venv
source venv/bin/activate
pip install openai-whisper torch flask
```

## Quick start

```bash
python app.py recording.mp3
```

This uses the `small` model (default) on the best available GPU. Output goes to `transcript.txt`.

### VRAM requirements

| Model | Estimated VRAM | Notes |
|---|---|---|
| `tiny` | ~1.5 GB | Fastest, lowest accuracy |
| `base` | ~2.0 GB | — |
| `small` | ~3.5 GB | Default, good balance |
| `medium` | ~9.0 GB | — |
| `large` | ~16.0 GB | Most accurate |

If the requested model exceeds available VRAM, the tool automatically downgrades to a smaller model and retries. If even `tiny` OOMs, a clear error is returned.

## CLI reference

```
python app.py <audio_file> [options]
```

| Option | Default | Description |
|---|---|---|
| `audio_file` | — | Path to audio file (required) |
| `--model` | `small` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--output` | `transcript.txt` | Output file path |
| `--language` | auto | Force language code, e.g. `en`, `es`, `fr` |
| `--timestamps` | off | Prepend `[HH:MM:SS.mmm]` to each line |
| `--temperature` | `0.0` | Decode temperature; 0 is deterministic, higher enables fallback |

## Examples

```bash
# Basic transcription
python app.py lecture.mp3

# Spanish with timestamps
python app.py entrevista.wav --language es --timestamps

# Use tiny model for low-VRAM GPUs
python app.py meeting.m4a --model tiny

# Custom output path
python app.py podcast.mp3 --output podcast.txt

# Deterministic decoding at temperature 0.2 (enables fallback retries)
python app.py meeting.wav --temperature 0.2
```

## How it works

1. GPU is detected; the device with most free VRAM is selected
2. Pre-flight VRAM check ensures the model fits on the selected GPU, downgrading if needed
3. Whisper model loads (with `torch.load` patched for newer PyTorch compatibility)
4. Audio is decoded via ffmpeg and fed to the model
5. Whisper returns segmented transcription with timestamps
6. Script formats each segment as a line, optionally timestamped
7. Output is written to file, VRAM is freed immediately (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`)
8. A 5-line preview is printed to stdout

## Project structure

```
.
├── app.py              # CLI entrypoint (~208 lines)
├── web_app.py          # Flask web UI (~414 lines)
├── vram_utils.py       # Shared VRAM utilities (GPU detection, fallback logic)
├── requirements.txt    # Python dependencies: openai-whisper, flask
├── Dockerfile          # Containerized deployment (PyTorch CUDA base image)
├── .dockerignore
├── voicetotext/        # Python virtual environment
│   ├── bin/            # Activation scripts, pip, python
│   ├── lib/            # Site-packages (whisper, torch, etc.)
│   └── pyvenv.cfg
├── README.md
└── docs/
    ├── usage.md
    ├── architecture.md
    └── development.md
```
