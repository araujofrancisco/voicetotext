# Voice-to-Text

Transcribe audio files using OpenAI's Whisper model. Outputs one sentence or segment per line, with optional timestamps.

## Requirements

- **Python** 3.12+
- **ffmpeg** on PATH (Whisper uses ffmpeg to decode audio)
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
pip install openai-whisper torch
```

## Quick start

```bash
python app.py recording.mp3
```

This uses the `small` model (default) on the best available device (CUDA if available, otherwise CPU). Output goes to `transcript.txt`.

## CLI reference

```
python app.py <audio_file> [options]
```

| Option | Default | Description |
|---|---|---|---|
| `audio_file` | — | Path to audio file (required) |
| `--model` | `small` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--output` | `transcript.txt` | Output file path |
| `--device` | auto | `cpu` or `cuda` (auto-detected from torch) |
| `--language` | auto | Force language code, e.g. `en`, `es`, `fr` |
| `--timestamps` | off | Prepend `[HH:MM:SS.mmm]` to each line |
| `--threads` | auto | CPU threads for PyTorch (default: auto-detect) |
| `--temperature` | `0.0` | Decode temperature; 0 is deterministic, higher enables fallback |

## Examples

```bash
# Basic transcription
python app.py lecture.mp3

# Spanish with timestamps
python app.py entrevista.wav --language es --timestamps

# Use tiny model on CPU for speed
python app.py meeting.m4a --model tiny --device cpu

# Custom output path
python app.py podcast.mp3 --output podcast.txt

# CPU with 4 threads and deterministic decoding
python app.py meeting.wav --model tiny --device cpu --threads 4
```

## How it works

1. Whisper model loads onto the selected device (with memory cleanup after load)
2. Audio is decoded via ffmpeg and fed to the model
3. Whisper returns segmented transcription with timestamps
4. Script formats each segment as a line, optionally timestamped
5. Output is written to file and a 5-line preview is printed

## Project structure

```
.
├── app.py              # CLI entrypoint (single file)
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
