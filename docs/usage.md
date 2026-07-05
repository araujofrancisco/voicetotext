# Usage guide

## Basic transcription

```bash
python app.py recording.mp3
```

Uses the `small` model and outputs to `transcript.txt`. A 5-line preview is printed to the terminal.

## Selecting a model

Whisper models trade speed for accuracy:

| Model | Disk | Speed | Quality |
|---|---|---|---|
| `tiny` | ~150 MB | fastest | lowest |
| `base` | ~300 MB | fast | low |
| `small` | ~1.5 GB | moderate | good |
| `medium` | ~3.5 GB | slow | better |
| `large` | ~6 GB | slowest | best |

```bash
python app.py recording.mp3 --model tiny
python app.py recording.mp3 --model large
```

Models are cached in `~/.cache/whisper/` after the first download and reused.

## Forcing language

By default, Whisper auto-detects the language. To skip detection and force a language:

```bash
python app.py recording.mp3 --language es     # Spanish
python app.py recording.mp3 --language fr     # French
python app.py recording.mp3 --language ja     # Japanese
```

Use ISO 639-1 two-letter codes. Forcing the language can improve accuracy and speed.

## Timestamps

Add `--timestamps` to prepend `[HH:MM:SS.mmm]` to each line:

```bash
python app.py recording.mp3 --timestamps
```

Output example:
```
[00:00:00.000] Good morning everyone.
[00:00:03.520] Today we're going to talk about machine learning.
[00:00:07.140] Let's start with the basics.
```

## Device selection

CUDA is auto-detected. To override:

```bash
python app.py recording.mp3 --device cpu
```

CPU transcription is slower but uses less RAM. CUDA transcription requires an NVIDIA GPU with sufficient VRAM (varies by model; ~2 GB+ for `small`).

## Custom output path

```bash
python app.py recording.mp3 --output custom_output.txt
```

## Combining options

```bash
# Fast CPU transcription with timestamps
python app.py meeting.wav --model tiny --device cpu --timestamps

# High-quality Spanish transcription
python app.py entrevista.wav --model large --language es
```
