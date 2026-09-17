# Usage guide

## Basic transcription

```bash
python app.py recording.mp3
```

Uses the `small` model (default) on the best available GPU and outputs to `transcript.txt`. A 5-line preview is printed to the terminal.

## Selecting a model

Whisper models trade speed for accuracy:

| Model | Disk | Estimated VRAM | Speed | Quality |
|---|---|---|---|---|
| `tiny` | ~150 MB | ~1.5 GB | fastest | lowest |
| `base` | ~300 MB | ~2.0 GB | fast | low |
| `small` | ~1.5 GB | ~3.5 GB | moderate | good |
| `medium` | ~3.5 GB | ~9.0 GB | slow | better |
| `large` | ~6 GB | ~16.0 GB | slowest | best |

```bash
python app.py recording.mp3 --model tiny
python app.py recording.mp3 --model large
```

Models are cached in `~/.cache/whisper/` after the first download and reused.

### VRAM-based automatic fallback

If you request a model that exceeds your GPU's available VRAM, the tool automatically downgrades to a smaller model:

- **Pre-flight check**: Before loading, if no GPU has enough free memory for the requested model, it selects the next smaller model in the fallback order (`large → medium → small → base → tiny`).
- **OOM retry**: If a model OOMs during load or transcription, it retries with the next smaller model automatically (up to 3 attempts).
- **Final error**: If even `tiny` fails with OOM, a clear error is returned: `"Out of memory -- even 'tiny' model requires more VRAM than available"`.

This happens transparently in both CLI and Web UI modes.

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

## GPU requirements

Transcription is **GPU-only** (CUDA required). The tool automatically selects the GPU with the most free memory if multiple GPUs are available.

To check your GPU info (CLI):
```bash
python app.py recording.mp3 --model tiny  # logs GPU info before loading
```

To check in the Web UI, open the dashboard -- it displays the selected GPU and its free VRAM at the top.

## Custom output path

```bash
python app.py recording.mp3 --output custom_output.txt
```

## Temperature

Temperature controls how deterministic the transcription is:

- `0.0` (default) -- single deterministic pass, fastest
- Higher values -- enable fallback retries on segments with low confidence

```bash
# Use temperature fallback chain (up to 6 attempts per segment)
python app.py recording.mp3 --temperature 0.2
```

The default `0.0` is sufficient for most recordings. Raise it if you see repeated or garbled segments.

## Combining options

```bash
# Fast transcription on low-VRAM GPU with timestamps
python app.py meeting.wav --model tiny --timestamps

# High-quality Spanish transcription
python app.py entrevista.wav --model large --language es
```

---

## Web UI usage

Start the web server:

```bash
python web_app.py
```

Open `http://localhost:5000` in a browser.

### Features

- **Drag-and-drop upload**: Drop audio files onto the upload zone, or click to browse.
- **Model selector**: Choose from `tiny`, `base`, `small`, `medium`, `large`.
- **Language input**: Leave empty for auto-detect, or enter a language code (e.g., `es`).
- **Temperature slider**: Set decode temperature (0.0–1.0).
- **Timestamps toggle**: Enable `[HH:MM:SS.mmm]` timestamps on each line.
- **Copy / Download**: Copy transcript to clipboard or download as `.txt`.
- **GPU info display**: Shows the selected GPU name and free VRAM at the top of the page.
- **Model downgrade toast**: If the requested model is auto-downgraded due to VRAM limits, a notification appears at the bottom of the screen.

### Docker deployment

```bash
docker build -t voicetotext:latest .
docker run --gpus all -p 5000:5000 voicetotext:latest
```

Open `http://localhost:5000`. The container uses a PyTorch CUDA base image for faster builds.

For CLI mode in Docker:
```bash
docker run --gpus all -v /path/to/audio:/data voicetotext:latest python app.py /data/file.wav --timestamps
```
