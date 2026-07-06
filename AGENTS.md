# AGENTS.md

## Project

Single-file CLI tool (`app.py`) that transcribes audio using OpenAI Whisper (`openai-whisper` v20250625 + `torch` 2.12.0). Outputs one sentence/segment per line, optionally with `[HH:MM:SS.mmm]` timestamps.

## Quick start

```bash
source voicetotext/bin/activate
python app.py audio.mp3
```

Or run via the venv Python directly:

```bash
voicetotext/bin/python app.py audio.mp3
```

## CLI

```
python app.py <audio_file> [--model {tiny,base,small,medium,large}] [--output transcript.txt] [--device {cpu,cuda}] [--language <code>] [--timestamps] [--threads <n>] [--temperature <n>]
```

- Device auto-detects CUDA; pass `--device cpu` to override.
- Requires `ffmpeg` on PATH (Whisper decodes audio via ffmpeg).
- Model downloads on first run (~1.5 GB for `small` default).
- Output path default is `transcript.txt`; a 5-line preview is printed to stdout.
- `--threads` controls CPU thread count (default: PyTorch auto-detect).
- `--temperature` controls decode temperature; 0 = deterministic (default), higher values enable fallback on low-quality segments.

## Dependencies

Installed in `voicetotext/` venv (Python 3.12.3). No `requirements.txt` — use `voicetotext/bin/pip list` to inspect. Key: `openai-whisper`, `torch`, plus `nvidia-*` CUDA packages.

## No tests / no CI / no linting

None present. Verify output manually.

## Architecture

Only file: `app.py:134` (`main()`). Flow: `parse_args → load_whisper_model → transcribe_audio → format_transcript → save_transcript`. Memory: calls `gc.collect()` + `torch.cuda.empty_cache()` after model load. `faulthandler` is enabled for crash debugging. `torch.set_num_threads(threads)` only called when `--threads` is passed.
