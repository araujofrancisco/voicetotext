# AGENTS.md

## Project

- `app.py` — CLI tool that transcribes audio via OpenAI Whisper. One segment per line, optional `[HH:MM:SS.mmm]` timestamps.
- `web_app.py` — Flask web interface wrapping the same transcription logic. Upload audio, configure model/language/timestamps, view results inline with copy/download.

## Setup

- Venv at `voicetotext/` (non-standard name): `source voicetotext/bin/activate` or `voicetotext/bin/python app.py`
- No `requirements.txt` — inspect with `voicetotext/bin/pip list`
- `ffmpeg` required on PATH
- Models cached in `~/.cache/whisper/` after first download

## Run

```bash
python app.py <audio_file> [--model {tiny,base,small,medium,large}] [--output transcript.txt] [--language <code>] [--timestamps] [--temperature <float>]
```

Default: `--model small`, GPU-only (CUDA required), output to `transcript.txt`. Exits with code 1 if no GPU detected.

## Web UI

```bash
python web_app.py
```

Serves on `http://0.0.0.0:5000`. Drag-and-drop upload, model/language/temperature controls, timestamp toggle, copy & download transcript.

### Docker (web UI default)

```bash
docker build -t voicetotext:latest .
docker run --gpus all -p 5000:5000 voicetotext:latest
```

Open `http://localhost:5000`.

For CLI mode in Docker:
```bash
docker run --gpus all -v /path/to/audio:/data voicetotext:latest python app.py /data/file.wav --timestamps

## VRAM management

Model loads on demand and VRAM is freed immediately after transcription (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`). No idle VRAM reservation — formatting and file I/O happen post-cleanup.

## Verification

No tests. Verify manually using these markers from stdout:
- `>>> SCRIPT STARTED <<<` / `>>> SCRIPT FINISHED <<<` bracket execution
- 5-line preview printed before save
- File-not-found exits with code 1 and an error message

Full checklist in `docs/development.md`.

## Gotchas

- `condition_on_previous_text=False` is hardcoded in `main()` — improves speed, avoids failure loops. Change in `transcribe_audio()` if per-segment context consistency matters.
- `format_transcript()` assumes Whisper's segment schema (`result["segments"]` with `start`, `end`, `text`). Verify on Whisper upgrades.
- `verbose=False` suppresses Whisper's internal logging; set to `True` temporarily for debugging.

## Architecture

- `app.py` (150 lines): CLI flow — `main() → parse_arguments → load_whisper_model → transcribe_audio → format_transcript → save_transcript`.
- `web_app.py`: Flask app with embedded HTML template. Routes: `GET /` (UI), `POST /transcribe` (API). Reuses same transcription logic with immediate VRAM release.

Detailed architecture in `docs/architecture.md`.
