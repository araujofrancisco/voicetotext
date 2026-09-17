# AGENTS.md

## Project

- `app.py` (~208 lines) -- CLI tool that transcribes audio via OpenAI Whisper. One segment per line, optional `[HH:MM:SS.mmm]` timestamps. GPU-only (exits with code 1 if no GPU).
- `web_app.py` (~414 lines) -- Flask web interface wrapping the same transcription logic. Routes: `GET /` (UI), `GET /gpu_info` (JSON), `POST /transcribe` (API). Upload audio, configure model/language/temperature/timestamps, view results inline with copy/download.
- `vram_utils.py` (~115 lines) -- Shared VRAM utilities: GPU detection, VRAM estimation, automatic model fallback/downgrade on pre-flight check or OOM.
- `requirements.txt` -- Lists `openai-whisper` and `flask`.

## Setup

- Venv at `voicetotext/` (non-standard name): `source voicetotext/bin/activate` or `voicetotext/bin/python app.py`
- Dependencies in `requirements.txt`: `openai-whisper`, `flask`
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

Serves on `http://0.0.0.0:5000`. Drag-and-drop upload, model/language/temperature controls, timestamp toggle, copy & download transcript. Displays selected GPU info at top of page. Shows toast notifications on model downgrade.

### Docker (web UI default)

```bash
docker build -t voicetotext:latest .
docker run --gpus all -p 5000:5000 voicetotext:latest
```

Open `http://localhost:5000`.

For CLI mode in Docker:
```bash
docker run --gpus all -v /path/to/audio:/data voicetotext:latest python app.py /data/file.wav --timestamps
```

## VRAM management

- **Pre-flight check**: Before loading a model, checks if any GPU has enough free memory. If not, auto-downgrades to the next smaller model in fallback order (`large → medium → small → base → tiny`).
- **OOM retry**: If a model OOMs during load or transcription, retries with the next smaller model automatically (up to 3 attempts).
- **Immediate VRAM release**: After transcription, `del model` + `gc.collect()` + `torch.cuda.empty_cache()` frees all VRAM. No idle reservation.
- **Multi-GPU**: If multiple GPUs are available, selects the one with most free memory. Parallelism across GPUs is not supported.
- **VRAM estimates**: `tiny=1.5GB`, `base=2.0GB`, `small=3.5GB`, `medium=9.0GB`, `large=16.0GB`. If even `tiny` OOMs, returns a clear error message.

## Verification

No tests. Verify manually using these markers from stdout:
- `>>> SCRIPT STARTED <<<` / `>>> SCRIPT FINISHED <<<` (CLI) or `>>> WEB APP STARTED <<<` (Web UI) bracket execution
- 5-line preview printed before save (CLI)
- File-not-found exits with code 1 and an error message
- **VRAM fallback**: Requesting a model larger than available VRAM triggers auto-downgrade (check logs for downgrade messages)
- **OOM retry**: OOM during transcription cascades to smaller models automatically
- **Web UI /gpu_info**: Returns JSON with `gpus` array and `selected` GPU object

Full checklist in `docs/development.md`.

## Gotchas

- **GPU-only**: No CPU fallback mode. Exits with code 1 if CUDA is not available.
- `condition_on_previous_text=False` is hardcoded in both CLI and Web UI -- improves speed, avoids failure loops. Change in `transcribe_audio()` or the web route if per-segment context consistency matters.
- **`torch.load` patching**: Both `app.py` and `web_app.py` patch `torch.load` to set `weights_only=False` temporarily, avoiding errors with newer PyTorch versions that default to `weights_only=True`. Uses try/finally to restore the original function.
- `format_transcript()` (CLI) assumes Whisper's segment schema (`result["segments"]` with `start`, `end`, `text`). Verify on Whisper upgrades.
- `verbose=False` suppresses Whisper's internal logging; set to `True` temporarily for debugging.

## Architecture

- `app.py` (~208 lines): CLI flow -- `main() → parse_arguments → get_gpu_info/find_best_gpu → load_whisper_model → transcribe_audio → format_transcript → save_transcript`. Includes nested try/except for OOM retry.
- `web_app.py` (~414 lines): Flask app with embedded HTML template. Routes: `GET /` (UI), `GET /gpu_info` (JSON), `POST /transcribe` (API). Reuses same transcription logic with pre-flight VRAM check and OOM retry loop.
- `vram_utils.py` (~115 lines): Shared utilities -- `get_gpu_info()`, `find_best_gpu()`, `find_best_gpu_from_list()`, `get_next_smaller_model()`, `MODEL_VRAM_ESTIMATES`, `format_vram()`, `format_gpu_info()`.

Detailed architecture in `docs/architecture.md`.
