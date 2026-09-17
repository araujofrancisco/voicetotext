# Architecture

## Entrypoints

### `app.py` (~208 lines) -- CLI entrypoint

A single-file Python script that transcribes audio via OpenAI Whisper. Runs via `python app.py <audio_file>`.

### `web_app.py` (~414 lines) -- Flask web UI

Serves a browser interface at `http://0.0.0.0:5000` with drag-and-drop upload, model/language/temperature controls, timestamp toggle, and copy/download transcript. Embeds an HTML template inline.

### `vram_utils.py` (~115 lines) -- Shared VRAM utilities

Provides GPU detection, VRAM estimation, automatic model fallback/downgrade logic, and formatting helpers. Used by both `app.py` and `web_app.py`.

## Execution flow (CLI)

```
main()                           # app.py:108
  ├─ parse_arguments()           # app.py:22   -- CLI arg parsing
  ├─ GPU check                   # app.py:123  -- torch.cuda.is_available()
  ├─ get_gpu_info()              # vram_utils  -- enumerate all CUDA devices
  ├─ find_best_gpu()             # vram_utils  -- select GPU + auto-downgrade model
  ├─ load_whisper_model()        # app.py:61   -- model load + torch.load patching
  ├─ transcribe_audio()          # app.py:89   -- Whisper transcribe call
  ├─ del model / gc / empty_cache# app.py:104  -- free VRAM immediately
  ├─ format_transcript()         # app.py:50   -- segments -> lines, optional timestamps
  ├─ save_transcript()           # app.py:102  -- write to file + print preview
  └─ (exit)
```

## Execution flow (Web UI)

```
GET /                          # web_app.py:308  -- serve HTML template
GET /gpu_info                  # web_app.py:317  -- JSON with GPU list + selected GPU
POST /transcribe               # web_app.py:324  -- handle upload, transcribe, return JSON
```

### `/transcribe` route flow

```
POST /transcribe
  ├─ validate file upload
  ├─ CUDA check
  ├─ pre-flight VRAM check (web_app.py:350)
  │   └─ if insufficient VRAM → auto-downgrade model via get_next_smaller_model()
  ├─ save uploaded file to temp
  ├─ try/loop with OOM retry (max 3 attempts)
  │   ├─ load_whisper_model()
  │   ├─ model.transcribe()
  │   ├─ del model / gc / empty_cache
  │   └─ if OOM → downgrade model and retry; tiny OOM → error response
  ├─ format segments inline (no separate function)
  └─ return JSON: {transcript, segments, model_used?, downgraded_from?}
```

## Module breakdown

### `parse_arguments()` (app.py:22)

Uses `argparse`. Accepts `--model`, `--output`, `--language`, `--timestamps`, `--temperature`. Returns a `Namespace` with all options. No `--device` or `--threads` flags exist in the current codebase; GPU is required and CPU threading is auto-detected by PyTorch.

### `load_whisper_model()` (app.py:61 / web_app.py:270)

Accepts optional `device` parameter (default `"cuda"`).
- **Patches `torch.load`** to set `weights_only=False` temporarily, avoiding errors with newer PyTorch versions that default to `weights_only=True`. Uses try/finally to restore the original function.
- Loads via `whisper.load_model(model_name, device=device)`.
- After loading, calls `gc.collect()` and `torch.cuda.empty_cache()`.
- Wraps errors: missing `openai-whisper` or `torch` raises `ImportError`; other failures raise `RuntimeError`.

### `transcribe_audio()` (app.py:89)

Accepts `temperature` (float, default `0.0`) and `condition_on_previous_text` (bool, default `False`).
- Calls `model.transcribe(file_path, language=language, verbose=False, temperature=temperature, condition_on_previous_text=condition_on_previous_text)`.
- `verbose=False` suppresses Whisper's internal progress output.
- Returns the raw Whisper result dict (keys: `text`, `segments`, `language`).

### `format_transcript()` (app.py:50) / inline formatting in web_app.py

- Iterates `result["segments"]` -- each segment has `start` (float seconds), `end`, `text`.
- Strips each segment's text and skips empty ones.
- If timestamps are enabled, formats `start` as `[HH:MM:SS.mmm]` via `format_time()`.
- Joins lines with `"\n"`.

### `format_time()` (app.py:42)

Converts float seconds to `HH:MM:SS.mmm` string.

### `save_transcript()` (app.py:102)

- Writes formatted text to the output path (UTF-8).
- On failure raises `RuntimeError`.

## VRAM utilities (`vram_utils.py`)

### Constants

- `MODEL_VRAM_ESTIMATES`: Dict mapping model names to estimated VRAM in bytes (`tiny=1.5GB`, `base=2.0GB`, `small=3.5GB`, `medium=9.0GB`, `large=16.0GB`).
- `MODEL_FALLBACK_ORDER`: `["tiny", "base", "small", "medium", "large"]` -- used for automatic downgrade cascading.

### Functions

- **`get_gpu_info()`** -- Returns a list of dicts with GPU info (`index`, `name`, `total_bytes`, `free_bytes`, `used_bytes`) for each available CUDA device.
- **`get_model_vram(model_name)`** -- Returns estimated VRAM in bytes for a model name (with buffer).
- **`format_vram(bytes_val)`** -- Converts bytes to human-readable string (e.g., `"5.5 GB"`).
- **`format_gpu_info(gpus)`** -- Human-readable GPU string for logging.
- **`find_best_gpu_from_list(gpus)`** -- Returns the GPU with the most free memory from a list.
- **`find_best_gpu(preferred_model)`** -- Main VRAM orchestration:
  1. Sorts GPUs by free memory descending.
  2. Checks if `preferred_model` fits on any GPU.
  3. If not, tries smaller models in fallback order (`tiny → base → small → medium → large`).
  4. Returns `(gpu_index, model_used, None | downgraded_from_original)`.
  5. If nothing fits, falls back to `tiny` on the best GPU (let torch handle OOM).
- **`get_next_smaller_model(current_model)`** -- Returns the next smaller model in fallback order, or `None` if already at smallest (`tiny`).

## Memory management

1. Pre-flight VRAM checks prevent unnecessary model loads when VRAM is insufficient.
2. `del model` + `gc.collect()` + `torch.cuda.empty_cache()` frees VRAM immediately after transcription (both CLI and web).
3. No idle VRAM reservation -- formatting and file I/O happen post-cleanup.
4. OOM retry loops cascade through model sizes automatically in both CLI and web routes.

## Error handling

- `faulthandler.enable()` at the top of both files dumps a C-level traceback on segfaults.
- `logging` is configured with `INFO` level, writing to stdout.
- Known errors (missing file, missing dependencies, transcription failure) raise `RuntimeError` caught in `main()`.
- Unexpected exceptions are caught as generic `Exception` and logged.
- Web UI returns JSON error responses with appropriate HTTP status codes.

## Dependencies

Only two direct Python packages at runtime:

- `openai-whisper` -- model loading, audio processing, transcription
- `torch` -- tensor computation, CUDA support, device management
- `flask` -- web framework (web_app.py only)

Plus `ffmpeg` on PATH (invoked by Whisper's audio loader).
