# Architecture

## Entrypoint

`app.py` — a single 135-line Python script with no package structure. The script is executable (`#!/usr/bin/env python3`) and runs via `python app.py`.

## Execution flow

```
main()                           # app.py:98
  ├─ parse_arguments()           # app.py:22  — CLI arg parsing + CUDA auto-detect
  ├─ load_whisper_model()        # app.py:62  — model download/load + memory cleanup
  ├─ transcribe_audio()          # app.py:81  — Whisper transcribe call
  ├─ format_transcript()         # app.py:46  — segments → lines, optional timestamps
  ├─ save_transcript()           # app.py:90  — write to file + print preview
  └─ (exit)
```

## Module breakdown

### `parse_arguments()` (line 22)

Uses `argparse`. Imports `torch` conditionally to check `torch.cuda.is_available()` and set the default device string. Returns a `Namespace` with all options.

### `load_whisper_model()` (line 62)

- Calls `torch.set_num_threads(1)` before loading to prevent thread contention with PyTorch's internal parallelism.
- Loads the model via `whisper.load_model(model_name, device=device)`.
- After loading, calls `gc.collect()` and `torch.cuda.empty_cache()` to reclaim memory.
- Wraps errors: missing `openai-whisper` or `torch` raises `ImportError` with install instructions; other failures raise `RuntimeError`.

### `transcribe_audio()` (line 81)

- Calls `model.transcribe(file_path, language=language, verbose=False)`.
- `verbose=False` suppresses Whisper's internal progress output.
- Returns the raw Whisper result dict (keys: `text`, `segments`, `language`).

### `format_transcript()` (line 46)

- Iterates `result["segments"]` — each segment has `start` (float seconds), `end`, `text`.
- Strips each segment's text and skips empty ones.
- If `--timestamps` is set, formats `start` as `[HH:MM:SS.mmm]` via `format_time()`.
- Joins lines with `"\n"`.

### `format_time()` (line 38)

Converts float seconds to `HH:MM:SS.mmm` string.

### `save_transcript()` (line 90)

- Writes formatted text to the output path (UTF-8).
- Prints the first 5 lines to stdout as a preview preview.
- On failure raises `RuntimeError`.

### `main()` (line 98)

- Prints `>>> SCRIPT STARTED <<<` on entry.
- Validates the audio file exists.
- Catches `RuntimeError` for known failures and generic `Exception` for unexpected issues.
- Prints `>>> SCRIPT FINISHED <<<` and `sys.exit(0)` regardless of outcome (errors are logged, not re-raised).

## Error handling

- `faulthandler.enable()` at the top of the file dumps a C-level traceback on segfaults.
- `logging` is configured with `INFO` level, writing to stdout.
- Known errors (missing file, missing dependencies, transcription failure) raise `RuntimeError` caught in `main()`.
- Unexpected exceptions are caught as generic `Exception` and logged.

## Memory management

1. `torch.set_num_threads(1)` reduces CPU thread pool size before model load.
2. `gc.collect()` forces Python garbage collection after model load.
3. `torch.cuda.empty_cache()` frees unused CUDA memory after model load.

This pattern exists because Whisper models are large and memory pressure can be significant, especially on GPU.

## Dependencies

Only two direct Python packages at runtime:

- `openai-whisper` — model loading, audio processing, transcription
- `torch` — tensor computation, CUDA support, device management

Plus `ffmpeg` on PATH (invoked by Whisper's audio loader).
