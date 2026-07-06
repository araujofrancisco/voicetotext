# Architecture

## Entrypoint

`app.py` — a single-file Python script (~150 lines). The script is executable (`#!/usr/bin/env python3`) and runs via `python app.py`.

## Execution flow

```
main()                           # app.py:108
  ├─ parse_arguments()           # app.py:22   — CLI arg parsing + CUDA auto-detect
  ├─ load_whisper_model()        # app.py:66   — model download/load + memory cleanup
  ├─ transcribe_audio()          # app.py:86   — Whisper transcribe call
  ├─ format_transcript()         # app.py:50   — segments → lines, optional timestamps
  ├─ save_transcript()           # app.py:100  — write to file + print preview
  └─ (exit)
```

## Module breakdown

### `parse_arguments()` (line 22)

Uses `argparse`. Imports `torch` conditionally to check `torch.cuda.is_available()` and set the default device string. Returns a `Namespace` with all options, including `--threads` (default `None` → auto-detect) and `--temperature` (default `0.0`).

### `load_whisper_model()` (line 66)

Accepts optional `threads` parameter. If set, calls `torch.set_num_threads(threads)` before loading; otherwise PyTorch auto-detects thread count.
- Loads the model via `whisper.load_model(model_name, device=device)`.
- After loading, calls `gc.collect()` and `torch.cuda.empty_cache()` to reclaim memory.
- Wraps errors: missing `openai-whisper` or `torch` raises `ImportError` with install instructions; other failures raise `RuntimeError`.

### `transcribe_audio()` (line 86)

Accepts `temperature` (float, default `0.0`) and `condition_on_previous_text` (bool, default `False`).
- Calls `model.transcribe(file_path, language=language, verbose=False, temperature=temperature, condition_on_previous_text=condition_on_previous_text)`.
- `verbose=False` suppresses Whisper's internal progress output.
- Returns the raw Whisper result dict (keys: `text`, `segments`, `language`).

### `format_transcript()` (line 50)

- Iterates `result["segments"]` — each segment has `start` (float seconds), `end`, `text`.
- Strips each segment's text and skips empty ones.
- If `--timestamps` is set, formats `start` as `[HH:MM:SS.mmm]` via `format_time()`.
- Joins lines with `"\n"`.

### `format_time()` (line 42)

Converts float seconds to `HH:MM:SS.mmm` string.

### `save_transcript()` (line 100)

- Writes formatted text to the output path (UTF-8).
- Prints the first 5 lines to stdout as a preview.
- On failure raises `RuntimeError`.

### `main()` (line 108)

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

1. `gc.collect()` forces Python garbage collection after model load.
2. `torch.cuda.empty_cache()` frees unused CUDA memory after model load.

This pattern exists because Whisper models are large and memory pressure can be significant, especially on GPU.

## Dependencies

Only two direct Python packages at runtime:

- `openai-whisper` — model loading, audio processing, transcription
- `torch` — tensor computation, CUDA support, device management

Plus `ffmpeg` on PATH (invoked by Whisper's audio loader).
