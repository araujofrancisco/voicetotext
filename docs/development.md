# Development guide

## Environment

The project uses a single Python virtual environment at `voicetotext/`. All development work happens in `app.py`.

```bash
# Activate
source voicetotext/bin/activate

# Verify
which python  # should point to .../voicetotext/bin/python
```

## Dependency management

There is no `requirements.txt`. The virtual environment itself is the source of truth.

```bash
# List installed packages
voicetotext/bin/pip list

# Add a new dependency
voicetotext/bin/pip install <package>

# Freeze for reproducibility (if needed)
voicetotext/bin/pip freeze > requirements.txt
```

## Workflow

1. Edit `app.py`
2. Run with sample audio:

```bash
python app.py some_test_file.mp3 --model tiny --device cpu
```

An agent should confirm the file exists before running — models download on first use and take time.

## Manual verification checklist

After any change, verify:

- [ ] Script starts (`>>> SCRIPT STARTED <<<`)
- [ ] Audio file not-found error works
- [ ] Transcription produces output
- [ ] Preview shows first 5 lines
- [ ] `--timestamps` includes `[HH:MM:SS.mmm]` format
- [ ] `--language` forces the expected language
- [ ] `--device cpu` works on any machine
- [ ] `--output` writes to the specified path
- [ ] `--threads 4` overrides CPU thread count
- [ ] `--temperature 0.2` produces output (non-deterministic)
- [ ] Script ends with `>>> SCRIPT FINISHED <<<`

## Adding a new model size

Whisper supports `tiny`, `base`, `small`, `medium`, `large`. To add a new model (e.g., `large-v3`):

1. Add the name to the `choices` list in `parse_arguments()`.
2. Verify it loads with `whisper.load_model()`.

## Adding a new CLI option

1. Add the argument in `parse_arguments()`.
2. Pass the value through the call chain in `main()`.
3. Use it in the relevant function.

## Known quirks

- `torch.set_num_threads()` is only called when `--threads` is passed. By default, PyTorch auto-detects thread count.
- `verbose=False` on `model.transcribe()` suppresses Whisper's own logging. If debugging, temporarily set to `True`.
- The `format_transcript()` function expects Whisper's segment format (`result["segments"]` with `start`, `end`, `text` keys). If upgrading Whisper, verify the result schema.
- `condition_on_previous_text=False` is passed to improve speed and avoid failure loops. If per-segment context consistency matters, change it to `True` in `transcribe_audio()`.
