# Development guide

## Environment

The project uses a single Python virtual environment at `voicetotext/`. All development work happens across `app.py`, `web_app.py`, and `vram_utils.py`.

```bash
# Activate
source voicetotext/bin/activate

# Verify
which python  # should point to .../voicetotext/bin/python
```

## Dependency management

Dependencies are listed in `requirements.txt`:

```
openai-whisper
flask
```

```bash
# Install from requirements.txt
pip install -r requirements.txt

# List installed packages
voicetotext/bin/pip list

# Add a new dependency
voicetotext/bin/pip install <package>
voicetotext/bin/pip freeze > requirements.txt
```

## Workflow

1. Edit the relevant file (`app.py`, `web_app.py`, or `vram_utils.py`)
2. Run with sample audio:

```bash
# CLI
python app.py some_test_file.mp3 --model tiny

# Web UI
python web_app.py  # then open http://localhost:5000
```

An agent should confirm the file exists before running -- models download on first use and take time.

## Manual verification checklist

After any change, verify:

- [ ] Script starts (`>>> SCRIPT STARTED <<<` / `>>> WEB APP STARTED <<<`)
- [ ] Audio file not-found error works (exits with code 1)
- [ ] Transcription produces output
- [ ] Preview shows first 5 lines (CLI) or transcript displays in browser (Web UI)
- [ ] `--timestamps` includes `[HH:MM:SS.mmm]` format
- [ ] `--language` forces the expected language
- [ ] `--output` writes to the specified path (CLI)
- [ ] `--temperature 0.2` produces output (non-deterministic)
- [ ] Script ends with `>>> SCRIPT FINISHED <<<` (CLI)
- [ ] **VRAM pre-flight check**: requesting a model larger than available VRAM auto-downgrades to a smaller model
- [ ] **OOM retry**: if a model OOMs during load/transcribe, it cascades down to the next smaller model automatically
- [ ] **tiny OOM**: if even `tiny` fails with OOM, a clear error message is returned (not a crash)
- [ ] **VRAM cleanup**: VRAM is freed immediately after transcription (`del model` + `gc.collect()` + `torch.cuda.empty_cache()`)
- [ ] **Web UI GPU info**: `/gpu_info` endpoint returns JSON with `gpus` list and `selected` GPU
- [ ] **Web UI toast notifications**: model downgrade events show a toast in the browser

## Adding a new model size

Whisper supports `tiny`, `base`, `small`, `medium`, `large`. To add a new model (e.g., `large-v3`):

1. Add the name to the `choices` list in `parse_arguments()`.
2. Add VRAM estimate to `MODEL_VRAM_ESTIMATES` in `vram_utils.py`.
3. Add to `MODEL_FALLBACK_ORDER` in `vram_utils.py` (in size order).
4. Verify it loads with `whisper.load_model()`.

## Adding a new CLI option

1. Add the argument in `parse_arguments()`.
2. Pass the value through the call chain in `main()`.
3. Use it in the relevant function.

## Web UI changes

The web app uses an embedded HTML template (`HTML_TEMPLATE` string). Changes to the UI require editing this string directly. Key areas:

- Settings grid (model/language/temperature): lines ~160-175 in `web_app.py`
- JavaScript transcribe handler: lines ~240-265 in `web_app.py`
- Toast notification: defined in CSS + JS, triggered on model downgrade

## Known quirks

- **GPU-only**: Transcription requires CUDA. The tool exits with code 1 if no GPU is available -- there is no CPU fallback mode.
- **`torch.load` patching**: Both `app.py` and `web_app.py` patch `torch.load` to set `weights_only=False` temporarily, avoiding errors with newer PyTorch versions that default to `weights_only=True`. This is done via try/finally to restore the original function after loading.
- **`verbose=False`** on `model.transcribe()` suppresses Whisper's own logging. If debugging, temporarily set to `True`.
- The `format_transcript()` function (CLI) and inline formatting (Web UI) expect Whisper's segment format (`result["segments"]` with `start`, `end`, `text` keys). If upgrading Whisper, verify the result schema.
- **`condition_on_previous_text=False`** is hardcoded in both CLI and Web UI -- improves speed and avoids failure loops. If per-segment context consistency matters, change it to `True` in `transcribe_audio()` or the web route.
