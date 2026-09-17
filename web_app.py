#!/usr/bin/env python3
"""
Web interface for Whisper audio transcription.
Serves a UI that wraps app.py's transcription logic via Flask.
"""

import os
import sys
import gc
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

import vram_utils

import faulthandler
faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice to Text</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
.container { max-width: 800px; width: 100%; padding: 2rem 1.5rem; }
h1 { font-size: 1.8rem; font-weight: 600; text-align: center; margin-bottom: 0.3rem; color: #fff; }
.subtitle { text-align: center; color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
.card { background: #1a1a1a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #2a2a2a; }
.drop-zone { border: 2px dashed #3a3a3a; border-radius: 10px; padding: 2.5rem 1.5rem; text-align: center; cursor: pointer; transition: all 0.2s; position: relative; }
.drop-zone:hover, .drop-zone.drag-over { border-color: #4f8cff; background: #1a1f2e; }
.drop-zone.has-file { border-color: #3cc65f; border-style: solid; }
.drop-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.drop-text { color: #aaa; font-size: 0.95rem; }
.drop-text span { color: #4f8cff; text-decoration: underline; }
.file-info { margin-top: 0.8rem; font-size: 0.85rem; color: #3cc65f; display: none; }
.file-info.visible { display: block; }
.settings { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
@media (max-width: 600px) { .settings { grid-template-columns: 1fr; } }
.field label { display: block; font-size: 0.8rem; color: #888; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.5px; }
.field select, .field input[type="text"], .field input[type="number"] { width: 100%; padding: 0.6rem 0.8rem; background: #0f0f0f; border: 1px solid #2a2a2a; border-radius: 8px; color: #e0e0e0; font-size: 0.9rem; outline: none; transition: border-color 0.2s; }
.field select:focus, .field input:focus { border-color: #4f8cff; }
.toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 0; }
.toggle-row label { text-transform: none; letter-spacing: 0; font-size: 0.9rem; color: #ccc; margin-bottom: 0; }
.toggle { position: relative; width: 44px; height: 24px; }
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle .slider { position: absolute; inset: 0; background: #333; border-radius: 12px; cursor: pointer; transition: 0.2s; }
.toggle .slider::before { content: ''; position: absolute; width: 18px; height: 18px; left: 3px; top: 3px; background: #fff; border-radius: 50%; transition: 0.2s; }
.toggle input:checked + .slider { background: #4f8cff; }
.toggle input:checked + .slider::before { transform: translateX(20px); }
.btn { width: 100%; padding: 0.9rem; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.btn-primary { background: #4f8cff; color: #fff; }
.btn-primary:hover:not(:disabled) { background: #3a7aff; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.status { text-align: center; padding: 1rem; font-size: 0.9rem; min-height: 2.5rem; }
.status.idle { color: #888; }
.status.loading { color: #4f8cff; }
.status.success { color: #3cc65f; }
.status.error { color: #ff5c5c; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #4f8cff; border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 0.5rem; }
@keyframes spin { to { transform: rotate(360deg); } }
.transcript-output { display: none; }
.transcript-output.visible { display: block; }
.transcript-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.transcript-header h2 { font-size: 1.1rem; font-weight: 600; }
.btn-group { display: flex; gap: 0.5rem; }
.btn-sm { padding: 0.4rem 0.8rem; border: 1px solid #2a2a2a; background: #0f0f0f; color: #ccc; border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
.btn-sm:hover { border-color: #4f8cff; color: #fff; }
.transcript-text { background: #0f0f0f; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; line-height: 1.7; white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; color: #d0d0d0; }
.toast { position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%) translateY(100px); background: #333; color: #fff; padding: 0.7rem 1.2rem; border-radius: 8px; font-size: 0.85rem; opacity: 0; transition: all 0.3s; pointer-events: none; }
.toast.show { transform: translateX(-50%) translateY(0); opacity: 1; }
</style>
</head>
<body>
<div class="container">
<h1>Voice to Text</h1>
<p class="subtitle">Transcribe audio with Whisper — powered by GPU</p>
<p class="subtitle" id="gpuInfo" style="display:none; color: #6a9f5b;"></p>

<div class="card">
<div class="drop-zone" id="dropZone">
<input type="file" id="fileInput" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm">
<div class="drop-icon">&#127911;</div>
<div class="drop-text">Drag & drop an audio file or <span>browse</span></div>
<div class="file-info" id="fileInfo"></div>
</div>
</div>

<div class="card">
<div class="settings">
<div class="field">
<label for="modelSelect">Model</label>
<select id="modelSelect">
<option value="tiny">Tiny (fastest)</option>
<option value="base">Base</option>
<option value="small" selected>Small (recommended)</option>
<option value="medium">Medium</option>
<option value="large">Large (most accurate)</option>
</select>
</div>
<div class="field">
<label for="languageInput">Language</label>
<input type="text" id="languageInput" placeholder="Auto-detect (leave empty)">
</div>
<div class="field">
<label for="tempSlider">Temperature</label>
<input type="number" id="tempSlider" min="0" max="1" step="0.1" value="0.0">
</div>
</div>
<div class="toggle-row">
<label for="tsToggle">Include timestamps</label>
<div class="toggle">
<label for="tsToggle">
<input type="checkbox" id="tsToggle">
<span class="slider"></span>
</label>
</div>
</div>
</div>

<button class="btn btn-primary" id="transcribeBtn" disabled>Transcribe</button>

<div class="status idle" id="status">Upload an audio file to begin</div>

<div class="transcript-output" id="transcriptOutput">
<div class="card">
<div class="transcript-header">
<h2>Transcript</h2>
<div class="btn-group">
<button class="btn-sm" id="copyBtn">Copy</button>
<button class="btn-sm" id="downloadBtn">Download .txt</button>
</div>
</div>
<div class="transcript-text" id="transcriptText"></div>
</div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const transcribeBtn = document.getElementById('transcribeBtn');
const status = document.getElementById('status');
const transcriptOutput = document.getElementById('transcriptOutput');
const transcriptText = document.getElementById('transcriptText');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const toast = document.getElementById('toast');

let selectedFile = null;
let lastTranscript = '';

fetch('/gpu_info')
  .then(r => r.json())
  .then(data => {
    if (data.selected) {
      const gpuInfoEl = document.getElementById('gpuInfo');
      gpuInfoEl.textContent = `Selected GPU: ${data.selected.name} (${(data.selected.free_bytes / 1024 / 1024 / 1024).toFixed(1)}GB free)`;
      gpuInfoEl.style.display = 'block';
    }
  })
  .catch(() => {});

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2000);
}

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  selectedFile = file;
  fileInfo.textContent = `${file.name} — ${formatSize(file.size)}`;
  fileInfo.classList.add('visible');
  dropZone.classList.add('has-file');
  transcribeBtn.disabled = false;
  status.className = 'status idle';
  status.textContent = 'Ready to transcribe';
  transcriptOutput.classList.remove('visible');
}

transcribeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  transcribeBtn.disabled = true;
  status.className = 'status loading';
  status.innerHTML = '<span class="spinner"></span>Transcribing...';
  transcriptOutput.classList.remove('visible');

  const formData = new FormData();
  formData.append('audio', selectedFile);
  formData.append('model', document.getElementById('modelSelect').value);
  formData.append('timestamps', document.getElementById('tsToggle').checked);
  formData.append('temperature', document.getElementById('tempSlider').value);
  const lang = document.getElementById('languageInput').value.trim();
  if (lang) formData.append('language', lang);

    try {
      const res = await fetch('/transcribe', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Transcription failed');
      lastTranscript = data.transcript;
      transcriptText.textContent = lastTranscript;
      transcriptOutput.classList.add('visible');
      status.className = 'status success';
      status.textContent = `Done — ${data.segments} segments`;
      if (data.downgraded_from) {
        showToast(`Model auto-downgraded: ${data.downgraded_from} → ${data.model_used}`);
      }
    } catch (err) {
    status.className = 'status error';
    status.textContent = err.message;
  } finally {
    transcribeBtn.disabled = false;
  }
});

copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(lastTranscript).then(() => showToast('Copied to clipboard'));
});

downloadBtn.addEventListener('click', () => {
  const blob = new Blob([lastTranscript], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (selectedFile?.name || 'transcript') + '.txt';
  a.click();
  URL.revokeObjectURL(url);
  showToast('Downloaded');
});
</script>
</body>
</html>
"""


def load_whisper_model(model_name):
    import torch
    import whisper
    logging.info(f"Loading Whisper model '{model_name}' on cuda...")

    # Patch torch.load to avoid "weights_only" errors with newer PyTorch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    try:
        model = whisper.load_model(model_name, device="cuda")
    finally:
        torch.load = _orig_torch_load

    gc.collect()
    torch.cuda.empty_cache()
    return model


@app.route("/")
def index():
    gpu_info = vram_utils.get_gpu_info()
    if gpu_info:
        logging.info(f"GPU info available: {len(gpu_info)} GPU(s)")
    else:
        logging.warning("No GPU info available")
    return render_template_string(HTML_TEMPLATE)


@app.route("/gpu_info")
def gpu_info_route():
    gpus = vram_utils.get_gpu_info()
    selected = None
    if gpus:
        selected = vram_utils.find_best_gpu_from_list(gpus)
    return jsonify(gpus=gpus, selected=selected)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify(error="No audio file uploaded"), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify(error="No file selected"), 400

    import torch
    if not torch.cuda.is_available():
        return jsonify(error="CUDA not available"), 500

    model_name = request.form.get("model", "small")
    include_timestamps = request.form.get("timestamps", "false").lower() == "true"
    temperature = float(request.form.get("temperature", 0.0))
    language = request.form.get("language", None) or None

    # Pre-flight VRAM check
    gpus = vram_utils.get_gpu_info()
    if gpus:
        best_gpu = vram_utils.find_best_gpu_from_list(gpus)
        if best_gpu:
            required = vram_utils.MODEL_VRAM_ESTIMATES.get(model_name)
            if required and best_gpu["free_bytes"] < required:
                logging.warning(
                    f"Insufficient VRAM on best GPU ({best_gpu['name']}: {best_gpu['free_bytes']/1024/1024:.0f}MB free, "
                    f"{required/1024/1024:.0f}MB required for '{model_name}')"
                )
                next_model = vram_utils.get_next_smaller_model(model_name)
                if next_model:
                    logging.info(f"Auto-downgrading model: {model_name} -> {next_model}")
                    model_name = next_model

    # Save uploaded file to temp
    ext = Path(file.filename).suffix
    tmp_path = tempfile.mktemp(suffix=ext)
    try:
        file.save(tmp_path)

        original_model = model_name
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                model = load_whisper_model(model_name)
                result = model.transcribe(
                    tmp_path, language=language, verbose=False,
                    temperature=temperature, condition_on_previous_text=False,
                )

                del model
                gc.collect()
                torch.cuda.empty_cache()

                # Format segments
                lines = []
                for seg in result["segments"]:
                    text = seg["text"].strip()
                    if not text:
                        continue
                    if include_timestamps:
                        h = int(seg["start"] // 3600)
                        m = int((seg["start"] % 3600) // 60)
                        s = int(seg["start"] % 60)
                        ms = int((seg["start"] % 1) * 1000)
                        text = f"[{h:02d}:{m:02d}:{s:02d}.{ms:03d}] {text}"
                    lines.append(text)

                transcript = "\n".join(lines)
                response = {"transcript": transcript, "segments": len(lines)}
                if model_name != original_model:
                    response["model_used"] = model_name
                    response["downgraded_from"] = original_model
                return jsonify(response)

            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"OOM on attempt {attempt + 1}: {e}")
                if 'model' in locals():
                    del model
                gc.collect()
                torch.cuda.empty_cache()

                if model_name == "tiny":
                    return jsonify(error="Out of memory — even 'tiny' model requires more VRAM than available"), 500

                next_model = vram_utils.get_next_smaller_model(model_name)
                if next_model:
                    logging.info(f"OOM retry: {model_name} -> {next_model}")
                    model_name = next_model
                else:
                    return jsonify(error="Out of memory — no smaller model available"), 500

    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print(">>> WEB APP STARTED <<<")
    app.run(host="0.0.0.0", port=5000, debug=False)
