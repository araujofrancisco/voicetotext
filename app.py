#!/usr/bin/env python3
"""
Audio Transcription Script using OpenAI Whisper.
Outputs one line per sentence/segment, with optional timestamps.
"""

import argparse
import os
import sys
import logging
import gc

import faulthandler
faulthandler.enable()

import vram_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper model.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size. (default: small)")
    parser.add_argument("--output", type=str, default="transcript.txt", help="Output text file path.")
    parser.add_argument("--language", type=str, default=None, help="Force language code (e.g., 'en', 'es').")
    parser.add_argument("--timestamps", action="store_true", 
                        help="Include [HH:MM:SS.mmm] timestamps at the start of each line.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decode temperature. 0 is deterministic; higher values enable fallback attempts. (default: 0.0)")
    return parser.parse_args()

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def format_transcript(result: dict, include_timestamps: bool) -> str:
    """Format Whisper segments into one line per sentence, optionally with timestamps."""
    lines = []
    for segment in result["segments"]:
        text = segment["text"].strip()
        if not text:
            continue
        
        if include_timestamps:
            start_time = format_time(segment["start"])
            lines.append(f"[{start_time}] {text}")
        else:
            lines.append(text)
            
    return "\n".join(lines)

def load_whisper_model(model_name: str, device: str = "cuda"):
    try:
        import torch
        import whisper
        
        logging.info(f"Loading Whisper model '{model_name}' on {device}...")

        # Patch torch.load to avoid "weights_only" errors with newer PyTorch
        _orig_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        try:
            model = whisper.load_model(model_name, device=device)
        finally:
            torch.load = _orig_torch_load
        
        gc.collect()
        torch.cuda.empty_cache()
            
        return model
    except ImportError as e:
        raise ImportError("Missing dependencies. Run: pip install openai-whisper torch") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

def transcribe_audio(model, file_path: str, language: str | None = None,
                     temperature: float = 0.0, condition_on_previous_text: bool = False) -> dict:
    try:
        logging.info(f"Starting transcription of: {os.path.basename(file_path)}")
        # verbose=False keeps Whisper's internal logging quiet; we handle our own output
        result = model.transcribe(
            file_path, language=language, verbose=False,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

def save_transcript(text: str, output_path: str) -> None:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Successfully saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write output file: {e}") from e

def main() -> None:
    print(">>> SCRIPT STARTED <<<")
    args = parse_arguments()

    if not os.path.exists(args.audio_file):
        logging.error(f"File not found: {args.audio_file}")
        print(">>> SCRIPT EXITED (File not found) <<<")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        logging.error("CUDA not available — GPU transcription required.")
        print(">>> SCRIPT EXITED (No GPU) <<<")
        sys.exit(1)

    gpu_info = vram_utils.get_gpu_info()
    if gpu_info:
        logging.info(f"GPUs: {vram_utils.format_gpu_info(gpu_info)}")
    
    gpu_idx, model_name, downgraded_from = vram_utils.find_best_gpu(args.model)
    
    if downgraded_from:
        needed_vram = vram_utils.get_model_vram(downgraded_from)
        logging.warning(
            f"Model '{downgraded_from}' needs {vram_utils.format_vram(needed_vram)} — "
            f"no GPU has enough; downgrading to '{model_name}'"
        )

    try:
        try:
            model = load_whisper_model(model_name, device=f"cuda:{gpu_idx}")
            
            raw_result = transcribe_audio(
                model, args.audio_file, args.language,
                temperature=args.temperature,
                condition_on_previous_text=False,
            )
            
            # Free VRAM immediately after transcription — idle VRAM is wasted on this hardware
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                while True:
                    smaller = vram_utils.get_next_smaller_model(model_name)
                    if smaller is None:
                        raise RuntimeError(
                            f"CUDA OOM even with 'tiny' — GPU VRAM insufficient for transcription."
                        ) from e
                    logging.error(
                        f"CUDA out of memory — downgrading from '{model_name}' to '{smaller}' and retrying..."
                    )
                    model_name = smaller
                    try:
                        model = load_whisper_model(model_name, device=f"cuda:{gpu_idx}")
                        raw_result = transcribe_audio(
                            model, args.audio_file, args.language,
                            temperature=args.temperature,
                            condition_on_previous_text=False,
                        )
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                    except RuntimeError as retry_e:
                        if "CUDA out of memory" in str(retry_e) or "out of memory" in str(retry_e).lower():
                            continue
                        raise retry_e from e
            else:
                raise
        
        # Format the output based on the --timestamps flag
        formatted_transcript = format_transcript(raw_result, args.timestamps)
        
        save_transcript(formatted_transcript, args.output)
        
        print("\n--- Transcription Preview ---")
        # Show first 5 lines or the whole thing if it's short
        preview_lines = formatted_transcript.split('\n')[:5]
        preview_text = '\n'.join(preview_lines)
        if len(formatted_transcript.split('\n')) > 5:
            preview_text += "\n..."
            
        print(preview_text)
        logging.info("Done!")
        
    except RuntimeError as e:
        logging.error(str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    
    print(">>> SCRIPT FINISHED <<<")
    sys.exit(0)

if __name__ == "__main__":
    main()
