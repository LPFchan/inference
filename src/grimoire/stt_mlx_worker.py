"""One persistent, serialized MLX Nemotron process, using the native worker protocol."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time

MODEL_NAME = "nemotron-3.5-transcribe-30b-a4b-4bit"
LANGUAGES = {
    "cs": "Czech", "nl": "Dutch", "en": "English", "fr": "French", "de": "German",
    "hi": "Hindi", "it": "Italian", "ko": "Korean", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sk": "Slovak", "es": "Spanish", "bg": "Bulgarian", "hr": "Croatian",
    "da": "Danish", "et": "Estonian", "fi": "Finnish", "hu": "Hungarian", "ja": "Japanese",
    "lv": "Latvian", "lt": "Lithuanian", "ro": "Romanian", "sl": "Slovenian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ar": "Arabic", "bn": "Bengali",
    "zh": "Chinese Mandarin", "el": "Greek", "he": "Hebrew", "id": "Indonesian",
    "mt": "Maltese", "mr": "Marathi", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu", "vi": "Vietnamese",
}
MAX_DURATION = 600
MAX_TOKENS = 4096


def prompt_for(language=None):
    if language is not None and language not in LANGUAGES:
        raise ValueError("Unsupported language; use an advertised ISO language code")
    return (f"Audio transcription. Language: {LANGUAGES.get(language, 'as spoken')}, in its standard writing system. "
            "Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization; "
            "numbers in their standard written form. Output: transcript only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads((args.model_dir / "conversion.json").read_text())
    for name, expected in evidence["files"].items():
        with (args.model_dir / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"Model artifact checksum mismatch: {name}")
    # Library warnings and model progress must never corrupt JSON stdout.
    with redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt import load
        from mlx_audio.stt.utils import load_audio
        mx.set_cache_limit(128 * 1024**2)
        model = load(str(args.model_dir), strict=True)
    print(json.dumps({"ready": True, "model": MODEL_NAME,
                      "artifact_sha256": hashlib.sha256((args.model_dir / "conversion.json").read_bytes()).hexdigest()}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            prompt = prompt_for(request.get("language"))
            before = time.perf_counter()
            with redirect_stdout(sys.stderr):
                audio = load_audio(request["path"], sr=16000)
                duration = len(audio) / 16000
                if not 0 < duration <= MAX_DURATION or not np.isfinite(np.asarray(audio)).all():
                    raise ValueError(f"Audio must be finite and between 0 and {MAX_DURATION} seconds")
                mx.reset_peak_memory()
                result = model.generate(audio, prompt=prompt, max_tokens=MAX_TOKENS,
                                        temperature=0, enable_thinking=False, prefill_step_size=256)
                mx.synchronize()
            if result.generation_tokens >= MAX_TOKENS:
                raise ValueError("Transcript exceeded the output limit")
            response = {"text": result.text, "seconds": time.perf_counter() - before,
                        "duration": duration, "language": request.get("language"),
                        "generation_tokens": result.generation_tokens, "prompt_tokens": result.prompt_tokens,
                        "peak_memory_bytes": mx.get_peak_memory()}
            mx.clear_cache()
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
