#!/usr/bin/env python3
"""
Generate English subtitles from video files using local AI models.

Usage:
  English source (transcription only):
    python generate_subtitles.py -i video.mkv

  Foreign source (transcription + translation):
    python generate_subtitles.py -i video.mkv -s es --translate

Requirements:
  pip install faster-whisper pysrt tqdm
  pip install ctranslate2 transformers  # only if using --translate

For translation, first run the NLLB model setup:
  ct2-transformers-converter --model facebook/nllb-200-3.3B \
    --output_dir ./models/nllb-ct2 --quantization int8_float16
"""

# Fix cuDNN library path - preload libraries before CUDA imports
import os
import ctypes
from pathlib import Path as _Path

def _preload_cuda_libs():
    """Preload cuDNN/cuBLAS from pip-installed nvidia packages."""
    # Find site-packages (check venv first, then user)
    for base in [
        _Path(__file__).parent / ".venv" / "lib",
        _Path.home() / ".local" / "lib",
    ]:
        for pyver in ["python3.12", "python3.11", "python3.10"]:
            sp = base / pyver / "site-packages" / "nvidia"
            if sp.exists():
                cudnn_lib = sp / "cudnn" / "lib" / "libcudnn.so.9"
                cublas_lib = sp / "cublas" / "lib" / "libcublas.so.12"
                if cudnn_lib.exists():
                    try:
                        ctypes.CDLL(str(cublas_lib), mode=ctypes.RTLD_GLOBAL)
                        ctypes.CDLL(str(cudnn_lib), mode=ctypes.RTLD_GLOBAL)
                        return True
                    except OSError:
                        pass
    return False

_preload_cuda_libs()

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import timedelta

try:
    import pysrt
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install pysrt tqdm")
    sys.exit(1)


def extract_audio(video_path, audio_path):
    """Extract 16kHz mono WAV for Whisper."""
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(audio_path), "-y"
    ], capture_output=True, check=True)


def transcribe(audio_path, language, quality="default"):
    """Transcribe audio using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Missing faster-whisper. Run: pip install faster-whisper")
        sys.exit(1)

    print(f"  Loading Whisper large-v3 model...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    # Quality presets
    if quality == "high":
        # High quality: no VAD filter (catches all dialogue), higher beam
        print(f"  Transcribing ({language}) [HIGH QUALITY MODE]...")
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=10,
            best_of=5,
            vad_filter=False,  # Don't filter - catch everything
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8),  # Retry with higher temp if low confidence
        )
    elif quality == "sensitive":
        # Sensitive: VAD with relaxed settings for quiet dialogue
        print(f"  Transcribing ({language}) [SENSITIVE MODE]...")
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=8,
            best_of=3,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,  # Shorter silence threshold
                speech_pad_ms=400,            # More padding around speech
                threshold=0.3,                # Lower threshold = more sensitive
            ),
            condition_on_previous_text=False,
            word_timestamps=True,
        )
    else:
        # Default: balanced speed/quality
        print(f"  Transcribing ({language})...")
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            condition_on_previous_text=False,
            word_timestamps=True,
        )

    return list(segments)


def segments_to_srt(segments, output_path):
    """Convert Whisper segments to SRT file with precise end times."""
    subs = pysrt.SubRipFile()
    for i, seg in enumerate(segments, 1):
        start_sec = seg.start

        # Use last word's end time for precise subtitle duration
        # Falls back to segment end if no word timestamps
        if hasattr(seg, 'words') and seg.words:
            end_sec = seg.words[-1].end
        else:
            end_sec = seg.end

        # Add small buffer (0.1s) so subtitle doesn't cut off too abruptly
        end_sec += 0.1

        start = timedelta(seconds=start_sec)
        end = timedelta(seconds=end_sec)

        subs.append(pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(
                hours=int(start.total_seconds() // 3600),
                minutes=int((start.total_seconds() % 3600) // 60),
                seconds=int(start.total_seconds() % 60),
                milliseconds=int((start.total_seconds() % 1) * 1000)
            ),
            end=pysrt.SubRipTime(
                hours=int(end.total_seconds() // 3600),
                minutes=int((end.total_seconds() % 3600) // 60),
                seconds=int(end.total_seconds() % 60),
                milliseconds=int((end.total_seconds() % 1) * 1000)
            ),
            text=seg.text.strip()
        ))
    subs.save(str(output_path), encoding='utf-8')
    return subs


def translate_srt(srt_path, output_path, src_lang, model_dir):
    """Translate SRT file to English using NLLB-200."""
    try:
        import ctranslate2
        import transformers
    except ImportError:
        print("Missing translation dependencies. Run: pip install ctranslate2 transformers")
        sys.exit(1)

    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"NLLB model not found at {model_dir}")
        print("Run the following to set it up:")
        print(f"  ct2-transformers-converter --model facebook/nllb-200-3.3B \\")
        print(f"    --output_dir {model_dir} --quantization int8_float16")
        sys.exit(1)

    print(f"  Loading NLLB translator...")
    translator = ctranslate2.Translator(str(model_dir), device="cuda", compute_type="int8_float16")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=src_lang)

    subs = pysrt.open(str(srt_path), encoding='utf-8')
    texts = [sub.text for sub in subs]
    translated = []

    print(f"  Translating {len(texts)} segments...")
    for i in tqdm(range(0, len(texts), 64), desc="  Batches"):
        batch = texts[i:i+64]
        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in batch]
        results = translator.translate_batch(
            tokenized,
            target_prefix=[["eng_Latn"]] * len(batch),
            beam_size=4
        )
        translated.extend([
            tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True)
            for r in results
        ])

    for sub, trans in zip(subs, translated):
        sub.text = trans
    subs.save(str(output_path), encoding='utf-8')


# NLLB language code mapping (Whisper code -> NLLB code)
NLLB_CODES = {
    "es": "spa_Latn",
    "pt": "por_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
}


def process_video(input_path, output_path, args):
    """Process a single video file."""
    import shutil

    work_dir = input_path.parent
    audio_path = work_dir / f"{input_path.stem}_temp.wav"
    source_srt = work_dir / f"{input_path.stem}_source.srt"
    english_srt = work_dir / f"{input_path.stem}.srt"

    print(f"\nProcessing: {input_path.name}")

    # Step 1: Extract audio
    print("\n[1/4] Extracting audio...")
    extract_audio(input_path, audio_path)

    # Step 2: Transcribe
    print("\n[2/4] Transcribing...")
    segments = transcribe(audio_path, args.source_lang, args.quality)
    print(f"  Found {len(segments)} segments")

    # Step 3: Save/Translate
    if args.translate and args.source_lang != "en":
        print("\n[3/4] Saving source subtitles...")
        segments_to_srt(segments, source_srt)

        print("\n[4/4] Translating to English...")
        nllb_code = NLLB_CODES.get(args.source_lang)
        if not nllb_code:
            print(f"Warning: Unknown language '{args.source_lang}', using Spanish (spa_Latn)")
            nllb_code = "spa_Latn"
        translate_srt(source_srt, english_srt, nllb_code, args.model_dir)
    else:
        print("\n[3/4] Saving subtitles...")
        segments_to_srt(segments, english_srt)
        print("\n[4/4] Skipping translation (English source)")

    # Step 4: Mux (unless --srt-only)
    if not args.srt_only:
        print("\n[5/5] Muxing subtitles into video...")

        # Build mkvmerge command
        mux_cmd = ["mkvmerge", "-o", str(output_path), str(input_path)]

        # Add English subtitles (always default)
        mux_cmd.extend([
            "--language", "0:eng",
            "--track-name", "0:English",
            "--default-track", "0:yes",
            str(english_srt)
        ])

        # Add source language subtitles if requested (for translation mode)
        if args.keep_source_subs and args.translate and args.source_lang != "en" and source_srt.exists():
            # Language name mapping for track names
            lang_names = {
                "es": "Spanish", "pt": "Portuguese", "fr": "French",
                "de": "German", "it": "Italian", "ja": "Japanese",
                "ko": "Korean", "zh": "Chinese", "ru": "Russian", "ar": "Arabic"
            }
            lang_name = lang_names.get(args.source_lang, args.source_lang.upper())

            mux_cmd.extend([
                "--language", f"0:{args.source_lang}",
                "--track-name", f"0:{lang_name}",
                "--default-track", "0:no",
                str(source_srt)
            ])
            print(f"  Including both English (default) and {lang_name} subtitles")

        subprocess.run(mux_cmd, capture_output=True, check=True)
        print(f"  Created: {output_path}")

    # Cleanup temp files
    if not args.keep_temp:
        audio_path.unlink(missing_ok=True)
        # Keep source SRT if --keep-source-subs is set (for --srt-only mode)
        if args.translate and not args.keep_source_subs:
            source_srt.unlink(missing_ok=True)
        if not args.srt_only:
            english_srt.unlink(missing_ok=True)

    return output_path if not args.srt_only else english_srt


def get_video_files(directory):
    """Get all video files from a directory."""
    video_extensions = {'.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    videos = []
    for ext in video_extensions:
        videos.extend(Path(directory).glob(f"*{ext}"))
        videos.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(
        description="Generate English subtitles from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    %(prog)s -i movie.mkv                    # English source
    %(prog)s -i pelicula.mkv -s es -t        # Spanish -> English

  Batch processing:
    %(prog)s --input-dir ./queue --originals-dir ./originals --processed-dir ./done
    %(prog)s --input-dir ./queue --originals-dir ./originals --processed-dir ./done -s es -t
        """
    )
    # Single file mode
    parser.add_argument("--input", "-i", help="Input video file (single file mode)")
    parser.add_argument("--output", "-o", help="Output video file (default: input_subtitled.mkv)")

    # Batch mode
    parser.add_argument("--input-dir", help="Input directory with videos to process (batch mode)")
    parser.add_argument("--originals-dir", help="Directory to move original files after processing")
    parser.add_argument("--processed-dir", help="Directory to move processed files to")

    # Common options
    parser.add_argument("--source-lang", "-s", default="en",
                        help="Source language code: en, es, pt, fr, de, ja, ko, zh, ru, ar (default: en)")
    parser.add_argument("--translate", "-t", action="store_true",
                        help="Translate to English (required for non-English source)")
    parser.add_argument("--quality", "-q", default="default",
                        choices=["default", "high", "sensitive"],
                        help="Quality preset: default (fast), high (no VAD, catches all), sensitive (relaxed VAD)")
    parser.add_argument("--model-dir", default="./models/nllb-ct2",
                        help="NLLB model directory (default: ./models/nllb-ct2)")
    parser.add_argument("--srt-only", action="store_true",
                        help="Only generate SRT file, don't mux into video")
    parser.add_argument("--keep-source-subs", action="store_true",
                        help="Include source language subtitles alongside English (translation mode only)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep intermediate files (audio, source SRT)")
    args = parser.parse_args()

    # Validate arguments
    if args.input_dir:
        # Batch mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        if not args.originals_dir or not args.processed_dir:
            print("Error: Batch mode requires --originals-dir and --processed-dir")
            sys.exit(1)

        originals_dir = Path(args.originals_dir)
        processed_dir = Path(args.processed_dir)

        # Create output directories if they don't exist
        originals_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Get all video files
        videos = get_video_files(input_dir)
        if not videos:
            print(f"No video files found in {input_dir}")
            sys.exit(0)

        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(videos)} videos")
        print(f"{'='*60}")
        print(f"  Input:      {input_dir}")
        print(f"  Originals:  {originals_dir}")
        print(f"  Processed:  {processed_dir}")
        print(f"  Language:   {args.source_lang}" + (" -> English" if args.translate else ""))
        print(f"  Quality:    {args.quality}")
        print(f"{'='*60}")

        import shutil
        success_count = 0
        fail_count = 0

        for idx, video_path in enumerate(videos, 1):
            print(f"\n[{idx}/{len(videos)}] {video_path.name}")
            print("-" * 40)

            try:
                # Process to a temp location in processed_dir
                output_path = processed_dir / video_path.name

                process_video(video_path, output_path, args)

                # Move original to originals folder
                original_dest = originals_dir / video_path.name
                shutil.move(str(video_path), str(original_dest))
                print(f"  Moved original -> {originals_dir.name}/")

                success_count += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                fail_count += 1
                continue

        print(f"\n{'='*60}")
        print(f"Batch Complete!")
        print(f"  Processed: {success_count}/{len(videos)}")
        if fail_count:
            print(f"  Failed: {fail_count}")
        print(f"{'='*60}")

    elif args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_subtitled")

        result = process_video(input_path, output_path, args)

        print(f"\nDone!")
        if args.srt_only:
            print(f"  SRT: {result}")
        else:
            print(f"  Output: {result}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
