# Local AI Subtitle Generation Workflow

A complete pipeline for generating English subtitles from video content using local AI models with GPU acceleration. Zero API costs, fully offline capable.

## Two Workflow Modes

| Mode | When to Use | Tools Needed |
|------|-------------|--------------|
| **Transcription Only** | English source, or you just want same-language subs | faster-whisper |
| **Transcription + Translation** | Foreign language source (Spanish, etc.) | faster-whisper + NLLB |

**NLLB is optional** - only install/use it when you need translation from a foreign language to English.

## Overview

- **faster-whisper** (large-v3) - Speech-to-text transcription (works for any language)
- **NLLB-200-3.3B** - Neural machine translation (only for foreign → English)
- **CTranslate2** - Optimized inference runtime
- **mkvmerge** - Subtitle muxing into MKV containers

**Hardware requirements**: NVIDIA GPU with 6GB+ VRAM (10GB+ if using NLLB)
**Processing speed**: ~1-2 minutes per 45-minute episode

## Prerequisites

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt install ffmpeg mkvtoolnix

# Check GPU
nvidia-smi
```

### Python Environment

**Transcription only (English source):**
```bash
pip install faster-whisper pysrt tqdm
```

**With translation (foreign source):**
```bash
pip install faster-whisper ctranslate2 transformers pysrt tqdm
```

### cuDNN Library Path (WSL2/Linux)
If you get cuDNN errors, set the library path before running:
```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
```

## One-Time Setup: NLLB Translation Model (Optional)

**Only needed for foreign language → English translation.**

Convert NLLB-200-3.3B to CTranslate2 format (downloads ~13GB, converts to ~3.2GB):

```bash
mkdir -p models/nllb-ct2

ct2-transformers-converter \
    --model facebook/nllb-200-3.3B \
    --output_dir ./models/nllb-ct2 \
    --quantization int8_float16 \
    --force
```

---

# Mode 1: Transcription Only (English Source)

Use this when the audio is already in English and you just need subtitles.

### Step 1: Extract Audio
```bash
ffmpeg -i "input_video.mkv" -vn -ac 1 -ar 16000 -acodec pcm_s16le "audio.wav" -y
```

### Step 2: Transcribe with faster-whisper
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "audio.wav",
    language="en",           # English source
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    condition_on_previous_text=False
)

segment_list = list(segments)
```

### Step 3: Save as SRT
```python
import pysrt
from datetime import timedelta

def segments_to_srt(segments, output_path):
    subs = pysrt.SubRipFile()
    for i, segment in enumerate(segments, 1):
        start = timedelta(seconds=segment.start)
        end = timedelta(seconds=segment.end)
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
            text=segment.text.strip()
        ))
    subs.save(output_path, encoding='utf-8')

segments_to_srt(segment_list, "english.srt")
```

### Step 4: Mux into MKV
```bash
mkvmerge -o "output.mkv" "input_video.mkv" \
    --language 0:eng --track-name "0:English" "english.srt"
```

**Done!** No translation needed.

---

# Mode 2: Transcription + Translation (Foreign Source)

Use this when the audio is in Spanish, Portuguese, Japanese, etc. and you want English subs.

### Steps 1-3: Same as above
But use the source language code:
```python
segments, info = model.transcribe(
    "audio.wav",
    language="es",           # Spanish source (or pt, ja, ko, etc.)
    ...
)
```

Save as `source.srt` (e.g., `spanish.srt`).

### Step 4: Translate to English with NLLB
```python
import ctranslate2
import transformers

# Load translator
translator = ctranslate2.Translator(
    "./models/nllb-ct2",
    device="cuda",
    compute_type="int8_float16"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "facebook/nllb-200-3.3B",
    src_lang="spa_Latn"  # Source language code (see table below)
)

# Load source subtitles
subs = pysrt.open("spanish.srt", encoding='utf-8')
texts = [sub.text for sub in subs]

# Batch translate
translated = []
batch_size = 64

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
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

# Save English subtitles (preserves original timing)
for sub, trans in zip(subs, translated):
    sub.text = trans

subs.save("english.srt", encoding='utf-8')
```

### Step 5: Mux into MKV
```bash
# English only
mkvmerge -o "output.mkv" "input_video.mkv" \
    --language 0:eng --track-name "0:English" "english.srt"

# Or both languages
mkvmerge -o "output.mkv" "input_video.mkv" \
    --language 0:spa --track-name "0:Spanish" "spanish.srt" \
    --language 0:eng --track-name "0:English" "english.srt"
```

---

## Language Codes Reference

### Whisper Language Codes
For `model.transcribe(language=...)`:

| Language | Code |
|----------|------|
| English | `en` |
| Spanish | `es` |
| Portuguese | `pt` |
| French | `fr` |
| German | `de` |
| Japanese | `ja` |
| Korean | `ko` |
| Chinese | `zh` |
| Russian | `ru` |
| Arabic | `ar` |

Or use `language=None` for auto-detection.

### NLLB Language Codes
For `src_lang` and `target_prefix`:

| Language | Code |
|----------|------|
| English | `eng_Latn` |
| Spanish | `spa_Latn` |
| Portuguese | `por_Latn` |
| French | `fra_Latn` |
| German | `deu_Latn` |
| Italian | `ita_Latn` |
| Japanese | `jpn_Jpan` |
| Korean | `kor_Hang` |
| Chinese (Simplified) | `zho_Hans` |
| Chinese (Traditional) | `zho_Hant` |
| Russian | `rus_Cyrl` |
| Arabic | `arb_Arab` |

---

## Complete Script

Save as `generate_subtitles.py`:

```python
#!/usr/bin/env python3
"""
Generate English subtitles from video files.

Usage:
  English source (transcription only):
    python generate_subtitles.py -i video.mkv -s en

  Foreign source (transcription + translation):
    python generate_subtitles.py -i video.mkv -s es --translate
"""

import argparse
import subprocess
from pathlib import Path
from datetime import timedelta
import pysrt
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(audio_path), "-y"
    ], capture_output=True, check=True)

def transcribe(audio_path, language):
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        condition_on_previous_text=False
    )
    return list(segments)

def segments_to_srt(segments, output_path):
    subs = pysrt.SubRipFile()
    for i, seg in enumerate(segments, 1):
        start = timedelta(seconds=seg.start)
        end = timedelta(seconds=seg.end)
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
    import ctranslate2
    import transformers

    translator = ctranslate2.Translator(model_dir, device="cuda", compute_type="int8_float16")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=src_lang)

    subs = pysrt.open(str(srt_path), encoding='utf-8')
    texts = [sub.text for sub in subs]
    translated = []

    for i in tqdm(range(0, len(texts), 64), desc="Translating"):
        batch = texts[i:i+64]
        tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in batch]
        results = translator.translate_batch(tokenized, target_prefix=[["eng_Latn"]] * len(batch), beam_size=4)
        translated.extend([
            tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True)
            for r in results
        ])

    for sub, trans in zip(subs, translated):
        sub.text = trans
    subs.save(str(output_path), encoding='utf-8')

# NLLB language code mapping
NLLB_CODES = {
    "es": "spa_Latn", "pt": "por_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "ru": "rus_Cyrl", "ar": "arb_Arab"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output video file")
    parser.add_argument("--source-lang", "-s", default="en", help="Source language (default: en)")
    parser.add_argument("--translate", "-t", action="store_true", help="Translate to English (required for non-English source)")
    parser.add_argument("--model-dir", default="./models/nllb-ct2", help="NLLB model directory")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_subtitled")

    work_dir = input_path.parent
    audio_path = work_dir / f"{input_path.stem}.wav"
    source_srt = work_dir / f"{input_path.stem}_source.srt"
    english_srt = work_dir / f"{input_path.stem}_english.srt"

    # Step 1: Extract audio
    print("Extracting audio...")
    extract_audio(input_path, audio_path)

    # Step 2: Transcribe
    print(f"Transcribing ({args.source_lang})...")
    segments = transcribe(audio_path, args.source_lang)
    print(f"  Found {len(segments)} segments")

    # Step 3: Save SRT
    if args.translate and args.source_lang != "en":
        # Save source language SRT, then translate
        segments_to_srt(segments, source_srt)

        # Step 4: Translate
        print("Translating to English...")
        nllb_code = NLLB_CODES.get(args.source_lang, "spa_Latn")
        translate_srt(source_srt, english_srt, nllb_code, args.model_dir)
        final_srt = english_srt
    else:
        # No translation needed - save directly as English
        segments_to_srt(segments, english_srt)
        final_srt = english_srt

    # Step 5: Mux
    print("Muxing subtitles...")
    subprocess.run([
        "mkvmerge", "-o", str(output_path), str(input_path),
        "--language", "0:eng", "--track-name", "0:English", str(final_srt)
    ], capture_output=True, check=True)

    # Cleanup
    if not args.keep_temp:
        audio_path.unlink(missing_ok=True)
        source_srt.unlink(missing_ok=True)
        english_srt.unlink(missing_ok=True)

    print(f"Done: {output_path}")

if __name__ == "__main__":
    main()
```

### Usage Examples

```bash
# English source - transcription only (no NLLB needed)
python generate_subtitles.py -i movie.mkv -s en

# Spanish source - transcribe + translate
python generate_subtitles.py -i pelicula.mkv -s es --translate

# Japanese source - transcribe + translate
python generate_subtitles.py -i anime.mkv -s ja --translate

# Batch process folder (English)
for f in *.mkv; do python generate_subtitles.py -i "$f" -s en; done

# Batch process folder (Spanish with translation)
for f in *.mkv; do python generate_subtitles.py -i "$f" -s es -t; done
```

---

## Firestick Capture Workflow Integration

For videos captured via Firestick -> Capture Card -> OBS:

1. **OBS Recording Settings**:
   - Container: MKV (for crash recovery)
   - Video: x264 or NVENC
   - Audio: AAC 192kbps stereo

2. **Post-capture remux** (if needed):
   ```bash
   ffmpeg -i recording.mkv -c copy clean.mkv
   ```

3. **Generate subtitles**:
   ```bash
   # English content
   python generate_subtitles.py -i clean.mkv -s en

   # Foreign content
   python generate_subtitles.py -i clean.mkv -s es --translate
   ```

---

## Resource Usage

| Mode | VRAM | Disk |
|------|------|------|
| Transcription only | ~4.5 GB | ~3 GB |
| + Translation | ~10 GB | ~6.2 GB |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| cuDNN not found | Set LD_LIBRARY_PATH (see Prerequisites) |
| Out of VRAM | Use `compute_type="int8"` |
| Hallucinations during silence | Enable `vad_filter=True` |
| Repetitive text | Set `condition_on_previous_text=False` |
| Wrong language detected | Always specify `language="xx"` explicitly |
# Subtitle-Engine
