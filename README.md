# Subgen - Local AI Subtitle Generator

Generate English subtitles from video files using local AI models with GPU acceleration. Supports transcription (English source) and translation (foreign language → English).

## Features

- **Transcription**: English audio → English subtitles (faster-whisper large-v3)
- **Translation**: Foreign audio → English subtitles (faster-whisper + NLLB-200)
- **Batch processing**: Process entire folders with automatic file organization
- **Quality presets**: Tune for speed vs accuracy
- **Precise timing**: Subtitles disappear when speech ends (word-level timestamps)

## Requirements

### System Dependencies
```bash
sudo apt install ffmpeg mkvtoolnix
```

### Hardware
- NVIDIA GPU with 6GB+ VRAM (transcription only)
- NVIDIA GPU with 10GB+ VRAM (transcription + translation)
- Tested on RTX 4090

## Installation

### 1. Clone and set up environment
```bash
cd ~/movies/subtitles
uv sync  # or: pip install faster-whisper pysrt tqdm
```

### 2. For translation support (optional)
```bash
uv sync --extra translate  # or: pip install ctranslate2 transformers
```

### 3. Set up NLLB model (only for foreign language translation)

Downloads ~13GB, converts to optimized ~3.2GB model. Takes 10-20 minutes:

```bash
mkdir -p ./models

ct2-transformers-converter --model facebook/nllb-200-3.3B \
  --output_dir ./models/nllb-ct2 --quantization int8_float16
```

## Usage

Always use the `./subgen` wrapper script (handles CUDA library paths automatically).

### Single File Mode

**English source** (transcription only):
```bash
./subgen -i movie.mkv
./subgen -i movie.mkv -q high          # Better quality, slower
./subgen -i movie.mkv -o output.mkv    # Custom output name
./subgen -i movie.mkv --srt-only       # Only generate .srt, don't mux
```

**Foreign language source** (transcription + translation):
```bash
./subgen -i pelicula.mkv -s es -t      # Spanish → English
./subgen -i filme.mkv -s pt -t         # Portuguese → English
./subgen -i film.mkv -s fr -t          # French → English
./subgen -i anime.mkv -s ja -t         # Japanese → English
./subgen -i drama.mkv -s ko -t         # Korean → English
```

**Keep both language tracks** (English default, source language as secondary):
```bash
./subgen -i pelicula.mkv -s es -t --keep-source-subs  # Spanish + English subs
```

### Batch Mode

Process all videos in a folder with automatic file organization:

```bash
./subgen --input-dir ./input \
         --originals-dir ./original \
         --processed-dir ./processed
```

With options:
```bash
# High quality English transcription
./subgen --input-dir ./input \
         --originals-dir ./original \
         --processed-dir ./processed \
         -q high

# Spanish to English translation
./subgen --input-dir ./input \
         --originals-dir ./original \
         --processed-dir ./processed \
         -s es -t -q high
```

**How batch mode works:**
1. Scans `--input-dir` for video files (.mkv, .mp4, .avi, .mov, .webm, etc.)
2. Processes each video with the specified settings
3. Saves subtitled version to `--processed-dir` (same filename)
4. Moves original file to `--originals-dir`
5. Continues to next video (errors don't stop the batch)

## Options Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input video file (single file mode) |
| `--output` | `-o` | Output filename (default: input_subtitled.mkv) |
| `--input-dir` | | Input directory (batch mode) |
| `--originals-dir` | | Where to move originals after processing |
| `--processed-dir` | | Where to save processed videos |
| `--source-lang` | `-s` | Source language code (default: en) |
| `--translate` | `-t` | Enable translation to English |
| `--quality` | `-q` | Quality preset: default, high, sensitive |
| `--model-dir` | | NLLB model location (default: ./models/nllb-ct2) |
| `--srt-only` | | Only generate .srt file, don't mux into video |
| `--keep-source-subs` | | Include source language subtitles alongside English |
| `--keep-temp` | | Keep intermediate files (audio, source SRT) |

## Quality Presets

| Preset | VAD Filter | Beam Size | Speed | Best For |
|--------|------------|-----------|-------|----------|
| `default` | On (aggressive) | 5 | Fast | Clean audio, clear speech |
| `sensitive` | On (relaxed) | 8 | Medium | Quiet dialogue, background noise |
| `high` | Off | 10 | Slow | Missing dialogue, music over speech |

**Use `--quality high` if dialogue is being missed.**

```bash
./subgen -i movie.mkv -q high
```

## Supported Languages

| Language | Code | Translation Required |
|----------|------|---------------------|
| English | `en` | No |
| Spanish | `es` | Yes |
| Portuguese | `pt` | Yes |
| French | `fr` | Yes |
| German | `de` | Yes |
| Italian | `it` | Yes |
| Japanese | `ja` | Yes |
| Korean | `ko` | Yes |
| Chinese | `zh` | Yes |
| Russian | `ru` | Yes |
| Arabic | `ar` | Yes |

## Example Workflows

### Firestick Capture Workflow

For videos captured via Firestick → Capture Card → OBS:

```bash
# 1. Record with OBS (use MKV container)
# 2. Move recordings to input folder
mv ~/Videos/*.mkv ~/movies/subtitles/input/

# 3. Batch process with high quality
./subgen --input-dir ./input \
         --originals-dir ./original \
         --processed-dir ./processed \
         -q high
```

### Foreign Show Processing

```bash
# Spanish telenovela
./subgen --input-dir "./shows/Metastasis" \
         --originals-dir "./shows/Metastasis/originals" \
         --processed-dir "./shows/Metastasis/subtitled" \
         -s es -t -q high

# Anime (Japanese)
./subgen --input-dir "./shows/Anime" \
         --originals-dir "./shows/Anime/raw" \
         --processed-dir "./shows/Anime/subbed" \
         -s ja -t
```

### Using Existing NLLB Model

If you have NLLB set up elsewhere:

```bash
./subgen -i video.mkv -s es -t --model-dir "/path/to/nllb-ct2"
```

## Troubleshooting

### cuDNN not found
Always use the `./subgen` wrapper instead of running `python generate_subtitles.py` directly. The wrapper sets the correct CUDA library paths.

### Missing dialogue
Use `--quality high` - this disables VAD filtering which can cut quiet dialogue.

### Wrong words / hallucinations
- Check audio quality (capture card settings, OBS audio bitrate)
- Try `--quality sensitive` as a middle ground

### NLLB model not found
Set up the model (see Installation) or point to an existing one:
```bash
./subgen -i video.mkv -s es -t --model-dir /path/to/nllb-ct2
```

### Subtitles stay on screen too long
This should be fixed - the script uses word-level timestamps. If still happening, check you have the latest version of the script.

### Out of VRAM
The script uses ~10GB VRAM with both models loaded. Close other GPU applications or use a smaller model.

## Resource Usage

| Mode | VRAM | Disk Space |
|------|------|------------|
| Transcription only | ~4.5 GB | ~3 GB (Whisper model) |
| + Translation | ~10 GB | ~6.2 GB (+ NLLB model) |

## Processing Speed

On RTX 4090:

| Quality | Speed (per 45-min episode) |
|---------|---------------------------|
| `default` | ~1-2 minutes |
| `sensitive` | ~2-3 minutes |
| `high` | ~4-6 minutes |

Translation adds ~30 seconds per episode.

## Technical Details

### Models Used
- **Transcription**: faster-whisper with Whisper large-v3 (float16)
- **Translation**: NLLB-200-3.3B via CTranslate2 (int8_float16 quantization)

### Subtitle Timing
Uses word-level timestamps from Whisper to set precise end times. Subtitles disappear 0.1s after the last word ends, not when the next segment starts.

### VAD (Voice Activity Detection)
- `default`: Aggressive filtering, may miss quiet dialogue
- `sensitive`: Relaxed thresholds, catches more but may include some noise
- `high`: Disabled entirely, processes all audio
