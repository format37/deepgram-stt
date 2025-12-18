# deepgram-stt
Speech to text using Deepgram API with speaker diarization.

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file with your Deepgram API key:
```
DEEPGRAM_API_KEY=your_api_key_here
```

## Usage

### 1. Transcribe audio
```bash
python speech_to_text.py <audio_file> [-l <language>]
```

**Examples:**
```bash
# Transcribe English audio (default)
python speech_to_text.py interview.wav

# Transcribe Russian audio
python speech_to_text.py podcast.mp3 -l ru
```

**Output:** Creates `output/<filename>/` folder with:
- `<filename>.json` - Raw API response
- `<filename>_speakers.txt` - Speaker-labeled transcript
- `<filename>_youtube.txt` - YouTube timestamp format

### 2. Clean transcript
Merge consecutive same-speaker lines into cleaner format:
```bash
python clean_transcript.py <folder>
```

**Example:**
```bash
# Clean transcript for interview.wav (uses output/interview/)
python clean_transcript.py interview
```

**Output:** Creates `output/<folder>/<folder>_clean.txt`

## Language codes
- `en` - English (default)
- `uk` - Ukrainian
- `ru` - Russian
- `de` - German
- `fr` - French
- `es` - Spanish

# Deepgram Real-time STT with Diarization

Real-time speech-to-text transcription with speaker diarization using Deepgram's WebSocket API.

## Features

- Real-time microphone transcription
- Speaker diarization (identifies different speakers)
- Color-coded speaker labels in terminal
- Interim results (see words as you speak)
- Automatic transcript saving

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Deepgram API key:
```
DEEPGRAM_API_KEY=your_api_key_here
```

Or copy from the deepgram-stt folder:
```bash
cp ../deepgram-stt/.env .
```

## Usage

Basic usage (English):
```bash
python realtime_stt.py
```

With different language:
```bash
python realtime_stt.py -l ru    # Russian
python realtime_stt.py -l de    # German
python realtime_stt.py -l fr    # French
```

Without saving transcript:
```bash
python realtime_stt.py --no-save
```

List available audio devices:
```bash
python realtime_stt.py --list-devices
```

Use specific audio device:
```bash
python realtime_stt.py -d 2     # Use device ID 2
```

## Output

Transcripts are saved to the `output/` folder:
- `YYYYMMDD_HHMMSS_realtime.txt` - Speaker-labeled transcript
- `YYYYMMDD_HHMMSS_realtime.json` - Full metadata with timestamps

## Controls

- **Ctrl+C** - Stop recording and save transcript

## Terminal Display

Each speaker is shown with a different color:
- Speaker 0: Cyan
- Speaker 1: Green
- Speaker 2: Yellow
- Speaker 3: Magenta
- etc.

Interim results are shown with `...` suffix and update in place.
Final results are printed on a new line.
