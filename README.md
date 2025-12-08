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
