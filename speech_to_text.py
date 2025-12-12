import argparse
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from clean_transcript import clean_transcript

load_dotenv(Path(__file__).parent / ".env")

from deepgram import DeepgramClient


def json_serializer(obj):
    """Handle datetime and other non-serializable objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to YouTube timestamp format (HH:MM:SS or MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def save_results(response, output_base: str = "transcript"):
    """Save transcription results in multiple formats."""

    # 1. Save raw JSON
    json_path = Path(output_base + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Convert response to dict if it has a model_dump method (Pydantic)
        if hasattr(response, "model_dump"):
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False, default=json_serializer)
        elif hasattr(response, "dict"):
            json.dump(response.dict(), f, indent=2, ensure_ascii=False, default=json_serializer)
        else:
            json.dump(response, f, indent=2, ensure_ascii=False, default=json_serializer)
    print(f"Saved raw JSON to: {json_path}")

    # Extract results
    results = response.results if hasattr(response, "results") else response.get("results", {})

    # 2. Save speaker-labeled text format
    text_lines = []
    youtube_lines = []

    # Try to get utterances first (best for speaker diarization)
    utterances = results.utterances if hasattr(results, "utterances") else results.get("utterances", [])

    if utterances:
        for utt in utterances:
            speaker = utt.speaker if hasattr(utt, "speaker") else utt.get("speaker", 0)
            transcript = utt.transcript if hasattr(utt, "transcript") else utt.get("transcript", "")
            start = utt.start if hasattr(utt, "start") else utt.get("start", 0)

            # Speaker-labeled format
            text_lines.append(f"[Speaker {speaker}] - {transcript}")

            # YouTube timestamp format
            timestamp = format_timestamp(start)
            youtube_lines.append(f"{timestamp} [Speaker {speaker}] {transcript}")
    else:
        # Fallback: use channels/alternatives with word-level diarization
        channels = results.channels if hasattr(results, "channels") else results.get("channels", [])

        if channels:
            channel = channels[0]
            alternatives = channel.alternatives if hasattr(channel, "alternatives") else channel.get("alternatives", [])

            if alternatives:
                alt = alternatives[0]
                words = alt.words if hasattr(alt, "words") else alt.get("words", [])

                # Group consecutive words by speaker
                current_speaker = None
                current_text = []
                current_start = 0

                for word in words:
                    speaker = word.speaker if hasattr(word, "speaker") else word.get("speaker", 0)
                    text = word.punctuated_word if hasattr(word, "punctuated_word") else word.get("punctuated_word", word.get("word", ""))
                    start = word.start if hasattr(word, "start") else word.get("start", 0)

                    if speaker != current_speaker:
                        if current_text:
                            text_lines.append(f"[Speaker {current_speaker}] - {' '.join(current_text)}")
                            timestamp = format_timestamp(current_start)
                            youtube_lines.append(f"{timestamp} [Speaker {current_speaker}] {' '.join(current_text)}")

                        current_speaker = speaker
                        current_text = [text]
                        current_start = start
                    else:
                        current_text.append(text)

                # Don't forget the last segment
                if current_text:
                    text_lines.append(f"[Speaker {current_speaker}] - {' '.join(current_text)}")
                    timestamp = format_timestamp(current_start)
                    youtube_lines.append(f"{timestamp} [Speaker {current_speaker}] {' '.join(current_text)}")

    # Save speaker-labeled text
    text_path = Path(output_base + "_speakers.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    print(f"Saved speaker text to: {text_path}")

    # Save YouTube timestamps
    youtube_path = Path(output_base + "_youtube.txt")
    with open(youtube_path, "w", encoding="utf-8") as f:
        f.write("\n".join(youtube_lines))
    print(f"Saved YouTube timestamps to: {youtube_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Deepgram")
    parser.add_argument("filename", help="Path to the audio file to transcribe")
    parser.add_argument("-l", "--language", default="en", help="Language code (e.g., en, ru, de, fr). Default: en")
    args = parser.parse_args()

    client = DeepgramClient()

    audio_file = Path(args.filename)

    if not audio_file.exists():
        print(f"Error: File not found: {audio_file}")
        return 1

    print(f"Transcribing: {audio_file}")

    with open(audio_file, "rb") as f:
        audio_data = f.read()

    print(f"Sending request to Deepgram (language: {args.language})...")
    response = client.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        language=args.language,  # Language code
        diarize=True,      # Enable speaker diarization
        utterances=True,   # Get utterance-level segments
        punctuate=True,    # Add punctuation
        smart_format=True, # Smart formatting
    )
    print("Response received!")

    # Save results in output/{filename_stem}/ folder
    output_dir = Path(__file__).parent / "output" / audio_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = str(output_dir / audio_file.stem)
    save_results(response, output_base)

    # Generate clean transcript
    speakers_file = output_dir / f"{audio_file.stem}_speakers.txt"
    clean_file = output_dir / f"{audio_file.stem}_clean.txt"
    clean_transcript(speakers_file, clean_file)

    # Print preview
    print("\n--- Preview ---")
    results = response.results if hasattr(response, "results") else response.get("results", {})
    utterances = results.utterances if hasattr(results, "utterances") else results.get("utterances", [])

    for utt in utterances[:5]:  # Show first 5 utterances
        speaker = utt.speaker if hasattr(utt, "speaker") else utt.get("speaker", 0)
        transcript = utt.transcript if hasattr(utt, "transcript") else utt.get("transcript", "")
        print(f"[Speaker {speaker}] - {transcript}")

    if len(utterances) > 5:
        print(f"... and {len(utterances) - 5} more utterances")


if __name__ == "__main__":
    main()
