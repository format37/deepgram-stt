import argparse
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from clean_transcript import clean_transcript

load_dotenv(Path(__file__).parent / ".env")

from deepgram import DeepgramClient

# Timeout for Deepgram API requests (seconds).
# Nova-3 processes ~40x real-time, so 5 hours of audio ≈ 7.5 min.
# 600s gives comfortable headroom.
REQUEST_TIMEOUT = 600

# Chunk duration for fallback splitting (seconds).
# 30 minutes balances reliability vs. chunk count.
CHUNK_DURATION = 30 * 60


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


def get_audio_duration(audio_file: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_file)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def split_audio(audio_file: Path, chunk_duration: int, output_dir: Path) -> list[Path]:
    """Split audio file into chunks using ffmpeg. Returns list of chunk paths."""
    duration = get_audio_duration(audio_file)
    if duration == 0:
        raise RuntimeError(f"Cannot determine duration of {audio_file}")

    chunks = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = output_dir / f"chunk_{idx:03d}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-ss", str(start), "-t", str(chunk_duration),
                "-acodec", "copy", str(chunk_path),
            ],
            capture_output=True,
        )
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(chunk_path)
        start += chunk_duration
        idx += 1

    return chunks


def transcribe_single(client, audio_data: bytes, language: str):
    """Transcribe a single audio buffer with timeout."""
    return client.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        language=language,
        diarize=True,
        utterances=True,
        punctuate=True,
        smart_format=True,
        request_options={"timeout_in_seconds": REQUEST_TIMEOUT},
    )


def response_to_dict(response) -> dict:
    """Convert a Deepgram response to a plain dict."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response


def merge_chunk_responses(chunk_responses: list[tuple[float, dict]]) -> dict:
    """Merge multiple chunk responses into a single combined response.

    Each entry is (time_offset, response_dict).
    Offsets all timestamps and concatenates utterances/words.
    """
    merged = {
        "metadata": {"duration": 0, "channels": 1},
        "results": {
            "channels": [{"alternatives": [{"transcript": "", "words": []}]}],
            "utterances": [],
        },
    }

    all_transcript_parts = []

    for offset, resp in chunk_responses:
        results = resp.get("results", {})

        # Merge utterances
        for utt in results.get("utterances", []):
            shifted = dict(utt)
            shifted["start"] = utt.get("start", 0) + offset
            shifted["end"] = utt.get("end", 0) + offset
            # Shift word timestamps inside utterance
            if "words" in shifted:
                shifted["words"] = [
                    {**w, "start": w.get("start", 0) + offset, "end": w.get("end", 0) + offset}
                    for w in shifted["words"]
                ]
            merged["results"]["utterances"].append(shifted)

        # Merge words from channels
        channels = results.get("channels", [])
        if channels:
            alt = channels[0].get("alternatives", [{}])[0] if channels[0].get("alternatives") else {}
            for w in alt.get("words", []):
                shifted_w = dict(w)
                shifted_w["start"] = w.get("start", 0) + offset
                shifted_w["end"] = w.get("end", 0) + offset
                merged["results"]["channels"][0]["alternatives"][0]["words"].append(shifted_w)
            all_transcript_parts.append(alt.get("transcript", ""))

        # Accumulate duration
        meta = resp.get("metadata", {})
        merged["metadata"]["duration"] += meta.get("duration", 0)

    merged["results"]["channels"][0]["alternatives"][0]["transcript"] = " ".join(all_transcript_parts)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Deepgram")
    parser.add_argument("filename", help="Path to the audio file to transcribe")
    parser.add_argument("-l", "--language", default="en", help="Language code (e.g., en, ru, de, fr). Default: en")
    parser.add_argument("--chunk", action="store_true", help="Use chunked processing (split audio into segments)")
    args = parser.parse_args()

    client = DeepgramClient()

    audio_file = Path(args.filename)

    if not audio_file.exists():
        print(f"Error: File not found: {audio_file}")
        return 1

    duration = get_audio_duration(audio_file)
    file_size_mb = audio_file.stat().st_size / (1024 * 1024)
    print(f"Transcribing: {audio_file} ({file_size_mb:.1f} MB, {duration/60:.1f} min)")

    # Decide whether to use chunked processing
    use_chunks = args.chunk

    if not use_chunks:
        # Single-file approach with extended timeout
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        print(f"Sending request to Deepgram (language: {args.language}, timeout: {REQUEST_TIMEOUT}s)...")
        t0 = time.time()
        try:
            response = transcribe_single(client, audio_data, args.language)
        except Exception as e:
            if "timeout" in str(e).lower() or "ReadTimeout" in type(e).__name__:
                print(f"\nTimeout after {time.time() - t0:.0f}s. Retrying with chunked processing...")
                use_chunks = True
            else:
                raise

        if not use_chunks:
            elapsed = time.time() - t0
            print(f"Response received in {elapsed:.1f}s")

    if use_chunks:
        response = transcribe_chunked(client, audio_file, args.language)

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


def transcribe_chunked(client, audio_file: Path, language: str):
    """Split audio into chunks, transcribe each, and merge results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Splitting audio into {CHUNK_DURATION // 60}-minute chunks...")
        chunks = split_audio(audio_file, CHUNK_DURATION, tmp_path)
        print(f"Split into {len(chunks)} chunks")

        chunk_responses = []
        for i, chunk_path in enumerate(chunks):
            chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
            offset = i * CHUNK_DURATION
            print(f"  Chunk {i + 1}/{len(chunks)} ({chunk_size_mb:.1f} MB, offset {format_timestamp(offset)})...")

            with open(chunk_path, "rb") as f:
                chunk_data = f.read()

            t0 = time.time()
            resp = transcribe_single(client, chunk_data, language)
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

            resp_dict = response_to_dict(resp)
            chunk_responses.append((offset, resp_dict))

    merged = merge_chunk_responses(chunk_responses)
    print(f"Merged {len(chunks)} chunks into single transcript")
    return merged


if __name__ == "__main__":
    main()
