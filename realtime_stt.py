#!/usr/bin/env python3
"""Real-time speech-to-text with speaker diarization using Deepgram SDK v5.3.0."""

import argparse
import json
import signal
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ResultsEvent

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 4000  # ~250ms chunks at 16kHz

# Speaker colors (up to 8 speakers)
SPEAKER_COLORS = [
    Fore.CYAN,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.RED,
    Fore.WHITE,
    Fore.LIGHTBLACK_EX,
]


class RealtimeTranscriber:
    """Real-time transcription with speaker diarization."""

    def __init__(self, language: str = "en", save_output: bool = True, debug: bool = False, model: str = "nova-3"):
        self.language = language
        self.save_output = save_output
        self.debug = debug
        self.model = model
        self.audio_queue = queue.Queue()
        self.transcript_lines = []
        self.current_interim = ""
        self.running = False
        self.client = DeepgramClient()
        self.connection = None
        self.start_time = datetime.now()

    def get_speaker_color(self, speaker_id: int) -> str:
        """Get color for a speaker ID."""
        idx = int(speaker_id) % len(SPEAKER_COLORS)
        return SPEAKER_COLORS[idx]

    def format_speaker_text(self, speaker_id: int, text: str, is_interim: bool = False) -> str:
        """Format text with speaker label and color."""
        color = self.get_speaker_color(speaker_id)
        suffix = "..." if is_interim else ""
        return f"{color}[Speaker {int(speaker_id)}]{Style.RESET_ALL} {text}{suffix}"

    def on_message(self, message):
        """Handle incoming messages from Deepgram."""
        try:
            # Only process transcript results
            if not isinstance(message, ListenV1ResultsEvent):
                if self.debug:
                    msg_type = type(message).__name__
                    print(f"\n{Fore.LIGHTBLACK_EX}[DEBUG] Non-result message: {msg_type}{Style.RESET_ALL}")
                return

            channel = message.channel
            if not channel or not channel.alternatives:
                return

            alt = channel.alternatives[0]
            transcript = alt.transcript if hasattr(alt, "transcript") else ""

            if not transcript:
                return

            # Check if this is a final result
            is_final = message.is_final if hasattr(message, "is_final") else False

            # Get speaker from words if available
            # Note: Diarization info is typically only in final results
            words = alt.words if hasattr(alt, "words") else []
            speaker_id = 0

            if self.debug and is_final and words:
                # Debug: show word-level speaker info
                word_speakers = []
                for w in words[:5]:  # First 5 words
                    spk = getattr(w, 'speaker', None)
                    word_speakers.append(f"{getattr(w, 'word', '?')}(s{spk})")
                print(f"\n{Fore.LIGHTBLACK_EX}[DEBUG] Words: {' '.join(word_speakers)}...{Style.RESET_ALL}")

            if words:
                # Try to get the dominant speaker from all words
                speaker_counts = {}
                for w in words:
                    spk = getattr(w, 'speaker', None)
                    if spk is not None:
                        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

                if speaker_counts:
                    # Use the most frequent speaker in this segment
                    speaker_id = max(speaker_counts, key=speaker_counts.get)
                else:
                    speaker_id = 0

            if is_final:
                # Clear the interim line and print final result
                if self.current_interim:
                    sys.stdout.write("\r" + " " * (len(self.current_interim) + 20) + "\r")
                    sys.stdout.flush()

                formatted = self.format_speaker_text(speaker_id, transcript, is_interim=False)
                print(formatted)

                # Store for saving later
                self.transcript_lines.append({
                    "speaker": int(speaker_id),
                    "text": transcript,
                    "timestamp": (datetime.now() - self.start_time).total_seconds(),
                })
                self.current_interim = ""
            else:
                # Show interim result (update in place)
                formatted = self.format_speaker_text(speaker_id, transcript, is_interim=True)
                sys.stdout.write("\r" + " " * (len(self.current_interim) + 20) + "\r")
                sys.stdout.write(formatted)
                sys.stdout.flush()
                self.current_interim = formatted

        except Exception as e:
            print(f"\n{Fore.RED}Error processing message: {e}{Style.RESET_ALL}")

    def on_error(self, error):
        """Handle errors."""
        print(f"\n{Fore.RED}Deepgram error: {error}{Style.RESET_ALL}")

    def on_close(self, _):
        """Handle connection close."""
        self.running = False

    def on_open(self, _):
        """Handle connection open."""
        print(f"{Fore.GREEN}Listening... (Press Ctrl+C to stop){Style.RESET_ALL}\n")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice audio input."""
        if status:
            print(f"{Fore.YELLOW}Audio status: {status}{Style.RESET_ALL}")
        # Convert float32 to int16 PCM
        audio_int16 = (indata * 32767).astype(np.int16)
        self.audio_queue.put(audio_int16.tobytes())

    def send_audio_loop(self, connection):
        """Send audio data to Deepgram in a loop."""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                connection._send(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"\n{Fore.RED}Error sending audio: {e}{Style.RESET_ALL}")
                break

    def save_transcript(self):
        """Save transcript to files."""
        if not self.transcript_lines:
            print(f"{Fore.YELLOW}No transcript to save.{Style.RESET_ALL}")
            return

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Save text format with speaker labels
        text_path = output_dir / f"{timestamp}_realtime.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            for line in self.transcript_lines:
                f.write(f"[Speaker {line['speaker']}] {line['text']}\n")
        print(f"Saved transcript to: {text_path}")

        # Save JSON with full metadata
        json_path = output_dir / f"{timestamp}_realtime.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "language": self.language,
                "transcript": self.transcript_lines,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to: {json_path}")

    def run(self):
        """Main run loop."""
        self.running = True
        self.start_time = datetime.now()

        try:
            # Connect to Deepgram with options as strings
            with self.client.listen.v1.connect(
                model=self.model,
                language=self.language,
                smart_format="true",
                diarize="true",
                interim_results="true",
                utterance_end_ms="1000",
                punctuate="true",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
                channels=str(CHANNELS),
            ) as connection:
                # Register event handlers
                connection.on(EventType.OPEN, self.on_open)
                connection.on(EventType.MESSAGE, self.on_message)
                connection.on(EventType.ERROR, self.on_error)
                connection.on(EventType.CLOSE, self.on_close)

                # Start audio sending thread
                send_thread = threading.Thread(
                    target=self.send_audio_loop,
                    args=(connection,),
                    daemon=True
                )
                send_thread.start()

                # Start audio input stream
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=np.float32,
                    blocksize=BLOCKSIZE,
                    callback=self.audio_callback,
                ):
                    # Start listening for messages (blocking)
                    connection.start_listening()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        if not self.running:
            return

        self.running = False
        print(f"\n{Fore.YELLOW}Stopping...{Style.RESET_ALL}")

        # Save transcript if enabled
        if self.save_output:
            self.save_transcript()


def main():
    # Initialize colorama for Windows support
    init()

    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text with speaker diarization"
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="Language code (e.g., en, ru, de, fr). Default: en"
    )
    parser.add_argument(
        "-m", "--model",
        default="nova-3",
        help="Deepgram model (nova-2, nova-3, etc.). Default: nova-3"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save transcript to file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        help="Audio input device ID (use --list-devices to see available devices)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output for diarization troubleshooting"
    )
    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        print("Available audio input devices:")
        print(sd.query_devices())
        return

    # Set device if specified
    if args.device is not None:
        sd.default.device = args.device

    print(f"{Fore.CYAN}Deepgram Real-time STT with Diarization{Style.RESET_ALL}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Save output: {not args.no_save}")
    print()

    transcriber = RealtimeTranscriber(
        language=args.language,
        save_output=not args.no_save,
        debug=args.debug,
        model=args.model,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        transcriber.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        transcriber.run()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
