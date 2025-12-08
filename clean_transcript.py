import argparse
import re
from pathlib import Path


def clean_transcript(input_file: Path, output_file: Path):
    """Remove speaker labels and concatenate same-speaker lines."""

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Pattern to match "[Speaker X.X] - " or "[Speaker X] - "
    pattern = r"^\[Speaker ([\d.]+)\] - (.*)$"

    result = []
    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()

            if speaker != current_speaker:
                # Save previous speaker's text
                if current_text:
                    result.append("- " + " ".join(current_text))

                # Start new speaker
                current_speaker = speaker
                current_text = [text] if text else []
            else:
                # Same speaker, concatenate
                if text:
                    current_text.append(text)
        else:
            # Line doesn't match pattern, add as-is
            if line:
                current_text.append(line)

    # Don't forget the last speaker
    if current_text:
        result.append("- " + " ".join(current_text))

    # Save result
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result))

    print(f"Saved cleaned transcript to: {output_file}")
    print(f"Total exchanges: {len(result)}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean transcript by removing speaker labels and merging consecutive same-speaker lines"
    )
    parser.add_argument(
        "folder",
        help="Folder name in output/ directory (e.g., 'audio' for output/audio/)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = base_dir / "output" / args.folder

    if not output_dir.exists():
        print(f"Error: Folder not found: {output_dir}")
        return 1

    input_file = output_dir / f"{args.folder}_speakers.txt"
    output_file = output_dir / f"{args.folder}_clean.txt"

    clean_transcript(input_file, output_file)


if __name__ == "__main__":
    main()
