from pathlib import Path
import librosa
import argparse
import sys


def main():
    parser = argparse.ArgumentParser("Data checker")
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    data_dir = Path(args.path)

    for file in sorted(list(data_dir.rglob("*.wav"))):
        try:
            yy, _ = librosa.load(file, sr=16_000)

            if len(yy) > 16_000:
                print(file)
        except Exception as e:
            print(f"for {file} ERROR: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
