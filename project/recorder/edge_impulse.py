#!/usr/bin/env python3
import argparse
import json
import os
import struct
import wave
from array import array
from pathlib import Path
from datetime import datetime


def json_to_wav(json_path):
    # load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    payload = data.get('payload', {})
    values = payload.get('values')
    interval_ms = payload.get('interval_ms')
    if values is None or interval_ms is None:
        raise ValueError(f"Missing 'values' or 'interval_ms' in {json_path}")

    # truncate to max 16 000 samples
    MAX_SAMPLES = 16_000
    if len(values) > MAX_SAMPLES:
        values = values[:MAX_SAMPLES]

    # compute sample rate
    sample_rate = int(1000 / interval_ms)

    # prepare output path
    base = Path(json_path).stem.split(".")
    label = base[0]
    id = base[1]
    if "helloworld" == label:
        print(f"Skipping {json_path}")
        return
    if label == "noise":
        label = "idle"
    wav_path = f"recordings/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{label}-{id}-edge-impulse.wav"

    # write wave file
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        arr = array('h', values)
        wf.writeframes(arr.tobytes())

    print(
        f"Converted {os.path.basename(json_path)} → {os.path.basename(wav_path)} "
        f"({len(values)} samples @ {sample_rate} Hz)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of JSON-encoded sample arrays into 16 kHz WAV files"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing .json files"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        parser.error(f"'{args.folder}' is not a directory")

    for fname in os.listdir(args.folder):
        if not fname.lower().endswith('.json'):
            continue
        fullpath = os.path.join(args.folder, fname)
        try:
            json_to_wav(fullpath)
        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == '__main__':
    main()
