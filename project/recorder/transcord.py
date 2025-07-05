import argparse
import os
import subprocess
import sys
import wave
from datetime import datetime
from time import sleep
from pathlib import Path

import serial

# — CONFIGURE THESE —
BAUD = 115200
RECORD_SECONDS = 1               # must match RECORD_SECONDS in your sketch
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2                 # bytes (16-bit PCM)

# derived
total_data_bytes = SAMPLE_RATE * RECORD_SECONDS * NUM_CHANNELS * SAMPLE_WIDTH
HEADER_SIZE = 44
TO_READ = HEADER_SIZE + total_data_bytes


def count_down(delay_in_s):
    for i in range(delay_in_s, 0, -1):
        print(f"{i}...")
        sleep(1)


def main():
    parser = argparse.ArgumentParser("Recorder with Playback")
    parser.add_argument("port", type=str,
                        help="Serial port (e.g. /dev/tty.usbserial-XYZ)")
    parser.add_argument("extra_label", type=str,
                        help="Will be added to the filename")
    parser.add_argument("playfile", type=str,
                        help="Path to the .wav file to play during recording")
    args = parser.parse_args()

    print(f"Opening {args.port} @ {BAUD}…")
    try:
        ser = serial.Serial(args.port, BAUD, timeout=RECORD_SECONDS + 2)
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}", file=sys.stderr)
        sys.exit(1)

    ser.reset_input_buffer()

    player = subprocess.Popen(["afplay", args.playfile])
    sleep(0.2)
    # trigger the microcontroller
    ser.write(b"r")
    ser.flush()

    # play the WAV file (macOS builtin)
    # subprocess.run(["afplay", args.playfile], check=True)

    # now read back the recorded data
    print("Waiting for recorded data from device…")
    data = ser.read(TO_READ)
    ser.close()

    player.wait()

    if len(data) < TO_READ:
        print(f"Warning: expected {TO_READ} bytes but got {len(data)}",
              file=sys.stderr)

    # write out the WAV (skip the 44-byte header)
    orig_path = Path(args.playfile)
    path = f"recordings/{orig_path.stem}-{args.extra_label}{orig_path.suffix}"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data[HEADER_SIZE:])

    print(f"Saved {path}")


if __name__ == "__main__":
    main()
