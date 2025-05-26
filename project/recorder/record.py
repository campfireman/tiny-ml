import serial
import wave
import sys
from datetime import datetime

# — CONFIGURE THESE —
PORT = "/dev/cu.usbmodem1101"   # ← your USB-CDC port
BAUD = 115200
RECORD_SECONDS = 5               # must match RECORD_SECONDS in your sketch
SAMPLE_RATE = 8000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2                 # bytes (16-bit PCM)

# derived
total_data_bytes = SAMPLE_RATE * RECORD_SECONDS * NUM_CHANNELS * SAMPLE_WIDTH
HEADER_SIZE = 44
TO_READ = HEADER_SIZE + total_data_bytes


def main():
    print(f"Opening {PORT} @ {BAUD}…")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=RECORD_SECONDS + 2)
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}", file=sys.stderr)
        sys.exit(1)

    # clear any old data
    ser.reset_input_buffer()

    # send the 'r' command to start recording
    ser.write(b"r")
    ser.flush()
    print("Sent trigger 'r', waiting for data…")

    # read exactly HEADER_SIZE + PCM bytes
    data = ser.read(TO_READ)
    ser.close()

    if len(data) < TO_READ:
        print(
            f"Warning: expected {TO_READ} bytes but got {len(data)}", file=sys.stderr)

    # write out the WAV (skip the 44-byte header in data)
    with wave.open(f"recordings/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-test.wav", "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data[HEADER_SIZE:])

    print("Saved recorded.wav")


if __name__ == "__main__":
    main()
