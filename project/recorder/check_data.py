from pathlib import Path
import librosa

data_dir = Path("tts")
labels = []
samples = []

for file in sorted(list(data_dir.iterdir())):
    try:
        yy, sr = librosa.load(file, sr=16_000)

        if len(yy) > 16_000:
            print(file)
        else:
            print("good")
    except Exception as e:
        print(f"for {file} ERROR: {e}")
