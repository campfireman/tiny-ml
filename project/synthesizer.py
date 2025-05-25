from TTS.api import TTS
from random import choice

# 1. Load the multi-speaker model
tts = TTS(
    model_name="tts_models/en/vctk/vits",
    progress_bar=False,
    gpu=False
)

# 2. List available speaker IDs
print("Available speakers:", tts.speakers)

# 3. Select one speaker (e.g. the first)
speaker_id = choice(tts.speakers)

# 4. Synthesize to WAV, supplying the speaker
tts.tts_to_file(
    text="chillaxo",
    speaker=speaker_id,
    file_path=f"hello_world_{speaker_id}.wav"
)
