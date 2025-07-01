"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech
from datetime import datetime

COMMANDS = [
    # "chillaxo",
    "goodnight",
    "sunblast",
    "zenmode",
]

STORAGE_DIR = "tts"

# Instantiates a client
client = texttospeech.TextToSpeechClient()


def list_voices(client, lang):
    resp = client.list_voices()
    return map(lambda v: v.name, filter(
        lambda voice: lang in voice.language_codes, resp.voices))


for lang in ["de-DE"]:
    for cmd in COMMANDS:
        for speaker in list_voices(client, lang):
            speaker_fmt = speaker.lower().replace(" ", "-")
            voice = texttospeech.VoiceSelectionParams(
                language_code=lang,
                name=speaker,
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            synthesis_input = texttospeech.SynthesisInput(text=cmd)

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            with open(f"{STORAGE_DIR}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{cmd}-tts-gcp-{speaker_fmt}-{lang}.wav", "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)
