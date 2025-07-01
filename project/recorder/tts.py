from random import choice

from torch.serialization import add_safe_globals, safe_globals
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from datetime import datetime

LANG = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
    "hi"
]

COMMANDS = [
    "chillaxo",
    "goodnight",
    "sunblast",
    # "zen mode",
]

STORAGE_DIR = "tts"

add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


with safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=False
    )

    speakers = list(tts.synthesizer.tts_model.speaker_manager.name_to_id)

    for lang in ["en"]:
        for cmd in COMMANDS:
            for speaker in speakers:
                speaker_fmt = speaker.lower().replace(" ", "-")
                tts.tts_to_file(
                    text=cmd,
                    speaker=speaker,
                    file_path=f"{STORAGE_DIR}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{cmd}-tts-local-{speaker_fmt}-{lang}.wav",
                    language=lang
                )
