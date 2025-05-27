from torch.serialization import safe_globals
from torch.serialization import add_safe_globals

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from random import choice
from TTS.api import TTS
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


with safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=False
    )

    tts.tts_to_file(
        text="chillaxo",
        speaker="Ana Florence",
        file_path=f"multi_test.wav",
        language="de"
    )
