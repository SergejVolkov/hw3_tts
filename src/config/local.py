from dataclasses import dataclass

from . import default
from .default import MelSpectrogramConfig, FastSpeechConfig


@dataclass
class TrainConfig(default.TrainConfig):
    data_dir = "C:/Users/foma/Downloads/data"

    batch_size = 4
