from dataclasses import dataclass

from . import default
from .default import MelSpectrogramConfig, FastSpeechConfig


@dataclass
class TrainConfig(default.TrainConfig):
    data_dir = "./data"
    device = "cuda:0"
    
    max_checkpoints = 30

    batch_size = 64
