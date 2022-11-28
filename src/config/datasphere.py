from dataclasses import dataclass

from . import default
from .default import MelSpectrogramConfig, FastSpeechConfig


@dataclass
class TrainConfig(default.TrainConfig):
    device = "cuda:0"
    
    max_checkpoints = 20

    batch_size = 64
