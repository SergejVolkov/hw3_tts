from dataclasses import dataclass

from . import default
from .default import MelSpectrogramConfig, FastSpeechConfig


@dataclass
class TrainConfig(default.TrainConfig):
    data_dir = "C:/Users/foma/Downloads/data"
    mel_ground_truth = data_dir + "/mels"
    alignment_path = data_dir + "/alignments"
    pitch_path = data_dir + "/pitch_profiles"
    data_path = data_dir + "/train.txt"

    batch_size = 4
