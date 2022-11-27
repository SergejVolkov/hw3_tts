import argparse

import os
import numpy as np

import torch
from torch import nn
from tqdm import tqdm

import sys
sys.path.append(".")

from src import config

parser = argparse.ArgumentParser(description="Generate and store pitch profiles.")
parser.add_argument("config", nargs="?", metavar="CONFIG", type=str,
                    default="default", help="Config filename (default: %(default)s).")
parser.add_argument("-w", "--window_size", metavar="INT", type=int,
                    default=5, help="Sliding window size for smoothing (default: %(default)s).")
parser.add_argument("-p", "--peak_height", metavar="FLOAT", type=float,
                    default=1 / 3,
                    help="Minimum peak height for peak detection (default: %(default)s std).")
args = parser.parse_args()

config_ = getattr(config, args.config)

train_config = config_.TrainConfig()


kernel_size = args.window_size
conv = nn.Conv1d(1, 1, kernel_size, padding="same", padding_mode="replicate", bias=False)
conv.weight.requires_grad = False
nn.init.constant_(conv.weight, 1 / kernel_size)

n_samples = len(os.listdir(train_config.mel_ground_truth))

os.makedirs(train_config.pitch_path, exist_ok=True)

for i in tqdm(range(n_samples), desc="Generating profiles"):
    mel = np.load(train_config.mel_ground_truth +
                  "/ljspeech-mel-%05d.npy" % (i + 1))

    cummax = np.maximum.accumulate(mel, axis=-1)
    idxs = np.argmax((cummax - mel) > mel.std() * args.peak_height, axis=-1)
    maxs = np.take_along_axis(cummax, np.expand_dims(idxs, axis=-1), axis=-1)
    peak = np.argmax(cummax == maxs, axis=-1)
    smoothed = conv(torch.tensor(peak, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()

    np.save(train_config.pitch_path + f"/{i}.npy", smoothed)
