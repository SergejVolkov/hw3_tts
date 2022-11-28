import argparse

import os
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

import sys
sys.path.append(".")

from src import config
from src.model import FastSpeech
from src.inference import inference
from src import utils


utils.cleanup_waveglow_warnings()


parser = argparse.ArgumentParser(description="Generate audio examples.")
parser.add_argument("checkpoint", metavar="CHECKPOINT", type=str,
                    help="Checkpoint name.")
parser.add_argument("texts_path", nargs="?", metavar="TEXT_PATH", type=str,
                    help="Path to text file (default: auto).")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str,
                    default="default", help="Config filename (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

mel_config = config_.MelSpectrogramConfig()
model_config = config_.FastSpeechConfig()
train_config = config_.TrainConfig()

texts_path = train_config.tests_path \
    if args.texts_path is None else args.texts_path


model = FastSpeech(model_config, mel_config)
model = model.to(train_config.device)

checkpoint = torch.load(train_config.checkpoint_path +
                        f"/{args.checkpoint}.pth.tar",
                        map_location=train_config.device)
model.load_state_dict(checkpoint["model"])
model = model.eval()

waveglow_model = utils.get_WaveGlow(train_config.waveglow_path)
waveglow_model = waveglow_model.cuda()

utils.cleanup_waveglow_files()

os.makedirs("results/" + args.checkpoint, exist_ok=True)


def inference2file(pname = None, **params):
    audios = inference(texts_path, model, waveglow_model,
                       train_config.text_cleaners,
                       train_config.device, pname, params)[0]
    for i, audio in enumerate(audios):
        audio *= 32767 / torch.abs(audio).max()  # int16 max
        filename = f"./results/{args.checkpoint}/sample{i}"
        for param, value in params.items():
            filename += f"_{param[0]}{value}"
        filename += ".wav"
        write(filename, rate=22050, data=audio.numpy().astype('int16'))


inference2file(pname="common")

for param in ["alpha", "beta", "gamma"]:
    for value in tqdm([0.8, 1.2], desc=f"{param}"):
        inference2file(**{param: value})

for value in tqdm([0.8, 1.2], desc="all"):
    inference2file(alpha=value, beta=value, gamma=value)
