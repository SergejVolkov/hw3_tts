import argparse

import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import seaborn as sns
sns.set()

import sys
sys.path.append(".")

from src import config
from src.dataset import get_data_to_buffer, BufferDataset, collate_fn_tensor
from src.model import FastSpeech
from src.loss import FastSpeechLoss
from src.wandb_writer import WanDBWriter
from src.train import train
from src import utils


utils.cleanup_waveglow_warnings()


parser = argparse.ArgumentParser(description="Train FastSpeech 2 TTS model.")
parser.add_argument("config", nargs="?", metavar="CONFIG", type=str,
                    default="default", help="Config filename (default: %(default)s).")
parser.add_argument("-r", "--resume_step", metavar="INT", type=int,
                    default=0, help="Resume training from checkpoint.")
parser.add_argument("-s", "--seed", metavar="INT", type=int,
                    default=42, help="Random seed (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)
resume = (args.resume_step != 0)

mel_config = config_.MelSpectrogramConfig()
model_config = config_.FastSpeechConfig()
train_config = config_.TrainConfig()


utils.set_random_seed(args.seed)

buffer = get_data_to_buffer(train_config)

dataset = BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor(train_config),
    drop_last=True,
    num_workers=0
)


model = FastSpeech(model_config, mel_config)
model = model.to(train_config.device)

fastspeech_loss = FastSpeechLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

if resume:
    checkpoint = torch.load(train_config.checkpoint_path +
                            f"/checkpoint_{args.resume_step}.pth.tar",
                            map_location=train_config.device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

logger = WanDBWriter(train_config, resume)

waveglow_model = utils.get_WaveGlow(train_config.waveglow_path)
waveglow_model = waveglow_model.cuda()

utils.cleanup_waveglow_files()

os.makedirs(train_config.checkpoint_path, exist_ok=True)

train(train_config, training_loader, model, optimizer, scheduler,
      fastspeech_loss, logger, waveglow_model, args.resume_step)
