import argparse

import os
import subprocess

import sys
sys.path.append(".")

from src import config

parser = argparse.ArgumentParser(description=".")
parser.add_argument("config", nargs="?", metavar="CONFIG", type=str,
                    default="default", help="Config filename (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

train_config = config_.TrainConfig()

os.chdir("./resources")

subprocess.run(["../bin/download.sh", train_config.data_dir])
