import argparse

import os
import subprocess

import sys
sys.path.append(".")

from src import config

parser = argparse.ArgumentParser(description=".")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str,
                    default="default", help="Config filename (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

if "datasphere" in args.config:
    sys.path.append("/home/jupyter/.local/bin")

train_config = config_.TrainConfig()
data_dir = os.path.abspath(train_config.data_dir)

os.chdir("./resources")

subprocess.run(["../bin/download.sh", data_dir])
