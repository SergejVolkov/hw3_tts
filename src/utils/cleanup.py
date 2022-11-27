import os
import warnings
from torch.serialization import SourceChangeWarning


def cleanup_waveglow_warnings():
    warnings.filterwarnings("ignore", category=SourceChangeWarning)


def cleanup_waveglow_files():
    for file in os.listdir("."):
        if file.endswith(".patch"):
            os.remove(file)
