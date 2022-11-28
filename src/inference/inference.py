from tqdm import tqdm

from .get_data import get_data
from .synthesis import synthesis
from .. import waveglow


def inference(texts_path, model, waveglow_model, text_cleaners,
              device, pname = None, params = None):
    if params is None:
        params = dict()

    data_list = get_data(texts_path, text_cleaners)
    audio_list = []
    mel_list = []
    if pname is None:
        iter = enumerate(data_list)
    else:
        iter = tqdm(enumerate(data_list), desc=pname)
    for i, phn in iter:
        mel, mel_cuda = synthesis(model, phn, device, **params)

        audio_list += [
            waveglow.inference.get_wav(mel_cuda, waveglow_model)
        ]
        mel_list += [mel]

    return audio_list, mel_list
