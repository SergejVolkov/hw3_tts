from .get_data import get_data
from .synthesis import synthesis
from .. import waveglow


def inference(texts_path, model, waveglow_model,
              text_cleaners, device, params=None):
    if params is None:
        params = dict()

    data_list = get_data(texts_path, text_cleaners)
    audio_list = []
    mel_list = []
    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, device, **params)

        audio_list += [
            waveglow.inference.get_wav(mel_cuda, waveglow_model)
        ]
        mel_list += [mel]

    return audio_list, mel_list
