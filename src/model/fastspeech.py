import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .length_regulator import LengthRegulator
from .pitch_regulator import PitchRegulator
from .energy_regulator import EnergyRegulator
from .utils import get_mask_from_lengths


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_regulator = PitchRegulator(model_config)
        self.energy_regulator = EnergyRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos = None, mel_max_length = None,
                length_target = None, pitch_target = None, energy_target = None,
                alpha = 1.0, beta = 1.0, gamma = 1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
            output, pitch_predictor_output = self.pitch_regulator(output, beta, pitch_target)
            output, energy_predictor_output = self.energy_regulator(output, gamma, energy_target)
            output = self.decoder(output, mel_pos)

            output = self.mask_tensor(output, mel_pos, mel_max_length)

            output = self.mel_linear(output)

            return output, duration_predictor_output, pitch_predictor_output, energy_predictor_output
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            output = self.pitch_regulator(output, beta)
            output = self.energy_regulator(output, gamma)
            output = self.decoder(output, mel_pos)

            output = self.mel_linear(output)

            return output
