import torch
from torch import nn

from .variance_predictor import VariancePredictor
from ..utils import quantize


class PitchRegulator(nn.Module):
    """ Pitch Regulator """

    def __init__(self, model_config):
        super(PitchRegulator, self).__init__()

        self.pitch_dim = model_config.pitch_dim
        self.pitch_range = model_config.pitch_range

        self.pitch_predictor = VariancePredictor(model_config)

        self.pitch_emb = nn.Embedding(
            model_config.pitch_dim,
            model_config.encoder_dim
        )

    def PR(self, x, pitch_predictor_output):
        output = x + self.pitch_emb(pitch_predictor_output)
        return output

    def forward(self, x, beta = 1.0, target = None):
        pitch_predictor_output = self.pitch_predictor(x)

        if target is not None:
            quantized = quantize(target, self.pitch_range, self.pitch_dim, toint=True)
            output = self.PR(x, quantized)
            return output, pitch_predictor_output
        else:
            quantized = quantize(pitch_predictor_output,
                                 self.pitch_range, self.pitch_dim)
            pitch_predictor_output = torch.clamp(
                (quantized * beta + 0.5).int(), min=0, max=self.pitch_dim - 1)

            output = self.PR(x, pitch_predictor_output)

            return output
