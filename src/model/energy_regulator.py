import torch
from torch import nn

from .variance_predictor import VariancePredictor
from ..utils import quantize


class EnergyRegulator(nn.Module):
    """ Energy Regulator """

    def __init__(self, model_config):
        super(EnergyRegulator, self).__init__()

        self.energy_dim = model_config.energy_dim
        self.energy_range = model_config.energy_range

        self.energy_predictor = VariancePredictor(model_config)

        self.energy_emb = nn.Embedding(
            model_config.energy_dim,
            model_config.encoder_dim
        )

    def ER(self, x, energy_predictor_output):
        output = x + self.energy_emb(energy_predictor_output)
        return output

    def forward(self, x, gamma = 1.0, target = None):
        energy_predictor_output = self.energy_predictor(x)

        if target is not None:
            quantized = quantize(target, self.energy_range, self.energy_dim, toint=True)
            output = self.ER(x, quantized)
            return output, energy_predictor_output
        else:
            quantized = quantize(energy_predictor_output,
                                 self.energy_range, self.energy_dim)
            energy_predictor_output = torch.clamp(
                (quantized * gamma + 0.5).int(), min=0, max=self.energy_dim - 1)

            output = self.ER(x, energy_predictor_output)

            return output
