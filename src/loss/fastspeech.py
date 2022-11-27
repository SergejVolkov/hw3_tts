import torch
import torch.nn as nn

from ..utils import normalize


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted,
                mel_target, duration_predictor_target, pitch_predictor_target,
                energy_predictor_target):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_target = torch.log(duration_predictor_target.float() + 1)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                                duration_predictor_target)

        pitch_predictor_loss = self.mse_loss(pitch_predicted,
                                             pitch_predictor_target)

        energy_predictor_loss = self.mse_loss(energy_predicted,
                                              energy_predictor_target)

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
