import torch
import numpy as np


def synthesis(model, phn, device, alpha = 1.0, beta = 1.0, gamma = 1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
