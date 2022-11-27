import torch


def normalize(x, range):
    return (x - range[0]) / (range[1] - range[0])


def quantize(x, range, dim, toint = False):
    output = normalize(x, range) * (dim - 1)

    if toint:
        output = (output + 0.5).int()
    else:
        output = torch.round(output)
    return torch.clamp(output, min=0, max=dim - 1)
