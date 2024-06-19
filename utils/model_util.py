import torch.nn as nn


def get_linear_layers_w_activations(
        layers, activation=nn.ReLU(), activation_at_end=False
    ):
    if len(layers) == 0:
        return []
    output = []
    for i in range(len(layers)-1):
        output.append(activation)
        output.append(nn.Linear(layers[i], layers[i+1]))
    if activation_at_end:
        output.append(activation)
    return output
