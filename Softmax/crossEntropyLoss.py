from torch import nn


def loss():
    loss1 = nn.CrossEntropyLoss()
    loss1.requires_grad_(True)
    return loss1
