import torch

from Softmax.initModelParam import num_inputs, W, b
from Softmax.softmax import softmax

# torch.mm 矩阵乘法
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
