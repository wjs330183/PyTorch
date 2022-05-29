import torch

from Softmax.defineAndInitModel import net

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
