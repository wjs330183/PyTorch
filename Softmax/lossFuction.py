import torch

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


# 下面实现了3.4节（softmax回归）中介绍的交叉熵损失函数。
def cross_entropy(y_hat, y):
    loss = - torch.log(y_hat.gather(1, y.view(-1, 1)))
    loss.requires_grad_(True)
    return loss
