import torch
from torch import nn
from torch.nn import init

linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
# 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的:
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)  # 单次梯度是3，两次所以就是6
