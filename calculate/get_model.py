import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
# 访问模型参数
print(type(net.named_parameters()))

# 回忆一下上一节中提到的Sequential类与Module类的继承关系。
# 对于Sequential实例中含模型参数的层，我们可以通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回），后者除了返回参数Tensor外还会返回其名字。
# 下面，访问多层感知机net的所有参数：
for name, param in net.named_parameters():
    print(name, param.size())

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name)

weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)  # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)
