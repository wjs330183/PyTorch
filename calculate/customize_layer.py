import torch
from torch import nn


# 定义一个不含模型参数的自定义层。
# 事实上，这和4.1节（模型构造）中介绍的使用Module类构造模型类似。
# 下面的CenteredLayer类通过继承Module类自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了forward函数里。
# 这个层里不含模型参数。
class CenteredLayer(nn.Module):
    # 1、*args和**kwargs主要用于定义函数的可变参数

    # 2、*args：发送一个非键值对的可变数量的参数列表给函数

    # 3、**kwargs：发送一个键值对的可变数量的参数列表给函数

    # 4、如果想要在函数内使用带有名称的变量（像字典那样），那么使用**kwargs。

    # 定义可变参数的目的是为了简化调用。

    # *和**在此处的作用：打包参数。
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# 我们可以实例化这个层，然后做前向计算。
layer = CenteredLayer()
layers = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
print(layers)

# 我们也可以用它来构造更复杂的模型。
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# 下面打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数。
y = net(torch.rand(4, 8))
ys = y.mean().item()
print(float(ys))


