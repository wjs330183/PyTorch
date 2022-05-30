import torch
from torch import nn


# 含模型参数的自定义层
# 介绍了Parameter类其实是Tensor的子类，如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。
# 所以在自定义含模型参数的层时，我们应该将参数定义成Parameter，除了像4.2.1节那样直接定义成Parameter类外，还可以使用ParameterList和ParameterDict分别定义参数的列表和字典。
# ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，另外也可以使用append和extend在列表后面新增参数。
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyDense()
print(net)


# 而ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。
# 例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等等，可参考官方文档。
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)
# 这样就可以根据传入的键值来进行不同的前向传播：
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))


net = nn.Sequential(
    MyDictDense(),
    MyDense(),
)
print(net)
print(net(x))
