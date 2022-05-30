import torch
from torch import nn

# 我们可以直接使用save函数和load函数分别存储和读取Tensor。
# save使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等。
# 而load使用pickle unpickle工具将pickle的对象文件反序列化为内存。
# 读写Tensor
x = torch.ones(3)
# torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
# torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

# torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


# 读写模型
# 1. state_dict
# 在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。
# state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
nets = net.state_dict()
print(nets)
# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizers = optimizer.state_dict()
print(optimizers)

# 保存和加载模型
# PyTorch中保存和加载训练模型有两种常见的方法:
# 1. 仅保存和加载模型参数(state_dict)；
# 2. 保存和加载整个模型。


X = torch.randn(2, 3)
Y = net(X)

PATH = "./net.pt"
# 1
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)

print(Y)
print(Y2)
print(Y2 == Y)
# 通过save函数和load函数可以很方便地读写Tensor。
# 通过save函数和load_state_dict函数可以很方便地读写模型的参数。
