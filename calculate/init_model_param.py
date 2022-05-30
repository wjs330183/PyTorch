import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

for name, param in net.named_parameters():
    if 'weight' in name:
        # torch.init.normal_：给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。
        # torch.init.normal_(tensor,mean=,std=) ,mean:均值，std:正态分布的标准差
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        # torch.init.constant_:初始化参数使其为常值，即每个参数值都相同。一般是给网络中bias进行初始化。
        # torch.nn.init.constant_(tensor,val),val:常量数值
        init.constant_(param, val=0)
        print(name, param.data)


# 自定义初始化方法
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def init_weight_(tensor):
    # with是python中上下文管理器，简单理解，当要进行固定的进入，返回操作时，可以将对应需要的操作，放在with所需要的语句中。
    # 比如文件的写入（需要打开关闭文件）等。
    # with后部分，可以将with后的语句运行，将其返回结果给到as后的变量（sh），之后的代码块对close进行操作。
    # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
    # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
    with torch.no_grad():
        # tensor从均匀分布中抽样数值进行填充。
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
