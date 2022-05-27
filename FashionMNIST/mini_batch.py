import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
# import d2lzh_pytorch as d2l
from PyTorch.FashionMNIST.getDataSet import mnist_train, mnist_test

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    # 在实践中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。
    # PyTorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取。
    # 这里我们通过参数num_workers来设置4个进程读取数据。
    num_workers = 0
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
# 最后我们查看读取一遍训练数据需要的时间。
print('%.2f sec' % (time.time() - start))
