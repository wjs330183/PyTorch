import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


# 本函数已保存在d2lzh包中方便以后使用
from PyTorch.LinearRegression.creatDataset import features, labels


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
