import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from PyTorch.LinearRegressionEasy.defineModel import net

# sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
# import d2lzh_pytorch as d2l

# 1. torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
# 2. torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
# 3. torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
# 4. torchvision.utils: 其他的一些有用的方法。


mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 通过下标来访问任意一个样本:
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width


# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
