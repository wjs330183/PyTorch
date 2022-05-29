import numpy as np
import torch

# 跟线性回归中的例子一样，我们将使用向量表示每个样本。
# 已知每个样本输入是高和宽均为28像素的图像。
# 模型的输入向量的长度是 28×28=78428×28=784：该向量的每个元素对应图像中每个像素。
# 由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784×10784×10和1×101×10的矩阵。
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
