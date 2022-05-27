from numpy import w

from PyTorch.LinearRegression.creatDataset import features, labels
from PyTorch.LinearRegression.initModelParam import linreg, squared_loss, sgd
from PyTorch.LinearRegression.readDataSet import batch_size
from PyTorch.LinearRegressionEasy.SGD import optimizer
from PyTorch.LinearRegressionEasy.creatDataset import true_w, true_b
from PyTorch.LinearRegressionEasy.defineModel import net
from PyTorch.LinearRegressionEasy.lossFuction import loss
from PyTorch.LinearRegressionEasy.readDataset import data_iter

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)

