from PyTorch.LinearRegressionEasy.SGD import optimizer
from PyTorch.LinearRegressionEasy.creatDataset import true_w, true_b
from PyTorch.LinearRegressionEasy.defineModel import net
from PyTorch.LinearRegressionEasy.lossFuction import loss
from PyTorch.LinearRegressionEasy.readDataset import data_iter

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
