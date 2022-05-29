from d2lzh_pytorch import train_ch3

from Softmax.crossEntropyLoss import loss
from Softmax.defineModel import net
from Softmax.getAndReadDataSet import train_iter, test_iter, batch_size
from Softmax.initModelParam import b, W
from Softmax.lossFuction import cross_entropy
from Softmax.softmaxSGD import optimizer

num_epochs, lr = 10, 0.1
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
