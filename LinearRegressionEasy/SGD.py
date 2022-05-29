import torch.optim as optim

from LinearRegressionEasy.defineModelEasy import net

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},  # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
# ], lr=0.03)

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率为之前的0.1倍
