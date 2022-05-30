from house_prices_advanced_regression_techniques.K_CrossValidation import k_fold
from house_prices_advanced_regression_techniques.standardization import train_features, train_labels

# 表示单次传递给程序用以训练的参数个数。比如我们的训练集有1000个数据。
# 这是如果我们设置batch_size=100，那么程序首先会用数据集中的前100个参数，即第1-100个数据来训练模型。
# 当训练完成后更新权重，再使用第101-200的个数据训练，直至第十次使用完训练集中的1000个数据后停止。
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 10, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
