import matplotlib.pyplot as plt
import pandas as pd

from house_prices_advanced_regression_techniques.getAndReadDataset_HPART import test_data
from house_prices_advanced_regression_techniques.select_kg_model import num_epochs, lr, weight_decay, batch_size
from house_prices_advanced_regression_techniques.standardization import train_features, test_features, train_labels
from house_prices_advanced_regression_techniques.trainModle import get_net, train

import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)
    plt.show()

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
