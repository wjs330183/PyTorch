# 如果没有安装pandas，则反注释下面一行
# !pip install pandas

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
train_data = pd.read_csv('../kaggle_house/train.csv')
test_data = pd.read_csv('../kaggle_house/test.csv')
# 输出 (1460, 81)
print(train_data.shape)
# 输出 (1459, 80)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features)
