import torch

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))

# 在下面的函数中，矩阵X的行数是样本数，列数是输出个数。
# 为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
# 这样一来，最终得到的矩阵每行元素和为1且非负。
# 因此，该矩阵每行都是合法的概率分布。
# softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))
