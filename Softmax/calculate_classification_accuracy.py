from d2lzh_pytorch import evaluate_accuracy

from Softmax.defineModel import net
from Softmax.getAndReadDataSet import test_iter
from Softmax.lossFuction import y_hat, y


# 给定一个类别的预测概率分布y_hat，我们把预测概率最大的类别作为输出类别。
# 如果它与真实类别y一致，说明这次预测是正确的。
# 分类准确率即正确预测数量与总预测数量之比。
def accuracy(y_hat, y):
    # 其中y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。
    return (y_hat.argmax(dim=1) == y).float().mean().item()


print(accuracy(y_hat, y))

print(test_iter)
print(evaluate_accuracy(test_iter, net))
