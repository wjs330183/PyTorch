import torch
from time import time

a = torch.ones(1000)
# print(a)

b = torch.ones(1000)
# print(b)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)
