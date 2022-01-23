from torch import Tensor, tensor, randn, zeros
import torch
import torch.nn as nn

a = torch.tensor(zeros(5), requires_grad=False)

b = nn.Linear(5, 5)

c = b(a)

loss = c.sum()

loss.backward()

if __name__ == '__main__':
    print()
    pass
