import torch.nn as nn
import torch
from torch import nn
import torch

model = nn.Linear(2, 202725) # 输入特征数为2，输出特征数为1

input = torch.Tensor([[1, 2]]) # 给一个样本，该样本有2个特征（这两个特征的值分别为1和2）
print(input.shape)
output = model(input)
print(output)