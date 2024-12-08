import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot

from torch_geometric.nn import GATConv

class SimplePromptAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int = 3):
        super(SimplePromptAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.attention = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_list)
        nn.init.xavier_uniform_(self.attention.weight)

    def add(self, x: torch.Tensor):
        # 计算注意力权重
        attention_scores = F.softmax(self.attention(x), dim=1)  # [N, p_num]
        # 使用注意力权重对 p_list 进行加权求和，生成每个节点的提示向量 p_i
        p = torch.matmul(attention_scores, self.p_list)  # [N, in_channels]
        # 返回输入特征加上生成的提示向量
        return x + p